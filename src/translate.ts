/**
 * Translate a Bedrock ConverseStream event stream into the harness
 * `StreamChunk` protocol.
 *
 * Bedrock keys every block by its own `contentBlockIndex` and delivers a
 * `contentBlockStart` / `contentBlockDelta`* / `contentBlockStop` sequence per
 * block, then a `messageStop` carrying the stop reason and a `metadata` event
 * carrying usage. This module keeps one open harness block per Bedrock index,
 * maps deltas as they arrive, and defers every `block-end`, the `usage`, and
 * the terminal `finish` to the end of the stream so nothing follows `finish`.
 *
 * A reasoning block additionally accumulates the opaque `signature` Bedrock
 * emits, and the finish chunk carries those signatures back as replay state so
 * a later turn can rebuild the thinking block (see {@link module:dsh-llm-bedrock/replay}).
 *
 * @module dsh-llm-bedrock/translate
 */

import { CallId, EMPTY_RESPONSE_CODE, LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, FinishReason, StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm'
import { REASONING_SIGNATURES_KEY } from './replay.js'
import type { BedrockTokenUsage, ConverseStreamOutput, StopReason } from './types.js'

/** One harness block under assembly, mapped from one Bedrock content-block index. */
interface OpenBlock {
  /** Harness stream index (dense, assigned first-seen). */
  index: number
  kind: 'text' | 'reasoning' | 'tool-call'
  text: string
  /** reasoning only: the Bedrock-issued signature accumulated across deltas. */
  signature?: string
  /** tool-call only */
  callId?: string
  name?: string
}

/**
 * Map a Bedrock stop reason to the harness FinishReason.
 * @param reason - the Bedrock `stopReason`, when present.
 * @returns the mapped reason; content-filter and guardrail stops become errors.
 */
export function mapStopReason(reason: StopReason | undefined): FinishReason {
  switch (reason) {
    case 'end_turn':
    case 'stop_sequence':
      return { kind: 'stop' }
    case 'tool_use':
      return { kind: 'tool-calls' }
    case 'max_tokens':
      return { kind: 'max-tokens' }
    case undefined:
      return { kind: 'stop' }
    default:
      // content_filtered, guardrail_intervened, future additions.
      return {
        kind: 'error',
        failure: { message: `model stopped: ${reason}`, code: String(reason).toUpperCase() },
      }
  }
}

/**
 * Map Bedrock usage onto the harness disjoint token convention. Bedrock's
 * `inputTokens` already excludes cache reads (they are reported separately as
 * `cacheReadInputTokens` / `cacheWriteInputTokens`), so no subtraction is
 * needed, unlike the DeepSeek adapter.
 * @param usage - Bedrock usage from the `metadata` event.
 * @returns disjoint harness counts; cache fields present only when reported.
 */
export function mapUsage(usage: BedrockTokenUsage): TokenUsage {
  const cacheRead = usage.cacheReadInputTokens
  const cacheWrite = usage.cacheWriteInputTokens
  return {
    inputTokens: usage.inputTokens ?? 0,
    outputTokens: usage.outputTokens ?? 0,
    ...cacheRead !== undefined && cacheRead !== null ? { cacheReadTokens: cacheRead } : {},
    ...cacheWrite !== undefined && cacheWrite !== null ? { cacheWriteTokens: cacheWrite } : {},
  }
}

/** Assemble the final harness ContentBlock for one open block. */
function closeBlock(block: OpenBlock): ContentBlock {
  switch (block.kind) {
    case 'text':
      return { type: 'text', text: block.text }
    case 'reasoning':
      return { type: 'reasoning', text: block.text }
    case 'tool-call':
      return {
        type: 'tool-call',
        id: CallId(block.callId ?? ''),
        name: block.name ?? '',
        arguments: block.text,
      }
  }
}

/**
 * Consume the SDK's async iterable of ConverseStream events and yield harness
 * StreamChunks. A degenerate completion (a `stop` finish with no content
 * blocks) maps to an `EMPTY_RESPONSE` error finish, matching the DeepSeek
 * adapter's contract.
 * @param events - the SDK `ConverseStreamCommandOutput.stream` async iterable.
 * @returns the harness chunk stream.
 */
export async function* translate(
  events: AsyncIterable<ConverseStreamOutput>,
): AsyncGenerator<StreamChunk> {
  let nextIndex = 0
  const byBedrockIndex = new Map<number, OpenBlock>()
  const order: OpenBlock[] = []
  let pendingFinish: FinishReason | undefined
  let pendingUsage: TokenUsage | undefined

  const open = (bedrockIndex: number, kind: OpenBlock['kind']): OpenBlock => {
    const block: OpenBlock = { index: nextIndex++, kind, text: '' }
    byBedrockIndex.set(bedrockIndex, block)
    order.push(block)
    return block
  }

  for await (const event of events) {
    if (event.contentBlockStart !== undefined) {
      const bedrockIndex = event.contentBlockStart.contentBlockIndex ?? 0
      const start = event.contentBlockStart.start
      if (start?.toolUse !== undefined) {
        const block = open(bedrockIndex, 'tool-call')
        if (start.toolUse.toolUseId !== undefined) block.callId = start.toolUse.toolUseId
        if (start.toolUse.name !== undefined) block.name = start.toolUse.name
        yield { type: 'block-start', index: block.index, blockType: 'tool-call' }
        yield {
          type: 'tool-call-delta',
          index: block.index,
          id: CallId(block.callId ?? ''),
          ...block.name !== undefined ? { name: block.name } : {},
          argumentsDelta: '',
        }
      }
      continue
    }

    if (event.contentBlockDelta !== undefined) {
      const bedrockIndex = event.contentBlockDelta.contentBlockIndex ?? 0
      const delta = event.contentBlockDelta.delta

      if (typeof delta?.text === 'string' && delta.text.length > 0) {
        let block = byBedrockIndex.get(bedrockIndex)
        if (block === undefined) {
          block = open(bedrockIndex, 'text')
          yield { type: 'block-start', index: block.index, blockType: 'text' }
        }
        block.text += delta.text
        yield { type: 'text-delta', index: block.index, text: delta.text }
        continue
      }

      const reasoning = delta?.reasoningContent
      if (reasoning !== undefined) {
        let block = byBedrockIndex.get(bedrockIndex)
        if (block === undefined) {
          block = open(bedrockIndex, 'reasoning')
          yield { type: 'block-start', index: block.index, blockType: 'reasoning' }
        }
        if (typeof reasoning.text === 'string' && reasoning.text.length > 0) {
          block.text += reasoning.text
          yield { type: 'reasoning-delta', index: block.index, text: reasoning.text }
        }
        // The signature arrives as its own delta near the block's end; keep it
        // for replay state without emitting a harness chunk (it is not text).
        if (typeof reasoning.signature === 'string' && reasoning.signature.length > 0) {
          block.signature = (block.signature ?? '') + reasoning.signature
        }
        continue
      }

      const toolUse = delta?.toolUse
      if (toolUse !== undefined && typeof toolUse.input === 'string') {
        const block = byBedrockIndex.get(bedrockIndex)
        if (block !== undefined && block.kind === 'tool-call') {
          block.text += toolUse.input
          yield {
            type: 'tool-call-delta',
            index: block.index,
            id: CallId(block.callId ?? ''),
            ...block.name !== undefined ? { name: block.name } : {},
            argumentsDelta: toolUse.input,
          }
        }
      }
      continue
    }

    if (event.messageStop !== undefined) {
      pendingFinish = mapStopReason(event.messageStop.stopReason)
      continue
    }

    if (event.metadata !== undefined && event.metadata.usage !== undefined) {
      pendingUsage = mapUsage(event.metadata.usage)
      continue
    }

    // messageStart and future event kinds carry nothing the harness maps.
  }

  for (const block of order) {
    yield { type: 'block-end', index: block.index, block: closeBlock(block) }
  }
  if (pendingUsage !== undefined) yield { type: 'usage', usage: pendingUsage }

  const reason = pendingFinish ?? { kind: 'stop' as const }
  if (reason.kind === 'stop' && order.length === 0) {
    yield {
      type: 'finish',
      reason: {
        kind: 'error',
        failure: { message: 'model returned a completed response with no content', code: EMPTY_RESPONSE_CODE },
      },
    }
    return
  }

  // Carry reasoning signatures back for replay, one entry per reasoning block
  // in the order the harness assembles them into the assistant message.
  const signatures = order
    .filter(block => block.kind === 'reasoning')
    .map(block => block.signature)
  yield {
    type: 'finish',
    reason,
    ...signatures.length > 0
      ? { replayState: { response: { [REASONING_SIGNATURES_KEY]: signatures } } }
      : {},
  }
}
