/**
 * Offline unit tests for `src/translate.ts`: the Bedrock ConverseStream event
 * stream → harness `StreamChunk` protocol. Events are hand-built plain objects
 * cast to the SDK union (each variant sets exactly one key), so no AWS access
 * is needed. Covers block start/delta/end, usage, finish, stopReason mapping,
 * reasoning-signature replay state, and the empty-response degenerate case.
 */

import { describe, expect, it } from 'vitest'
import { mapStopReason, mapUsage, translate } from '../src/translate.js'
import type { StreamChunk } from '@deepseek-ai/dsh-llm'
import type { BedrockTokenUsage, ConverseStreamOutput, StopReason } from '../src/types.js'

/** Build an async iterable from a fixed list of events. */
async function* stream(events: ConverseStreamOutput[]): AsyncGenerator<ConverseStreamOutput> {
  for (const event of events) yield event
}

/** Drain translate() into an array of chunks. */
async function collect(events: ConverseStreamOutput[]): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of translate(stream(events))) chunks.push(chunk)
  return chunks
}

/** Cast a plain event literal to the SDK union member. */
function ev(e: Record<string, unknown>): ConverseStreamOutput {
  return e as unknown as ConverseStreamOutput
}

describe('mapStopReason', () => {
  it('maps normal terminations to stop / tool-calls / max-tokens', () => {
    expect(mapStopReason('end_turn')).toEqual({ kind: 'stop' })
    expect(mapStopReason('stop_sequence')).toEqual({ kind: 'stop' })
    expect(mapStopReason('tool_use')).toEqual({ kind: 'tool-calls' })
    expect(mapStopReason('max_tokens')).toEqual({ kind: 'max-tokens' })
    expect(mapStopReason(undefined)).toEqual({ kind: 'stop' })
  })

  it('maps content-filter / guardrail stops to an error finish', () => {
    const filtered = mapStopReason('content_filtered' as StopReason)
    expect(filtered.kind).toBe('error')
    expect(filtered).toMatchObject({ kind: 'error', failure: { code: 'CONTENT_FILTERED' } })
    const guardrail = mapStopReason('guardrail_intervened' as StopReason)
    expect(guardrail).toMatchObject({ kind: 'error', failure: { code: 'GUARDRAIL_INTERVENED' } })
  })
})

describe('mapUsage', () => {
  it('maps input/output tokens and omits absent cache fields', () => {
    expect(mapUsage({ inputTokens: 10, outputTokens: 20 } as BedrockTokenUsage)).toEqual({
      inputTokens: 10, outputTokens: 20,
    })
  })

  it('carries cache read/write only when reported', () => {
    expect(mapUsage({ inputTokens: 5, outputTokens: 6, cacheReadInputTokens: 3, cacheWriteInputTokens: 2 } as BedrockTokenUsage)).toEqual({
      inputTokens: 5, outputTokens: 6, cacheReadTokens: 3, cacheWriteTokens: 2,
    })
  })

  it('defaults missing token counts to zero', () => {
    expect(mapUsage({} as BedrockTokenUsage)).toEqual({ inputTokens: 0, outputTokens: 0 })
  })
})

describe('translate — text stream', () => {
  it('emits block-start, text-deltas, deferred block-end, usage, then finish', async () => {
    const chunks = await collect([
      ev({ contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'Hel' } } }),
      ev({ contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'lo' } } }),
      ev({ contentBlockStop: { contentBlockIndex: 0 } }),
      ev({ messageStop: { stopReason: 'end_turn' } }),
      ev({ metadata: { usage: { inputTokens: 4, outputTokens: 2 } } }),
    ])
    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'Hel' },
      { type: 'text-delta', index: 0, text: 'lo' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello' } },
      { type: 'usage', usage: { inputTokens: 4, outputTokens: 2 } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])
  })
})

describe('translate — tool-call stream', () => {
  it('opens a tool-call on contentBlockStart and accumulates argument deltas', async () => {
    const chunks = await collect([
      ev({ contentBlockStart: { contentBlockIndex: 0, start: { toolUse: { toolUseId: 't1', name: 'search' } } } }),
      ev({ contentBlockDelta: { contentBlockIndex: 0, delta: { toolUse: { input: '{"q":' } } } }),
      ev({ contentBlockDelta: { contentBlockIndex: 0, delta: { toolUse: { input: '"cats"}' } } } }),
      ev({ contentBlockStop: { contentBlockIndex: 0 } }),
      ev({ messageStop: { stopReason: 'tool_use' } }),
    ])
    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'tool-call' },
      { type: 'tool-call-delta', index: 0, id: 't1', name: 'search', argumentsDelta: '' },
      { type: 'tool-call-delta', index: 0, id: 't1', name: 'search', argumentsDelta: '{"q":' },
      { type: 'tool-call-delta', index: 0, id: 't1', name: 'search', argumentsDelta: '"cats"}' },
      { type: 'block-end', index: 0, block: { type: 'tool-call', id: 't1', name: 'search', arguments: '{"q":"cats"}' } },
      { type: 'finish', reason: { kind: 'tool-calls' } },
    ])
  })
})

describe('translate — reasoning stream', () => {
  it('emits reasoning-deltas, accumulates the signature off-band, and carries it as replay state', async () => {
    const chunks = await collect([
      ev({ contentBlockDelta: { contentBlockIndex: 0, delta: { reasoningContent: { text: 'let me think' } } } }),
      ev({ contentBlockDelta: { contentBlockIndex: 0, delta: { reasoningContent: { signature: 'sig-1' } } } }),
      ev({ contentBlockStop: { contentBlockIndex: 0 } }),
      ev({ contentBlockDelta: { contentBlockIndex: 1, delta: { text: 'answer' } } }),
      ev({ contentBlockStop: { contentBlockIndex: 1 } }),
      ev({ messageStop: { stopReason: 'end_turn' } }),
    ])
    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'reasoning' },
      { type: 'reasoning-delta', index: 0, text: 'let me think' },
      { type: 'block-start', index: 1, blockType: 'text' },
      { type: 'text-delta', index: 1, text: 'answer' },
      { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'let me think' } },
      { type: 'block-end', index: 1, block: { type: 'text', text: 'answer' } },
      { type: 'finish', reason: { kind: 'stop' }, replayState: { response: { bedrockReasoningSignatures: ['sig-1'] } } },
    ])
  })
})

describe('translate — max-tokens and multi-block ordering', () => {
  it('maps max_tokens and defers every block-end to after the last delta', async () => {
    const chunks = await collect([
      ev({ contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'a' } } }),
      ev({ contentBlockDelta: { contentBlockIndex: 1, delta: { text: 'b' } } }),
      ev({ messageStop: { stopReason: 'max_tokens' } }),
    ])
    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'a' },
      { type: 'block-start', index: 1, blockType: 'text' },
      { type: 'text-delta', index: 1, text: 'b' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'a' } },
      { type: 'block-end', index: 1, block: { type: 'text', text: 'b' } },
      { type: 'finish', reason: { kind: 'max-tokens' } },
    ])
  })
})

describe('translate — empty response', () => {
  it('maps a stop finish with no content blocks to an EMPTY_RESPONSE error finish', async () => {
    const chunks = await collect([ev({ messageStop: { stopReason: 'end_turn' } })])
    expect(chunks).toHaveLength(1)
    expect(chunks[0]).toMatchObject({
      type: 'finish',
      reason: { kind: 'error', failure: { code: 'EMPTY_RESPONSE' } },
    })
  })

  it('does NOT treat a non-empty stop as empty', async () => {
    const chunks = await collect([
      ev({ contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'hi' } } }),
      ev({ messageStop: { stopReason: 'end_turn' } }),
    ])
    expect(chunks.at(-1)).toEqual({ type: 'finish', reason: { kind: 'stop' } })
  })
})
