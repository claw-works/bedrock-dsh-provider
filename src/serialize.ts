/**
 * Serialize harness messages into a Bedrock ConverseStream request.
 *
 * The Converse API differs from an OpenAI-compatible wire in three ways this
 * module bridges:
 *  - the system prompt is its own top-level `system` field, not a message;
 *  - tool results ride inside a `user` message as `toolResult` content blocks,
 *    keyed by the tool-use id, rather than a separate `tool` role; and
 *  - consecutive same-role turns must be merged, because Converse requires the
 *    conversation to strictly alternate user / assistant.
 *
 * Claude extended-thinking replay needs the opaque `signature` Bedrock issued
 * with each reasoning block; the adapter carries those signatures out-of-band
 * on the assembled message's replay state (see {@link module:dsh-llm-bedrock/adapter}),
 * and {@link serializeMessages} rebuilds the `reasoningContent` block only when
 * the signature is present. Without it, the reasoning block is dropped from
 * replay — safe, because Bedrock rejects a thinking block with no signature.
 *
 * @module dsh-llm-bedrock/serialize
 */

import { LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock as HarnessBlock, GenerateOptions, Message } from '@deepseek-ai/dsh-llm'
import type { ModelCapabilities } from './model.ts'
import { REASONING_SIGNATURES_KEY, type ReasoningSignatures } from './replay.ts'
import type {
  ContentBlock,
  ConverseStreamCommandInput,
  BedrockMessage,
  DocumentType,
  SystemContentBlock,
  Tool,
  ToolConfiguration,
} from './types.ts'

/** Adapter-level request defaults derived from plugin config. */
export interface RequestDefaults {
  /** Token budget for the Claude thinking channel when reasoning is on. */
  thinkingBudgetTokens: number
}

/** Read the reasoning signatures an assistant message carried on its replay state. */
function reasoningSignatures(message: Message): ReasoningSignatures | undefined {
  if (message.source.kind !== 'model') return undefined
  const replay = message.source.replayState
  if (replay === null || typeof replay !== 'object') return undefined
  // The harness stores the whole ReplayEnvelope ({ response, blocks? }); the
  // adapter writes signatures under the response half.
  const response = (replay as Record<string, unknown>).response
  if (response === null || typeof response !== 'object') return undefined
  const signatures = (response as Record<string, unknown>)[REASONING_SIGNATURES_KEY]
  return Array.isArray(signatures) ? (signatures as ReasoningSignatures) : undefined
}

/** Flatten the text blocks of a harness message into one string. */
function flattenText(blocks: readonly HarnessBlock[]): string {
  return blocks
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')
}

/** Reject image content: this first version maps text, tool, and reasoning only. */
function assertNoImages(blocks: readonly HarnessBlock[]): void {
  if (blocks.some(block => block.type === 'image')) {
    throw new LlmError('The Bedrock adapter does not support image content yet.', 'UNSUPPORTED_CONTENT')
  }
}

/** Build the Bedrock content blocks for one assistant turn. */
function assistantContent(message: Message): ContentBlock[] {
  const blocks: ContentBlock[] = []
  const signatures = reasoningSignatures(message)
  let reasoningIndex = 0
  for (const block of message.content) {
    switch (block.type) {
      case 'reasoning': {
        // Replay a thinking block only with its Bedrock-issued signature;
        // the position in the assistant content selects the matching entry.
        const signature = signatures?.[reasoningIndex]
        reasoningIndex += 1
        if (signature === undefined || block.text.length === 0) break
        blocks.push({
          reasoningContent: { reasoningText: { text: block.text, signature } },
        })
        break
      }
      case 'text':
        if (block.text.length > 0) blocks.push({ text: block.text })
        break
      case 'tool-call':
        blocks.push({
          toolUse: {
            toolUseId: block.id,
            name: block.name,
            // Converse wants a parsed JSON object; the harness holds the raw
            // model string. Empty arguments become an empty object.
            input: parseToolInput(block.arguments),
          },
        })
        break
      default:
        break
    }
  }
  return blocks
}

/** Parse a raw model tool-argument string into the object Converse expects. */
function parseToolInput(raw: string): DocumentType {
  const trimmed = raw.trim()
  if (trimmed.length === 0) return {}
  try {
    return JSON.parse(trimmed) as DocumentType
  } catch {
    // A mid-stream truncation can leave invalid JSON; forward it as a string
    // rather than fail the whole replay, letting the model see what it emitted.
    return { __raw: raw }
  }
}

/** Build the Bedrock content blocks contributed by one user-role harness turn. */
function userContent(message: Message): ContentBlock[] {
  const blocks: ContentBlock[] = []
  const text = flattenText(message.content)
  if (text.length > 0) blocks.push({ text })
  for (const block of message.content) {
    if (block.type !== 'tool-result') continue
    const resultText = flattenText(block.content)
    blocks.push({
      toolResult: {
        toolUseId: block.toolCallId,
        content: [{ text: resultText.length > 0 ? resultText : '(no output)' }],
        ...block.isError === true ? { status: 'error' as const } : {},
      },
    })
  }
  return blocks
}

/**
 * Convert the harness conversation to Bedrock messages, merging consecutive
 * same-role turns so the result strictly alternates user / assistant.
 * @param messages - the harness conversation, in order.
 * @returns Bedrock messages ready for the Converse `messages` field.
 */
export function serializeMessages(messages: readonly Message[]): BedrockMessage[] {
  const wire: BedrockMessage[] = []
  for (const message of messages) {
    assertNoImages(message.content)
    // A system-role message here would be unusual (the system prompt travels
    // in GenerateOptions.system), but map it as a leading user note if present.
    const role: 'user' | 'assistant' = message.role === 'assistant' ? 'assistant' : 'user'
    const content = role === 'assistant' ? assistantContent(message) : userContent(message)
    if (content.length === 0) continue
    const last = wire[wire.length - 1]
    if (last !== undefined && last.role === role) {
      last.content = [...(last.content ?? []), ...content]
    } else {
      wire.push({ role, content })
    }
  }
  return wire
}

/** Map the harness tool schemas onto a Bedrock `toolConfig`. */
function toolConfig(options: GenerateOptions): ToolConfiguration | undefined {
  if (options.tools === undefined || options.tools.length === 0) return undefined
  const tools: Tool[] = options.tools.map(tool => ({
    toolSpec: {
      name: tool.name,
      description: tool.description,
      inputSchema: { json: tool.parameters as DocumentType },
    },
  }))
  return { tools }
}

/**
 * Build the full ConverseStream request body.
 * @param options - the harness request (model, history, system, tools, sampling).
 * @param capabilities - the resolved capabilities of `options.model`.
 * @param defaults - adapter-level request defaults (thinking budget).
 * @returns the ConverseStream command input.
 */
export function serializeRequest(
  options: GenerateOptions,
  capabilities: ModelCapabilities,
  defaults: RequestDefaults,
): ConverseStreamCommandInput {
  const system: SystemContentBlock[] | undefined = options.system === undefined
    ? undefined
    : [{ text: options.system }]

  const inferenceConfig = {
    ...options.maxTokens === undefined ? {} : { maxTokens: options.maxTokens },
    ...options.temperature === undefined ? {} : { temperature: options.temperature },
    ...options.stop === undefined ? {} : { stopSequences: options.stop },
  }

  const config = capabilities.tools ? toolConfig(options) : undefined
  const thinking = resolveThinking(options, capabilities, defaults)

  return {
    modelId: options.model,
    messages: serializeMessages(options.messages),
    ...system === undefined ? {} : { system },
    ...config === undefined ? {} : { toolConfig: config },
    ...Object.keys(inferenceConfig).length > 0 ? { inferenceConfig } : {},
    ...thinking === undefined ? {} : { additionalModelRequestFields: thinking },
  }
}

/**
 * Resolve the `additionalModelRequestFields` that enable Claude extended
 * thinking, or `undefined` when reasoning is off or unsupported. Bedrock
 * requires the sampling temperature to be unset (it forces 1.0) while thinking
 * is on; the caller-supplied temperature is simply not sent in that case, which
 * `serializeRequest` already does by leaving `inferenceConfig.temperature`
 * absent whenever the caller omits it — a caller that both sets temperature and
 * requests thinking gets a provider-side rejection that names the conflict.
 * @param options - the harness request.
 * @param capabilities - resolved capabilities of the model.
 * @param defaults - adapter defaults carrying the thinking token budget.
 * @returns the request-fields object, or undefined when thinking stays off.
 */
function resolveThinking(
  options: GenerateOptions,
  capabilities: ModelCapabilities,
  defaults: RequestDefaults,
): DocumentType | undefined {
  if (capabilities.reasoning !== 'claude-thinking') return undefined
  if (options.purpose === 'session-title') return undefined
  const effort = options.reasoningEffort
  if (effort === undefined || effort === 'off') return undefined
  return {
    thinking: { type: 'enabled', budget_tokens: defaults.thinkingBudgetTokens },
  }
}
