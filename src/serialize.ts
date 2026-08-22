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
import type { ImageAttachmentRef, ImageMediaType } from '@deepseek-ai/dsh-attachment'
import type { ModelCapabilities } from './model.js'
import { REASONING_SIGNATURES_KEY, type ReasoningSignatures } from './replay.js'
import type {
  ContentBlock,
  ConverseStreamCommandInput,
  BedrockMessage,
  DocumentType,
  ImageFormat,
  SystemContentBlock,
  Tool,
  ToolConfiguration,
} from './types.js'

/** Adapter-level request defaults derived from plugin config. */
export interface RequestDefaults {
  /** Token budget for the Claude thinking channel when reasoning is on. */
  thinkingBudgetTokens: number
}

/**
 * Read the raw bytes for one durable image reference through the attachment
 * seam (`ctx.attachments.readImage`). Serialization stays decoupled from cordis
 * by taking this thunk rather than the whole store; the adapter supplies it.
 * @param ref - the durable image reference carried on a harness image block.
 * @returns the verified encoded image bytes.
 */
export type ImageReader = (ref: ImageAttachmentRef) => Promise<Uint8Array>

/**
 * Map a harness attachment media type onto the Bedrock Converse image format.
 * The two vocabularies are 1:1 for the version-one raster set; an unrecognised
 * media type is a programming error against {@link ImageMediaType}, surfaced as
 * an explicit mapping failure rather than a silently dropped block.
 */
const IMAGE_FORMAT_BY_MEDIA_TYPE: Readonly<Record<ImageMediaType, ImageFormat>> = {
  'image/png': 'png',
  'image/jpeg': 'jpeg',
  'image/gif': 'gif',
  'image/webp': 'webp',
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

/**
 * Build the Bedrock image content block for one harness image block, reading
 * its bytes through the attachment seam. The caller has already established the
 * model accepts images; this maps the media type to a Converse `format` and
 * carries the raw bytes inline (the AWS SDK base64-encodes them on the wire).
 * @param block - the harness image block.
 * @param readImage - the attachment-seam byte reader.
 * @returns the Converse image content block.
 */
async function imageContent(
  block: Extract<HarnessBlock, { type: 'image' }>,
  readImage: ImageReader,
): Promise<ContentBlock> {
  // Index through a Partial view so an out-of-vocabulary media type (a widened
  // ImageMediaType from a future harness) reaches the runtime guard below
  // rather than being assumed present by the type.
  const format = (IMAGE_FORMAT_BY_MEDIA_TYPE as Partial<Record<string, ImageFormat>>)[block.attachment.mediaType]
  if (format === undefined) {
    throw new LlmError(
      `The Bedrock adapter cannot map image media type "${block.attachment.mediaType}" to a Converse format.`,
      'UNSUPPORTED_CONTENT',
    )
  }
  const bytes = await readImage(block.attachment)
  return { image: { format, source: { bytes } } }
}

/**
 * Reject image content on a model whose family does not accept images. The
 * error names both the model id and the resolved family so an operator can see
 * why a request that carried an image was refused.
 * @param capabilities - the resolved capabilities of the request model.
 * @param modelId - the model id the request targets.
 * @throws LlmError('UNSUPPORTED_CONTENT') when the family cannot take images.
 */
function assertImagesAllowed(capabilities: ModelCapabilities, modelId: string): void {
  if (capabilities.images) return
  throw new LlmError(
    `The Bedrock model "${modelId}" (family "${capabilities.family}") does not accept image content.`,
    'UNSUPPORTED_CONTENT',
  )
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
async function userContent(message: Message, readImage: ImageReader): Promise<ContentBlock[]> {
  const blocks: ContentBlock[] = []
  const text = flattenText(message.content)
  if (text.length > 0) blocks.push({ text })
  for (const block of message.content) {
    if (block.type === 'image') {
      blocks.push(await imageContent(block, readImage))
      continue
    }
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
 * same-role turns so the result strictly alternates user / assistant. Image
 * content is mapped through the attachment seam when the model accepts images,
 * and rejected with a clear error otherwise.
 * @param messages - the harness conversation, in order.
 * @param capabilities - resolved capabilities of the request model.
 * @param modelId - the model id the request targets (named in image errors).
 * @param readImage - attachment-seam byte reader for durable image references.
 * @returns Bedrock messages ready for the Converse `messages` field.
 */
export async function serializeMessages(
  messages: readonly Message[],
  capabilities: ModelCapabilities,
  modelId: string,
  readImage: ImageReader,
): Promise<BedrockMessage[]> {
  const wire: BedrockMessage[] = []
  for (const message of messages) {
    if (message.content.some(block => block.type === 'image')) {
      assertImagesAllowed(capabilities, modelId)
    }
    // A system-role message here would be unusual (the system prompt travels
    // in GenerateOptions.system), but map it as a leading user note if present.
    const role: 'user' | 'assistant' = message.role === 'assistant' ? 'assistant' : 'user'
    const content = role === 'assistant' ? assistantContent(message) : await userContent(message, readImage)
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
 * @param readImage - attachment-seam byte reader for durable image references.
 * @returns the ConverseStream command input.
 */
export async function serializeRequest(
  options: GenerateOptions,
  capabilities: ModelCapabilities,
  defaults: RequestDefaults,
  readImage: ImageReader,
): Promise<ConverseStreamCommandInput> {
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
  const messages = await serializeMessages(options.messages, capabilities, options.model, readImage)

  return {
    modelId: options.model,
    messages,
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
