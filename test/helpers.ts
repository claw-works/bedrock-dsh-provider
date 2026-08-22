/**
 * Shared fixtures for the Bedrock adapter's offline unit tests. Builds harness
 * `Message` / `GenerateOptions` values and fake attachment refs without any
 * cordis context or AWS access, so every suite that imports this runs green in
 * CI with no credentials.
 *
 * @module dsh-llm-bedrock/test/helpers
 */

import { CallId, ReasoningEffortId, createAssistantMessage, createToolResultMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message } from '@deepseek-ai/dsh-llm'
import type { ImageAttachmentRef, ImageMediaType } from '@deepseek-ai/dsh-attachment'
import type { AttachmentId } from '@deepseek-ai/dsh-attachment'
import type { ImageReader, RequestDefaults } from '../src/serialize.js'

/** A plain user turn carrying only text. */
export function userText(text: string): Message {
  return createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })
}

/** A user turn carrying arbitrary content blocks (text, image, ...). */
export function userBlocks(content: ContentBlock[]): Message {
  return createUserMessage({ content, source: { kind: 'user' } })
}

/** A model assistant turn carrying arbitrary content blocks. */
export function assistant(content: ContentBlock[], replayState?: unknown): Message {
  return createAssistantMessage({
    content,
    source: {
      provider: 'bedrock',
      model: 'test',
      ...replayState === undefined ? {} : { replayState },
    },
  })
}

/** A tool-result user turn correlated to a prior tool call. */
export function toolResult(callId: string, text: string, isError = false): Message {
  return createToolResultMessage({
    callId: CallId(callId),
    content: [{ type: 'text', text }],
    isError,
  })
}

/** Build a durable image reference fixture. */
export function imageRef(mediaType: ImageMediaType, name = 'img'): ImageAttachmentRef {
  return {
    attachmentId: 'att-1' as AttachmentId,
    mediaType,
    bytes: 4,
    width: 2,
    height: 2,
    name,
  }
}

/** An image content block referencing a fixture attachment. */
export function imageBlock(mediaType: ImageMediaType): ContentBlock {
  return { type: 'image', attachment: imageRef(mediaType) }
}

/**
 * Deterministic image byte reader for serialization tests. Returns fixed bytes
 * so the produced `image.source.bytes` is stable across runs; snapshot tests
 * convert it to an array for JSON-stable output.
 */
export const fakeReadImage: ImageReader = async () => new Uint8Array([1, 2, 3, 4])

/** Brand a reasoning effort id for GenerateOptions. */
export function effort(id: string): ReasoningEffortId {
  return ReasoningEffortId(id)
}

/** Assemble a GenerateOptions with sane defaults over an override bag. */
export function options(overrides: Partial<GenerateOptions> & Pick<GenerateOptions, 'model'>): GenerateOptions {
  return {
    provider: 'bedrock',
    messages: [],
    ...overrides,
  }
}

/** Adapter defaults mirroring the plugin's DEFAULT_THINKING_BUDGET_BY_EFFORT. */
export const DEFAULTS: RequestDefaults = {
  thinkingBudgetTokens: 4_096,
  thinkingBudgetByEffort: { low: 1_024, high: 4_096, max: 16_384 },
}

/** Adapter defaults with no per-effort tiers (pre-tiered fallback behaviour). */
export const DEFAULTS_FLAT: RequestDefaults = {
  thinkingBudgetTokens: 4_096,
}
