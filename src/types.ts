/**
 * AWS Bedrock Converse wire format, re-exported from the AWS SDK. Types only.
 *
 * The Bedrock Converse API is a unified message API across every Bedrock model
 * (Claude, Nova, Llama, DeepSeek, ...). Unlike the DeepSeek OpenAI-compatible
 * wire, its content blocks are a tagged union where each member is a distinct
 * object key (`{ text }`, `{ toolUse }`, `{ reasoningContent }`, ...) rather
 * than a discriminant `type` field, and the streaming events are a structured
 * union delivered over the AWS EventStream framing that the SDK decodes for us.
 *
 * Source of truth: `@aws-sdk/client-bedrock-runtime` (ConverseStream API).
 *
 * @module dsh-llm-bedrock/types
 */

import type {
  ContentBlock,
  ConverseStreamCommandInput,
  ConverseStreamOutput,
  ConverseStreamMetadataEvent,
  Message as BedrockMessage,
  ReasoningContentBlockDelta,
  StopReason,
  SystemContentBlock,
  Tool,
  ToolConfiguration,
  ToolResultContentBlock,
  TokenUsage as BedrockTokenUsage,
} from '@aws-sdk/client-bedrock-runtime'
import type { DocumentType } from '@smithy/types'

export type {
  ContentBlock,
  ConverseStreamCommandInput,
  ConverseStreamOutput,
  ConverseStreamMetadataEvent,
  BedrockMessage,
  DocumentType,
  ReasoningContentBlockDelta,
  StopReason,
  SystemContentBlock,
  Tool,
  ToolConfiguration,
  ToolResultContentBlock,
  BedrockTokenUsage,
}

/**
 * `additionalModelRequestFields` payload used to turn on Anthropic Claude
 * extended thinking. Bedrock forwards this object verbatim to the model, so
 * the shape is Anthropic's, not Bedrock's own; only Claude reasoning-capable
 * models accept it (see {@link module:dsh-llm-bedrock/model}).
 */
export interface ClaudeThinkingRequestFields {
  thinking: {
    type: 'enabled'
    /** Token budget the model may spend on the reasoning channel. */
    budget_tokens: number
  }
}
