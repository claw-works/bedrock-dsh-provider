/**
 * Adapter-private replay state carried on the terminal `finish` chunk's
 * {@link ReplayEnvelope}. Claude extended thinking is only replayable with the
 * opaque `signature` Bedrock issues alongside each reasoning block; the harness
 * reasoning block itself carries only text, so the adapter stashes the
 * signatures here, one entry per emitted reasoning block in stream order, and
 * reads them back when serializing that assistant turn on a later request.
 *
 * @module dsh-llm-bedrock/replay
 */

/** Key under which reasoning signatures live in {@link ReplayEnvelope.response}. */
export const REASONING_SIGNATURES_KEY = 'bedrockReasoningSignatures'

/**
 * Bedrock-issued reasoning signatures, one per emitted reasoning block in the
 * order those blocks appeared in the assistant turn. An entry is `undefined`
 * when the model produced a reasoning block with no signature (redacted
 * thinking), which is simply not replayed.
 */
export type ReasoningSignatures = readonly (string | undefined)[]

/** Shape of the response-level replay metadata this adapter writes. */
export interface BedrockReplayResponse {
  [REASONING_SIGNATURES_KEY]: ReasoningSignatures
}
