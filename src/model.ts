/**
 * Per-model capability judgement, keyed by Bedrock model id (or inference
 * profile id / ARN, which embed the same family token). The Converse API is
 * uniform, but reasoning, tool use, and image input differ by model, so a
 * request must consult the family before it maps optional features.
 *
 * The primary target is Anthropic Claude; other families are recognised only
 * enough to avoid sending them a request feature they will reject. Add a
 * branch here when a new family needs a different mapping.
 *
 * @module dsh-llm-bedrock/model
 */

/** Coarse Bedrock model family derived from a model id, inference profile, or ARN. */
export type ModelFamily =
  | 'claude'
  | 'nova'
  | 'llama'
  | 'deepseek'
  | 'mistral'
  | 'titan'
  | 'openai'
  | 'grok'
  | 'unknown'

/** Provider-neutral capability facts one request maps against. */
export interface ModelCapabilities {
  family: ModelFamily
  /** The model accepts `toolConfig`. */
  tools: boolean
  /** The model accepts image content blocks in user messages. */
  images: boolean
  /**
   * The model exposes a reasoning / extended-thinking channel and how it is
   * turned on:
   *  - `claude-thinking` drives Claude extended thinking through
   *    `additionalModelRequestFields.thinking` with an effort-tiered
   *    `budget_tokens`;
   *  - `openai-reasoning` and `grok-reasoning` drive the OpenAI (gpt-oss +
   *    GPT-5.x) and xAI Grok reasoning channels through a `reasoning_effort`
   *    string in `additionalModelRequestFields`;
   *  - `none` means the model has no reasoning knob this adapter drives.
   */
  reasoning: 'claude-thinking' | 'openai-reasoning' | 'grok-reasoning' | 'none'
}

/**
 * Classify a Bedrock model id into a coarse family. Matches the family token
 * anywhere in the string so cross-region inference profiles
 * (`us.anthropic.claude-...`) and full ARNs classify the same as a bare id.
 * @param modelId - the configured Bedrock model id, inference profile id, or ARN.
 * @returns the recognised family, or `unknown`.
 */
export function modelFamily(modelId: string): ModelFamily {
  const id = modelId.toLowerCase()
  if (id.includes('anthropic') || id.includes('claude')) return 'claude'
  if (id.includes('nova')) return 'nova'
  if (id.includes('llama') || id.includes('meta.')) return 'llama'
  if (id.includes('deepseek')) return 'deepseek'
  if (id.includes('mistral') || id.includes('mixtral')) return 'mistral'
  if (id.includes('titan')) return 'titan'
  // xAI Grok: `xai.grok-*` (plus `us.`/`eu.`/`apac.` cross-region prefixes).
  if (id.includes('grok') || id.includes('xai.')) return 'grok'
  // OpenAI on Bedrock: gpt-oss (`openai.gpt-oss-*`) and the GPT-5.x line
  // (`openai.gpt-5.6-*`). Both carry the `openai` provider token, so match that
  // (plus the `gpt-oss` model token for safety). Deliberately NOT a bare `gpt`
  // substring — that would misclassify unrelated ids like `custom.my-gpt-proxy`
  // as OpenAI and rob them of the unknown-family safe floor.
  if (id.includes('openai') || id.includes('gpt-oss')) return 'openai'
  return 'unknown'
}

/**
 * Whether a Claude model id names a generation with an extended-thinking
 * channel. Claude 3.7 Sonnet was the first; 4.x and later carry it too. Older
 * 3.0/3.5 ids do not, and sending them `thinking` request fields fails, so the
 * check is a positive allowlist of the generations known to support it.
 */
function claudeReasons(id: string): boolean {
  const lower = id.toLowerCase()
  if (lower.includes('claude-3-7') || lower.includes('claude-3.7')) return true
  // Claude 4 and 4.5 families (sonnet-4, opus-4, haiku-4, ...).
  return /claude-(?:sonnet-|opus-|haiku-)?[4-9]/.test(lower)
}

/**
 * Whether an OpenAI-family id names a gpt-oss model rather than a hosted
 * GPT-5.x model. The two split on two things: image input (gpt-oss is text-only
 * on Bedrock, the GPT-5.x line accepts images) and the reasoning wire shape
 * (gpt-oss takes a flat `reasoning_effort`, GPT-5.x takes a nested
 * `reasoning.effort` — see {@link module:dsh-llm-bedrock/serialize}). Exported so
 * the request serializer can pick the right reasoning field within the one
 * OpenAI family, a family-level branch rather than a per-model allowlist.
 */
export function openaiIsGptOss(id: string): boolean {
  return id.toLowerCase().includes('gpt-oss')
}

/**
 * Resolve the capabilities the adapter maps against for one model id. Unknown
 * families default to the safe common denominator the Converse API guarantees
 * for messages: tools yes (the API is tool-capable across families that
 * support messages), images and reasoning no, so an unverified model is never
 * sent a feature it may reject on every turn.
 * @param modelId - the configured Bedrock model id, inference profile id, or ARN.
 * @returns the capability facts for that model.
 */
export function modelCapabilities(modelId: string): ModelCapabilities {
  const family = modelFamily(modelId)
  switch (family) {
    case 'claude':
      return {
        family,
        tools: true,
        images: true,
        reasoning: claudeReasons(modelId) ? 'claude-thinking' : 'none',
      }
    case 'nova':
      return { family, tools: true, images: true, reasoning: 'none' }
    case 'llama':
      return { family, tools: true, images: true, reasoning: 'none' }
    case 'deepseek':
      return { family, tools: true, images: false, reasoning: 'none' }
    case 'mistral':
      return { family, tools: true, images: false, reasoning: 'none' }
    case 'titan':
      return { family, tools: false, images: false, reasoning: 'none' }
    case 'openai':
      // gpt-oss is text-only on Bedrock; the GPT-5.x line takes image input.
      // Both drive the OpenAI reasoning channel via `reasoning_effort`.
      return {
        family,
        tools: true,
        images: !openaiIsGptOss(modelId),
        reasoning: 'openai-reasoning',
      }
    case 'grok':
      // Grok 4.x is reasoning-first with a configurable effort; it accepts image
      // input on Bedrock.
      return { family, tools: true, images: true, reasoning: 'grok-reasoning' }
    default:
      return { family: 'unknown', tools: true, images: false, reasoning: 'none' }
  }
}
