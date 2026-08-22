/**
 * Register a {@link BedrockAdapter} for the `bedrock` provider route on
 * `ctx.llm`. Connection facts resolve per request from the plugin's
 * `cordis.yml` entry config layered under the optional `llm-bedrock`
 * user-settings section, so a changed region, endpoint, or catalog reaches the
 * next request without a restart while an in-flight stream keeps the facts it
 * started with. The one registration-captured fact — the retry policy —
 * re-registers the route in place when it changes.
 *
 * AWS credentials are NOT a harness credential-seam concern here: Bedrock
 * authenticates with SigV4, resolved by the AWS SDK's default provider chain
 * (environment, shared config / profile, SSO, container, or instance role).
 * The plugin only selects region / profile / endpoint.
 *
 * @module @deepseek-ai/dsh-llm-bedrock
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { resolveRetryPolicy, RetryPolicySchema } from '@deepseek-ai/dsh-llm'
import type { RetryPolicyConfig } from '@deepseek-ai/dsh-llm'
import { launchEnvironmentOf, type LaunchEnvironmentSnapshot } from '@deepseek-ai/dsh-launch-environment'
import { deepEqualJson, installSettingsSection, settingsNamespace } from '@deepseek-ai/dsh-settings'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import {
  BedrockAdapter,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_TOKENS,
  DEFAULT_STREAM_IDLE_TIMEOUT_MS,
  DEFAULT_THINKING_BUDGET_BY_EFFORT,
  DEFAULT_THINKING_BUDGET_TOKENS,
} from './adapter.js'
import type { BedrockCatalogModel, BedrockConnectionOptions } from './adapter.js'
import type { ThinkingBudgetByEffort } from './serialize.js'

export {
  BedrockAdapter,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_TOKENS,
  DEFAULT_STREAM_IDLE_TIMEOUT_MS,
  DEFAULT_THINKING_BUDGET_BY_EFFORT,
  DEFAULT_THINKING_BUDGET_TOKENS,
} from './adapter.js'
export type { BedrockAdapterOptions, BedrockCatalogModel, BedrockConnectionOptions } from './adapter.js'
export type { RequestDefaults, ThinkingBudgetByEffort, ThinkingEffortTier } from './serialize.js'
export { modelCapabilities, modelFamily } from './model.js'
export type { ModelCapabilities, ModelFamily } from './model.js'
export type * from './types.js'

export const name = 'llm-bedrock'
export const inject = ['llm']

const NS = settingsNamespace('llm-bedrock')
/** The single provider route this plugin owns. */
const PROVIDER = 'bedrock'
/** Environment variables naming the region, honored from trusted layers. */
const REGION_ENV = ['AWS_REGION', 'AWS_DEFAULT_REGION'] as const

const DEFAULT_MODELS: BedrockCatalogModel[] = [
  {
    id: 'us.anthropic.claude-sonnet-4-20250514-v1:0',
    name: 'Claude Sonnet 4',
    contextWindow: DEFAULT_CONTEXT_WINDOW,
  },
  {
    id: 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
    name: 'Claude 3.7 Sonnet',
    contextWindow: DEFAULT_CONTEXT_WINDOW,
  },
  {
    id: 'openai.gpt-oss-120b-1:0',
    name: 'OpenAI GPT-OSS 120B',
    contextWindow: DEFAULT_CONTEXT_WINDOW,
  },
  {
    id: 'xai.grok-4.6',
    name: 'xAI Grok 4.6',
    contextWindow: DEFAULT_CONTEXT_WINDOW,
  },
]

/**
 * Plugin config, validated by the same-named schemastery schema and doubling
 * as the `llm-bedrock` settings-section shape. Every field is optional in yml:
 * an omitted region falls back to the AWS SDK's own resolution (AWS_REGION,
 * shared config), and credentials are never configured here.
 */
export interface Config {
  /** AWS region; falls back to $AWS_REGION / $AWS_DEFAULT_REGION, then SDK resolution. */
  region?: string
  /** Shared-config profile name; omission uses the SDK default. */
  profile?: string
  /** Override endpoint URL (VPC endpoint / gateway); omission uses the regional default. */
  endpoint?: string
  /** Default per-request output cap (default 8192); a model's own cap and explicit request values win. */
  maxTokens?: number
  /** Token budget for the Claude thinking channel when reasoning is on (default 4096); the per-effort fallback. */
  thinkingBudgetTokens?: number
  /**
   * Per-effort Claude thinking budgets (`low` / `high` / `max`). Any tier
   * omitted falls back to {@link thinkingBudgetTokens}; omitting the whole
   * object keeps the built-in tiered defaults. `medium` is not a separate tier
   * (the resolver treats it as `high`).
   */
  thinkingBudgetByEffort?: Partial<Record<'low' | 'high' | 'max', number>>
  /** Positive context capacity used when the selected model has no exact value (default 200000). */
  defaultContextWindow?: number
  /** Advisory models shown by discovery consumers; defaults to two Claude entries. */
  models?: BedrockCatalogModel[]
  /** Maximum provider idle time while one stream read is outstanding (default five minutes). */
  streamIdleTimeoutMs?: number
  /** Provider-owned model-request retry policy; omission uses normal mode with five retries. */
  retryPolicy?: RetryPolicyConfig
}

const catalogModel: z<BedrockCatalogModel> = z.object({
  id: z.string().required(),
  name: z.string(),
  description: z.string(),
  contextWindow: z.number().step(1).min(1),
  maxTokens: z.number().step(1).min(1),
})

export const Config: z<Config> = z.object({
  region: z.string(),
  profile: z.string(),
  endpoint: z.string(),
  maxTokens: z.number().step(1).min(1).max(Number.MAX_SAFE_INTEGER).default(DEFAULT_MAX_TOKENS),
  thinkingBudgetTokens: z.number().step(1).min(1).default(DEFAULT_THINKING_BUDGET_TOKENS),
  thinkingBudgetByEffort: z.object({
    low: z.number().step(1).min(1),
    high: z.number().step(1).min(1),
    max: z.number().step(1).min(1),
  }),
  defaultContextWindow: z.number().step(1).min(1).default(DEFAULT_CONTEXT_WINDOW),
  models: z.array(catalogModel).default(DEFAULT_MODELS),
  streamIdleTimeoutMs: z.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
  retryPolicy: RetryPolicySchema,
})

/**
 * Merge configured per-effort thinking budgets over the built-in tiered
 * defaults, validating every provided value. Returns the resolved `low` / `high`
 * / `max` map. The map is always present so tiered budgeting is the default;
 * `thinkingBudgetTokens` still acts as the resolver's final fallback for any
 * effort id outside these tiers.
 * @param configured - the optional per-effort overrides from plugin config.
 * @returns the resolved per-effort budget map.
 */
function resolveThinkingBudgetByEffort(
  configured: Partial<Record<'low' | 'high' | 'max', number>> | undefined,
): ThinkingBudgetByEffort {
  const merged: Record<'low' | 'high' | 'max', number> = { ...DEFAULT_THINKING_BUDGET_BY_EFFORT }
  if (configured !== undefined) {
    for (const tier of ['low', 'high', 'max'] as const) {
      const value = configured[tier]
      if (value === undefined) continue
      if (!Number.isSafeInteger(value) || value <= 0) {
        throw new Error(`llm-bedrock: thinkingBudgetByEffort.${tier} must be a positive safe integer`)
      }
      merged[tier] = value
    }
  }
  return merged
}

/** Resolve, validate, and detach the advisory model catalog. */
function resolveModels(models: readonly BedrockCatalogModel[] | undefined): BedrockCatalogModel[] {
  const seen = new Set<string>()
  return (models ?? DEFAULT_MODELS).map((model) => {
    if (model.id.length === 0) throw new Error('llm-bedrock: catalog model ids must be non-empty')
    if (model.name !== undefined && model.name.length === 0) {
      throw new Error(`llm-bedrock: catalog model "${model.id}" has an empty name`)
    }
    if (model.contextWindow !== undefined
      && (!Number.isInteger(model.contextWindow) || model.contextWindow <= 0)) {
      throw new Error(`llm-bedrock: catalog model "${model.id}" contextWindow must be a positive integer`)
    }
    if (model.maxTokens !== undefined
      && (!Number.isInteger(model.maxTokens) || model.maxTokens <= 0)) {
      throw new Error(`llm-bedrock: catalog model "${model.id}" maxTokens must be a positive integer`)
    }
    if (seen.has(model.id)) throw new Error(`llm-bedrock: duplicate catalog model "${model.id}"`)
    seen.add(model.id)
    return {
      id: model.id,
      ...model.name === undefined ? {} : { name: model.name },
      ...model.description === undefined ? {} : { description: model.description },
      ...model.contextWindow === undefined ? {} : { contextWindow: model.contextWindow },
      ...model.maxTokens === undefined ? {} : { maxTokens: model.maxTokens },
    }
  })
}

/**
 * The one explicit resolve step from raw config to validated connection facts.
 * @param config - raw plugin config or resolved settings snapshot.
 * @param environment - this run's environment layers, or `undefined` outside the product CLI.
 * @returns validated connection facts.
 */
export function resolveAdapterOptions(
  config: Config,
  environment?: LaunchEnvironmentSnapshot,
): BedrockConnectionOptions {
  if (config.defaultContextWindow !== undefined
    && (!Number.isInteger(config.defaultContextWindow) || config.defaultContextWindow <= 0)) {
    throw new Error('llm-bedrock: defaultContextWindow must be a positive integer')
  }
  if (config.maxTokens !== undefined
    && (!Number.isSafeInteger(config.maxTokens) || config.maxTokens <= 0)) {
    throw new Error('llm-bedrock: maxTokens must be a positive safe integer')
  }
  const thinkingBudgetTokens = config.thinkingBudgetTokens ?? DEFAULT_THINKING_BUDGET_TOKENS
  if (!Number.isSafeInteger(thinkingBudgetTokens) || thinkingBudgetTokens <= 0) {
    throw new Error('llm-bedrock: thinkingBudgetTokens must be a positive safe integer')
  }
  const thinkingBudgetByEffort = resolveThinkingBudgetByEffort(config.thinkingBudgetByEffort)
  const streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS
  if (!Number.isFinite(streamIdleTimeoutMs)
    || streamIdleTimeoutMs <= 0
    || streamIdleTimeoutMs > MAX_TIMER_DELAY_MS) {
    throw new Error(
      `llm-bedrock: streamIdleTimeoutMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS}`,
    )
  }
  const region = config.region
    ?? REGION_ENV.map(key => environment?.get(key)?.value).find(value => value !== undefined && value.length > 0)
  return {
    ...region === undefined ? {} : { region },
    ...config.profile === undefined ? {} : { profile: config.profile },
    ...config.endpoint === undefined ? {} : { endpoint: config.endpoint },
    defaults: { thinkingBudgetTokens, thinkingBudgetByEffort },
    maxTokens: config.maxTokens ?? DEFAULT_MAX_TOKENS,
    defaultContextWindow: config.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW,
    models: resolveModels(config.models),
    streamIdleTimeoutMs,
    retryPolicy: resolveRetryPolicy(config.retryPolicy, 'llm-bedrock: retryPolicy'),
  }
}

export function apply(ctx: Context, config: Config): void {
  let current: () => Config = () => config
  let lastRaw: Config | undefined
  let lastGood: BedrockConnectionOptions | undefined
  const options = (): BedrockConnectionOptions => {
    const raw = current()
    if (raw === lastRaw && lastGood !== undefined) return lastGood
    try {
      const next = resolveAdapterOptions(raw, launchEnvironmentOf(ctx))
      lastRaw = raw
      lastGood = next
      return next
    } catch (error) {
      if (lastGood === undefined) throw error
      lastRaw = raw
      ctx.logger.error('llm-bedrock: keeping the last good configuration after an invalid settings section')
      ctx.logger.error(error)
      return lastGood
    }
  }
  options()

  const adapter = new BedrockAdapter({ options })
  ctx.llm.registerConfigurableProviders([
    { provider: PROVIDER, displayName: 'Amazon Bedrock', settingsNs: NS, settingsPath: [] },
  ])
  const registration = ctx.llm.registerAdapter([PROVIDER], adapter)
  let registeredPolicy = options().retryPolicy
  const ensureRegistrationFacts = (): void => {
    const policy = options().retryPolicy
    if (deepEqualJson(policy, registeredPolicy)) return
    registration.replace([PROVIDER])
    registeredPolicy = policy
  }

  installSettingsSection(ctx, NS, Config, config, {
    setSource: (source) => {
      current = source
    },
    onChange: ensureRegistrationFacts,
  })
}
