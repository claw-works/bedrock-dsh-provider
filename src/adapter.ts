/**
 * `BedrockAdapter`: AWS SDK ConverseStream against Amazon Bedrock, emitting
 * harness StreamChunks. Transport-only, like the DeepSeek adapter: connection
 * facts arrive through a thunk resolved once per operation, and AWS credentials
 * come from the SDK's default provider chain (environment, shared config /
 * profile, SSO, container, or instance role) rather than the harness credential
 * seam, because Bedrock authenticates with SigV4, not a bearer key.
 *
 * @module dsh-llm-bedrock/adapter
 */

import {
  BedrockRuntimeClient,
  ConverseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime'
import type { BedrockRuntimeClientConfig } from '@aws-sdk/client-bedrock-runtime'
import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  LlmAdapter,
  LlmError,
  ProviderRequestId,
  QUOTA_EXCEEDED_CODE,
  ReasoningEffortId,
} from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ResolvedRetryPolicy,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import { modelCapabilities } from './model.js'
import { serializeRequest } from './serialize.js'
import type { RequestDefaults } from './serialize.js'
import { translate } from './translate.js'
import type { ConverseStreamOutput } from './types.js'

/** One optional model entry advertised by the Bedrock adapter. */
export interface BedrockCatalogModel {
  /** Bedrock model id, inference profile id, or model ARN accepted by ConverseStream. */
  id: string
  /** Selector label; defaults to {@link id}. */
  name?: string
  /** Optional selector detail. */
  description?: string
  /** Known combined request/response context capacity; omitted when unavailable. */
  contextWindow?: number
  /** Per-request output cap for this model; omission falls back to the profile default. */
  maxTokens?: number
}

/** Validated connection facts for one operation, produced by the plugin's resolve step. */
export interface BedrockConnectionOptions {
  /** AWS region hosting the Bedrock endpoint; omission lets the SDK resolve it. */
  region?: string
  /** Shared-config profile name; omission uses the SDK's default profile resolution. */
  profile?: string
  /** Full override endpoint URL (VPC endpoint, gateway); omission uses the regional default. */
  endpoint?: string
  /** Request defaults applied to every call (thinking budget). */
  defaults: RequestDefaults
  /** Default per-request output cap; explicit request values win. */
  maxTokens: number
  /** Positive context capacity used when the selected model has no exact value. */
  defaultContextWindow: number
  /** Advisory models exposed to discovery consumers; requests remain unrestricted. */
  models: readonly BedrockCatalogModel[]
  /** Maximum provider idle time while one stream read is outstanding. */
  streamIdleTimeoutMs: number
  /** Provider-owned model-request retry policy, already resolved. */
  retryPolicy: ResolvedRetryPolicy
}

/** Constructor options: the operation-local resolution hooks the plugin owns. */
export interface BedrockAdapterOptions {
  /** Current validated connection facts; called once per operation. */
  options: () => BedrockConnectionOptions
}

/** Default maximum idle interval while an adapter stream read is outstanding. */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000
/** Default combined request/response context capacity (Claude 3.7 / 4 window). */
export const DEFAULT_CONTEXT_WINDOW = 200_000
/** Default per-request output-token cap. */
export const DEFAULT_MAX_TOKENS = 8_192
/** Default token budget for the Claude thinking channel. */
export const DEFAULT_THINKING_BUDGET_TOKENS = 4_096
const STREAM_IDLE_TIMEOUT_CODE = 'LLM_STREAM_IDLE_TIMEOUT'

const OFF_REASONING_EFFORT = ReasoningEffortId('off')
const LOW_REASONING_EFFORT = ReasoningEffortId('low')
const HIGH_REASONING_EFFORT = ReasoningEffortId('high')
const MAX_REASONING_EFFORT = ReasoningEffortId('max')
const REASONING_EFFORTS = [
  { id: OFF_REASONING_EFFORT, name: 'Off' },
  { id: LOW_REASONING_EFFORT, name: 'Low' },
  { id: HIGH_REASONING_EFFORT, name: 'High' },
  { id: MAX_REASONING_EFFORT, name: 'Max' },
] as const

function modelInfo(provider: string, model: BedrockCatalogModel): LlmModelInfo {
  return {
    provider,
    id: model.id,
    name: model.name ?? model.id,
    ...model.description === undefined ? {} : { description: model.description },
    inputModalities: ['text'],
  }
}

/** AWS SDK error carrying the fields we map onto an LlmError. */
interface AwsError {
  name?: string
  message?: string
  $metadata?: { httpStatusCode?: number; requestId?: string }
}

function asAwsError(error: unknown): AwsError {
  return typeof error === 'object' && error !== null ? (error as AwsError) : {}
}

/**
 * Map an AWS SDK error to a stable LlmError code, keying on the exception name
 * first (stable across the SDK) and the HTTP status second.
 * @param error - the thrown SDK error.
 * @returns the normalized harness error code.
 */
export function awsErrorCode(error: AwsError): string {
  const name = error.name ?? ''
  const status = error.$metadata?.httpStatusCode
  switch (name) {
    case 'AccessDeniedException':
    case 'UnrecognizedClientException':
    case 'ExpiredTokenException':
      return 'AUTH'
    case 'ThrottlingException':
    case 'TooManyRequestsException':
      return 'RATE_LIMIT'
    case 'ServiceQuotaExceededException':
      return QUOTA_EXCEEDED_CODE
    case 'ValidationException':
      return /context window|too many tokens|input is too long|maximum context/i.test(error.message ?? '')
        ? CONTEXT_WINDOW_EXCEEDED_CODE
        : 'INVALID_REQUEST'
    case 'ResourceNotFoundException':
      return 'INVALID_REQUEST'
    case 'ModelTimeoutException':
      return 'TIMEOUT'
    case 'InternalServerException':
    case 'ServiceUnavailableException':
    case 'ModelErrorException':
    case 'ModelStreamErrorException':
    case 'ModelNotReadyException':
      return 'SERVER'
    default:
      break
  }
  if (status === 401 || status === 403) return 'AUTH'
  if (status === 429) return 'RATE_LIMIT'
  if (status === 400) return 'INVALID_REQUEST'
  if (status !== undefined && status >= 500) return 'SERVER'
  return name.length > 0 ? name : 'TRANSPORT'
}

/**
 * A `LlmAdapter` backed by the AWS Bedrock ConverseStream API. One instance
 * serves every model id it was registered under (the harness model name IS the
 * Bedrock model id / inference profile id / ARN).
 *
 * One stable signal reaches both the SDK send and the stream reads. Caller
 * aborts map to `ABORTED`; the configured per-read idle watchdog maps to
 * `TIMEOUT`.
 */
export class BedrockAdapter extends LlmAdapter {
  /** One SDK client per distinct connection key, reused across requests. */
  private clients = new Map<string, BedrockRuntimeClient>()

  constructor(private readonly config: BedrockAdapterOptions) {
    super()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'Amazon Bedrock' }
  }

  override providerRetryPolicy(_provider: string): ResolvedRetryPolicy {
    return this.config.options().retryPolicy
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return Promise.resolve(this.config.options().models.map(model => modelInfo(provider, model)))
  }

  override resolveModel(
    provider: string,
    model: string,
    _signal?: AbortSignal,
  ): Promise<LlmResolvedModelInfo> {
    const connection = this.config.options()
    const configured = connection.models.find(entry => entry.id === model)
    const capabilities = modelCapabilities(model)
    const contextWindow = configured?.contextWindow ?? connection.defaultContextWindow
    return Promise.resolve({
      ...configured === undefined
        ? { provider, id: model, name: model, inputModalities: ['text' as const] }
        : modelInfo(provider, configured),
      context: { contextWindow },
      defaultMaxTokens: configured?.maxTokens ?? connection.maxTokens,
      // Only reasoning-capable Claude generations expose an effort selector;
      // every other model reports no reasoning knob at all.
      ...capabilities.reasoning === 'claude-thinking'
        ? {
          reasoning: {
            efforts: REASONING_EFFORTS,
            defaultEffort: OFF_REASONING_EFFORT,
          },
        }
        : {},
    })
  }

  /** Build (or reuse) the SDK client for one resolved connection. */
  private client(connection: BedrockConnectionOptions): BedrockRuntimeClient {
    const key = JSON.stringify([connection.region, connection.profile, connection.endpoint])
    const existing = this.clients.get(key)
    if (existing !== undefined) return existing
    const clientConfig: BedrockRuntimeClientConfig = {
      ...connection.region === undefined ? {} : { region: connection.region },
      ...connection.endpoint === undefined ? {} : { endpoint: connection.endpoint },
      // The default SDK credential chain resolves the profile from AWS_PROFILE;
      // an explicit profile is applied by constructing a provider only when set,
      // to avoid pinning credentials when the deployment relies on a role.
      ...connection.profile === undefined ? {} : { profile: connection.profile },
      // The harness owns retries at the seam; do not double-retry in the SDK.
      maxAttempts: 1,
    }
    const client = new BedrockRuntimeClient(clientConfig)
    this.clients.set(key, client)
    return client
  }

  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    // One resolution per stream call: connection facts freeze here and hold for
    // this whole request, so an in-flight stream never observes a config change.
    const connection = this.config.options()
    const consumer = new AbortController()
    const upstream = options.signal === undefined
      ? consumer.signal
      : AbortSignal.any([options.signal, consumer.signal])
    using watchdog = idleWatchdog(upstream, connection.streamIdleTimeoutMs, STREAM_IDLE_TIMEOUT_CODE)
    const iterator = this.request(options, watchdog.signal, connection)[Symbol.asyncIterator]()
    let exhausted = false
    try {
      while (true) {
        const result = await watchdog.next(iterator)
        if (result.done) {
          exhausted = true
          return
        }
        yield result.value
      }
    } catch (error: unknown) {
      if (timeoutOf(watchdog.signal, STREAM_IDLE_TIMEOUT_CODE) !== undefined) {
        throw new LlmError(
          `Bedrock stream idle timeout after ${connection.streamIdleTimeoutMs}ms`,
          'TIMEOUT',
          { cause: error },
        )
      }
      if (options.signal?.aborted) {
        throw new LlmError('Bedrock request aborted by caller', 'ABORTED', { cause: error })
      }
      if (error instanceof LlmError) throw error
      const aws = asAwsError(error)
      throw new LlmError(
        aws.message ?? 'Bedrock ConverseStream failed',
        awsErrorCode(aws),
        {
          ...aws.$metadata?.httpStatusCode === undefined ? {} : { status: aws.$metadata.httpStatusCode },
          ...aws.$metadata?.requestId === undefined || aws.$metadata.requestId.length === 0
            ? {}
            : { requestId: ProviderRequestId(aws.$metadata.requestId) },
          cause: error,
        },
      )
    } finally {
      consumer.abort('Bedrock stream consumer stopped')
      if (!exhausted && iterator.return !== undefined) {
        try {
          await iterator.return()
        } catch (_teardown) {
          // The consumer controller already owns termination; a return-time abort adds no second outcome.
        }
      }
    }
  }

  private async * request(
    options: GenerateOptions,
    signal: AbortSignal,
    connection: BedrockConnectionOptions,
  ): AsyncIterable<StreamChunk> {
    const capabilities = modelCapabilities(options.model)
    // Serialization runs before the send so a mapping failure surfaces as its
    // own error rather than being labelled a transport failure.
    const input = serializeRequest(options, capabilities, connection.defaults)
    const command = new ConverseStreamCommand(input)

    let response
    try {
      response = await this.client(connection).send(command, { abortSignal: signal })
    } catch (error: unknown) {
      if (signal.aborted) throw error
      const aws = asAwsError(error)
      throw new LlmError(
        aws.message ?? `Bedrock ConverseStream request failed`,
        awsErrorCode(aws),
        {
          ...aws.$metadata?.httpStatusCode === undefined ? {} : { status: aws.$metadata.httpStatusCode },
          ...aws.$metadata?.requestId === undefined || aws.$metadata.requestId.length === 0
            ? {}
            : { requestId: ProviderRequestId(aws.$metadata.requestId) },
          cause: error,
        },
      )
    }

    if (response.stream === undefined) {
      throw new LlmError('Bedrock ConverseStream returned no event stream', 'EMPTY_RESPONSE')
    }

    // The SDK stream is an async iterable of ConverseStreamOutput; each read
    // pulses the watchdog through the outer `watchdog.next` wrapper.
    yield* translate(response.stream as AsyncIterable<ConverseStreamOutput>)
  }
}
