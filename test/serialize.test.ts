/**
 * Offline unit tests for `src/serialize.ts`: the harness → Bedrock
 * ConverseStream request mapping. No AWS access; image bytes are supplied by a
 * fake reader. Covers system slot, toolConfig, same-role merge, toolResult
 * folding, Stage-1 Claude effort tiers, Stage-2 OpenAI/Grok reasoning mapping,
 * and Stage-3 image blocks.
 */

import { describe, expect, it } from 'vitest'
import type { GenerateOptions } from '@deepseek-ai/dsh-llm'
import { modelCapabilities } from '../src/model.js'
import { serializeMessages, serializeRequest } from '../src/serialize.js'
import type { ImageReader } from '../src/serialize.js'
import {
  DEFAULTS,
  DEFAULTS_FLAT,
  assistant,
  effort,
  fakeReadImage,
  imageBlock,
  options,
  toolResult,
  userBlocks,
  userText,
} from './helpers.js'

const CLAUDE = 'anthropic.claude-sonnet-4-20250514-v1:0'
const CLAUDE_CAPS = modelCapabilities(CLAUDE)

/** Convenience: build a request for CLAUDE with tiered defaults. */
async function claudeRequest(overrides: Partial<GenerateOptions>) {
  const opts = options({ ...overrides, model: overrides.model ?? CLAUDE })
  return serializeRequest(opts, modelCapabilities(opts.model), DEFAULTS, fakeReadImage)
}

describe('serializeRequest — request skeleton', () => {
  it('maps model id, system slot, and inferenceConfig', async () => {
    const input = await claudeRequest({
      system: 'be terse',
      messages: [userText('hi')],
      maxTokens: 512,
      temperature: 0.3,
      stop: ['STOP'],
    })
    expect(input.modelId).toBe(CLAUDE)
    expect(input.system).toEqual([{ text: 'be terse' }])
    expect(input.inferenceConfig).toEqual({ maxTokens: 512, temperature: 0.3, stopSequences: ['STOP'] })
    expect(input.messages).toEqual([{ role: 'user', content: [{ text: 'hi' }] }])
  })

  it('omits system and inferenceConfig when the caller supplies neither', async () => {
    const input = await claudeRequest({ messages: [userText('hi')] })
    expect(input.system).toBeUndefined()
    expect(input.inferenceConfig).toBeUndefined()
    expect(input.additionalModelRequestFields).toBeUndefined()
  })

  it('builds a toolConfig from tool schemas for a tool-capable model', async () => {
    const input = await claudeRequest({
      messages: [userText('hi')],
      tools: [{ name: 'get_time', description: 'now', parameters: { type: 'object', properties: {} } }],
    })
    expect(input.toolConfig).toEqual({
      tools: [{
        toolSpec: {
          name: 'get_time',
          description: 'now',
          inputSchema: { json: { type: 'object', properties: {} } },
        },
      }],
    })
  })

  it('drops toolConfig for a family that cannot take tools (titan)', async () => {
    const model = 'amazon.titan-text-premier-v1:0'
    const input = await serializeRequest(
      options({ model, messages: [userText('hi')], tools: [{ name: 't', description: 'd', parameters: {} }] }),
      modelCapabilities(model),
      DEFAULTS,
      fakeReadImage,
    )
    expect(input.toolConfig).toBeUndefined()
  })
})

describe('serializeMessages — role merge and tool results', () => {
  it('merges consecutive same-role turns so the wire strictly alternates', async () => {
    const wire = await serializeMessages(
      [userText('one'), userText('two'), assistant([{ type: 'text', text: 'reply' }])],
      CLAUDE_CAPS, CLAUDE, fakeReadImage,
    )
    expect(wire).toEqual([
      { role: 'user', content: [{ text: 'one' }, { text: 'two' }] },
      { role: 'assistant', content: [{ text: 'reply' }] },
    ])
  })

  it('folds a tool-result into a user turn keyed by tool-use id', async () => {
    const wire = await serializeMessages(
      [toolResult('call-1', 'the answer is 42')],
      CLAUDE_CAPS, CLAUDE, fakeReadImage,
    )
    expect(wire).toEqual([
      { role: 'user', content: [{ toolResult: { toolUseId: 'call-1', content: [{ text: 'the answer is 42' }] } }] },
    ])
  })

  it('marks an errored tool-result and substitutes placeholder text for empty output', async () => {
    const wire = await serializeMessages(
      [toolResult('call-2', '', true)],
      CLAUDE_CAPS, CLAUDE, fakeReadImage,
    )
    expect(wire).toEqual([
      { role: 'user', content: [{ toolResult: { toolUseId: 'call-2', content: [{ text: '(no output)' }], status: 'error' } }] },
    ])
  })

  it('serializes an assistant tool-call, parsing its raw JSON arguments', async () => {
    const wire = await serializeMessages(
      [assistant([{ type: 'tool-call', id: 'call-9' as never, name: 'search', arguments: '{"q":"cats"}' }])],
      CLAUDE_CAPS, CLAUDE, fakeReadImage,
    )
    expect(wire).toEqual([
      { role: 'assistant', content: [{ toolUse: { toolUseId: 'call-9', name: 'search', input: { q: 'cats' } } }] },
    ])
  })

  it('replays a reasoning block only when its signature is present on replay state', async () => {
    const replayState = { response: { bedrockReasoningSignatures: ['sig-abc'] } }
    const withSig = assistant([{ type: 'reasoning', text: 'thinking...' }, { type: 'text', text: 'answer' }], replayState)
    const wireWith = await serializeMessages([withSig], CLAUDE_CAPS, CLAUDE, fakeReadImage)
    expect(wireWith).toEqual([
      { role: 'assistant', content: [
        { reasoningContent: { reasoningText: { text: 'thinking...', signature: 'sig-abc' } } },
        { text: 'answer' },
      ] },
    ])

    const noSig = assistant([{ type: 'reasoning', text: 'thinking...' }, { type: 'text', text: 'answer' }])
    const wireNo = await serializeMessages([noSig], CLAUDE_CAPS, CLAUDE, fakeReadImage)
    expect(wireNo).toEqual([{ role: 'assistant', content: [{ text: 'answer' }] }])
  })
})

describe('serializeMessages — image blocks (Stage 3)', () => {
  it('maps png/jpeg/gif/webp image blocks to Converse image blocks for an image-capable model', async () => {
    for (const [media, format] of [
      ['image/png', 'png'], ['image/jpeg', 'jpeg'], ['image/gif', 'gif'], ['image/webp', 'webp'],
    ] as const) {
      const wire = await serializeMessages(
        [userBlocks([{ type: 'text', text: 'see' }, imageBlock(media)])],
        CLAUDE_CAPS, CLAUDE, fakeReadImage,
      )
      expect(wire).toHaveLength(1)
      const content = wire[0]!.content!
      expect(content[0]).toEqual({ text: 'see' })
      expect(content[1]).toEqual({ image: { format, source: { bytes: new Uint8Array([1, 2, 3, 4]) } } })
    }
  })

  it('rejects image content on a family that cannot take images, naming model and family', async () => {
    const model = 'deepseek.r1-v1:0'
    await expect(
      serializeMessages([userBlocks([imageBlock('image/png')])], modelCapabilities(model), model, fakeReadImage),
    ).rejects.toThrow(/deepseek\.r1-v1:0.*family "deepseek".*image content/s)
  })

  it('rejects an image media type it cannot map to a Converse format', async () => {
    const badReader: ImageReader = async () => new Uint8Array([0])
    const badBlock = { type: 'image', attachment: { attachmentId: 'x' as never, mediaType: 'image/tiff' as never, bytes: 1, width: 1, height: 1 } } as never
    await expect(
      serializeMessages([userBlocks([badBlock])], CLAUDE_CAPS, CLAUDE, badReader),
    ).rejects.toThrow(/cannot map image media type "image\/tiff"/)
  })
})

describe('resolveThinking — Claude effort tiers (Stage 1)', () => {
  it('maps low/high/max to distinct budget_tokens', async () => {
    const low = await claudeRequest({ messages: [userText('x')], reasoningEffort: effort('low') })
    const high = await claudeRequest({ messages: [userText('x')], reasoningEffort: effort('high') })
    const max = await claudeRequest({ messages: [userText('x')], reasoningEffort: effort('max') })
    expect(low.additionalModelRequestFields).toEqual({ thinking: { type: 'enabled', budget_tokens: 1_024 } })
    expect(high.additionalModelRequestFields).toEqual({ thinking: { type: 'enabled', budget_tokens: 4_096 } })
    expect(max.additionalModelRequestFields).toEqual({ thinking: { type: 'enabled', budget_tokens: 16_384 } })
  })

  it('treats medium as the high tier', async () => {
    const medium = await claudeRequest({ messages: [userText('x')], reasoningEffort: effort('medium') })
    expect(medium.additionalModelRequestFields).toEqual({ thinking: { type: 'enabled', budget_tokens: 4_096 } })
  })

  it('falls back to the single default budget when no per-effort tiers are configured', async () => {
    const opts = options({ model: CLAUDE, messages: [userText('x')], reasoningEffort: effort('low') })
    const input = await serializeRequest(opts, CLAUDE_CAPS, DEFAULTS_FLAT, fakeReadImage)
    expect(input.additionalModelRequestFields).toEqual({ thinking: { type: 'enabled', budget_tokens: 4_096 } })
  })

  it('omits thinking when reasoning is off, absent, or the purpose is session-title', async () => {
    expect((await claudeRequest({ messages: [userText('x')], reasoningEffort: effort('off') })).additionalModelRequestFields).toBeUndefined()
    expect((await claudeRequest({ messages: [userText('x')] })).additionalModelRequestFields).toBeUndefined()
    expect((await claudeRequest({ messages: [userText('x')], reasoningEffort: effort('high'), purpose: 'session-title' })).additionalModelRequestFields).toBeUndefined()
  })

  it('does not send Claude thinking fields to a non-reasoning Claude generation', async () => {
    const model = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
    const input = await serializeRequest(
      options({ model, messages: [userText('x')], reasoningEffort: effort('high') }),
      modelCapabilities(model), DEFAULTS, fakeReadImage,
    )
    expect(input.additionalModelRequestFields).toBeUndefined()
  })
})

describe('resolveThinking — OpenAI / Grok reasoning (Stage 2)', () => {
  it('gpt-oss uses a flat reasoning_effort string', async () => {
    const model = 'openai.gpt-oss-120b-1:0'
    const input = await serializeRequest(
      options({ model, messages: [userText('x')], reasoningEffort: effort('high') }),
      modelCapabilities(model), DEFAULTS, fakeReadImage,
    )
    expect(input.additionalModelRequestFields).toEqual({ reasoning_effort: 'high' })
  })

  it('GPT-5.x uses a nested reasoning.effort object', async () => {
    const model = 'us.openai.gpt-5.6-2025xx'
    const input = await serializeRequest(
      options({ model, messages: [userText('x')], reasoningEffort: effort('low') }),
      modelCapabilities(model), DEFAULTS, fakeReadImage,
    )
    expect(input.additionalModelRequestFields).toEqual({ reasoning: { effort: 'low' } })
  })

  it('grok uses a flat reasoning_effort string, folding max into high', async () => {
    const model = 'us.xai.grok-4.6'
    const input = await serializeRequest(
      options({ model, messages: [userText('x')], reasoningEffort: effort('max') }),
      modelCapabilities(model), DEFAULTS, fakeReadImage,
    )
    expect(input.additionalModelRequestFields).toEqual({ reasoning_effort: 'high' })
  })

  it('never sends Claude thinking fields to openai/grok', async () => {
    for (const model of ['openai.gpt-oss-120b-1:0', 'us.xai.grok-4.6']) {
      const input = await serializeRequest(
        options({ model, messages: [userText('x')], reasoningEffort: effort('high') }),
        modelCapabilities(model), DEFAULTS, fakeReadImage,
      )
      expect((input.additionalModelRequestFields as Record<string, unknown>).thinking).toBeUndefined()
    }
  })
})

describe('serializeRequest — full body snapshot', () => {
  it('produces a stable ConverseStream body for a mixed conversation', async () => {
    const input = await claudeRequest({
      system: 'assistant',
      messages: [
        userText('what time is it?'),
        assistant([{ type: 'tool-call', id: 'c1' as never, name: 'clock', arguments: '{}' }]),
        toolResult('c1', '12:00'),
      ],
      tools: [{ name: 'clock', description: 'clock', parameters: { type: 'object' } }],
      maxTokens: 1024,
    })
    expect(input).toMatchSnapshot()
  })
})
