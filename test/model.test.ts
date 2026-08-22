/**
 * Offline unit tests for `src/model.ts`: family classification and capability
 * resolution across every recognised Bedrock family plus the unknown fallback.
 * Covers the Stage-2 OpenAI/Grok additions and their cross-region prefixes.
 */

import { describe, expect, it } from 'vitest'
import { modelCapabilities, modelFamily, openaiIsGptOss } from '../src/model.js'

describe('modelFamily', () => {
  it('classifies Claude by both anthropic and claude tokens', () => {
    expect(modelFamily('anthropic.claude-3-5-sonnet-20241022-v2:0')).toBe('claude')
    expect(modelFamily('claude-sonnet-4-20250514')).toBe('claude')
  })

  it('classifies the other native families', () => {
    expect(modelFamily('amazon.nova-pro-v1:0')).toBe('nova')
    expect(modelFamily('meta.llama3-1-70b-instruct-v1:0')).toBe('llama')
    expect(modelFamily('us.meta.llama3-3-70b-instruct-v1:0')).toBe('llama')
    expect(modelFamily('deepseek.r1-v1:0')).toBe('deepseek')
    expect(modelFamily('mistral.mistral-large-2407-v1:0')).toBe('mistral')
    expect(modelFamily('mistral.mixtral-8x7b-instruct-v0:1')).toBe('mistral')
    expect(modelFamily('amazon.titan-text-premier-v1:0')).toBe('titan')
  })

  it('classifies OpenAI (gpt-oss + GPT-5.x) via the openai/gpt-oss tokens', () => {
    expect(modelFamily('openai.gpt-oss-120b-1:0')).toBe('openai')
    expect(modelFamily('openai.gpt-oss-20b-1:0')).toBe('openai')
    expect(modelFamily('us.openai.gpt-5.6-2025xx')).toBe('openai')
  })

  it('classifies xAI Grok via grok/xai tokens including cross-region prefixes', () => {
    expect(modelFamily('xai.grok-4.6')).toBe('grok')
    expect(modelFamily('us.xai.grok-4.6')).toBe('grok')
    expect(modelFamily('eu.grok-3-mini')).toBe('grok')
  })

  it('matches the family token anywhere so inference profiles and ARNs classify the same', () => {
    expect(modelFamily('us.anthropic.claude-sonnet-4-20250514-v1:0')).toBe('claude')
    expect(modelFamily('arn:aws:bedrock:us-east-1:123:inference-profile/us.anthropic.claude-opus-4')).toBe('claude')
  })

  it('does NOT misclassify a bare gpt substring as OpenAI (guards the unknown floor)', () => {
    expect(modelFamily('custom.my-gpt-proxy')).toBe('unknown')
  })

  it('returns unknown for an unrecognised id', () => {
    expect(modelFamily('some.brand-new-model-v1:0')).toBe('unknown')
  })
})

describe('openaiIsGptOss', () => {
  it('splits gpt-oss from the hosted GPT-5.x line', () => {
    expect(openaiIsGptOss('openai.gpt-oss-120b-1:0')).toBe(true)
    expect(openaiIsGptOss('us.openai.gpt-5.6-2025xx')).toBe(false)
  })
})

describe('modelCapabilities', () => {
  it('claude 3.7 / 4.x expose the thinking channel; older 3.x do not', () => {
    expect(modelCapabilities('anthropic.claude-3-7-sonnet-20250219-v1:0')).toEqual({
      family: 'claude', tools: true, images: true, reasoning: 'claude-thinking',
    })
    expect(modelCapabilities('anthropic.claude-sonnet-4-20250514-v1:0').reasoning).toBe('claude-thinking')
    expect(modelCapabilities('anthropic.claude-3-5-sonnet-20241022-v2:0')).toEqual({
      family: 'claude', tools: true, images: true, reasoning: 'none',
    })
  })

  it('nova and llama take tools + images with no reasoning knob', () => {
    expect(modelCapabilities('amazon.nova-pro-v1:0')).toEqual({
      family: 'nova', tools: true, images: true, reasoning: 'none',
    })
    expect(modelCapabilities('meta.llama3-1-70b-instruct-v1:0')).toEqual({
      family: 'llama', tools: true, images: true, reasoning: 'none',
    })
  })

  it('deepseek and mistral take tools but not images', () => {
    expect(modelCapabilities('deepseek.r1-v1:0')).toEqual({
      family: 'deepseek', tools: true, images: false, reasoning: 'none',
    })
    expect(modelCapabilities('mistral.mistral-large-2407-v1:0')).toEqual({
      family: 'mistral', tools: true, images: false, reasoning: 'none',
    })
  })

  it('titan takes neither tools nor images', () => {
    expect(modelCapabilities('amazon.titan-text-premier-v1:0')).toEqual({
      family: 'titan', tools: false, images: false, reasoning: 'none',
    })
  })

  it('openai gpt-oss is text-only; GPT-5.x accepts images; both drive openai-reasoning', () => {
    expect(modelCapabilities('openai.gpt-oss-120b-1:0')).toEqual({
      family: 'openai', tools: true, images: false, reasoning: 'openai-reasoning',
    })
    expect(modelCapabilities('us.openai.gpt-5.6-2025xx')).toEqual({
      family: 'openai', tools: true, images: true, reasoning: 'openai-reasoning',
    })
  })

  it('grok takes tools + images and drives grok-reasoning', () => {
    expect(modelCapabilities('us.xai.grok-4.6')).toEqual({
      family: 'grok', tools: true, images: true, reasoning: 'grok-reasoning',
    })
  })

  it('unknown families fall back to tools-yes, images-no, reasoning-none', () => {
    expect(modelCapabilities('some.brand-new-model-v1:0')).toEqual({
      family: 'unknown', tools: true, images: false, reasoning: 'none',
    })
  })
})
