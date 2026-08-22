/**
 * Optional real-Bedrock end-to-end smoke test. Skipped by default so `npm test`
 * is green in CI with no AWS credentials; opt in by setting `BEDROCK_E2E=1`
 * (and providing AWS credentials + `BEDROCK_E2E_MODEL` / `AWS_REGION`) to run a
 * single live streaming turn against the Converse API.
 *
 * This suite deliberately holds no assertion when it is skipped — the offline
 * suites own correctness; this one only proves the wiring reaches a live
 * endpoint when an operator asks for it.
 */

import { describe, expect, it } from 'vitest'

const E2E_ENABLED = process.env.BEDROCK_E2E === '1' || process.env.BEDROCK_E2E === 'true'

describe.skipIf(!E2E_ENABLED)('bedrock e2e (live, opt-in via BEDROCK_E2E)', () => {
  it('streams a short completion from a real Bedrock model', async () => {
    const region = process.env.AWS_REGION ?? 'us-east-1'
    const model = process.env.BEDROCK_E2E_MODEL ?? 'anthropic.claude-3-5-haiku-20241022-v1:0'

    // Imported lazily so the SDK client is only constructed when the suite runs.
    const { BedrockRuntimeClient, ConverseStreamCommand } = await import('@aws-sdk/client-bedrock-runtime')
    const { serializeRequest } = await import('../src/serialize.js')
    const { modelCapabilities } = await import('../src/model.js')
    const { translate } = await import('../src/translate.js')

    const input = await serializeRequest(
      {
        provider: 'bedrock',
        model,
        messages: [{ id: 'm1' as never, role: 'user', content: [{ type: 'text', text: 'Say "pong" and nothing else.' }], source: { kind: 'user' } }],
      },
      modelCapabilities(model),
      { thinkingBudgetTokens: 1024 },
      async () => { throw new Error('e2e smoke test sends no images') },
    )

    const client = new BedrockRuntimeClient({ region })
    const response = await client.send(new ConverseStreamCommand(input))

    let text = ''
    let finished = false
    for await (const chunk of translate(response.stream!)) {
      if (chunk.type === 'text-delta') text += chunk.text
      if (chunk.type === 'finish') finished = true
    }
    expect(finished).toBe(true)
    expect(text.length).toBeGreaterThan(0)
  }, 60_000)
})
