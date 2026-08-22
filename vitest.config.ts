import { defineConfig } from 'vitest/config'

/**
 * Vitest is the adapter's only test runner. Every suite here is a pure-function
 * unit / snapshot test that runs offline with no AWS credentials; the one real
 * Bedrock e2e suite is env-gated (`BEDROCK_E2E`) and skips by default, so `npm
 * test` is green in CI without any cloud access.
 */
export default defineConfig({
  test: {
    // Node environment: the code under test is server-side, no DOM.
    environment: 'node',
    include: ['test/**/*.test.ts'],
    // Keep the snapshot files beside the suites under test/.
    globals: false,
  },
})
