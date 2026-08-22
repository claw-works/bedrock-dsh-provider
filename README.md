# @deepseek-ai/dsh-llm-bedrock

**English** | [中文](README.zh-CN.md)

An **AWS Bedrock Converse** model provider for the [DeepSeek Harness (dsh)](https://github.com/deepseek-ai/deepseek-harness) LLM capability seam. It drives the AWS SDK via `ConverseStreamCommand`, lets the SDK's default credential chain handle authentication, targets **Anthropic Claude** primarily, and switches behaviour by model name to accommodate the capability differences of other families (Nova / Llama / DeepSeek, etc.).

The source layout mirrors dsh's built-in `llm-deepseek` provider exactly (same `LlmAdapter` interface, per-request config resolution, registration pattern), so it drops into the monorepo as an internal package with zero changes.

## What it does

It implements the three roles of the dsh LLM seam:

- **Service Provider**: `BedrockAdapter extends LlmAdapter`; the sole required method `stream()` returns an `AsyncIterable<StreamChunk>`. It also implements `resolveModel()` / `listModels()` / `providerRetryPolicy()`.
- **Registration**: inside `apply()`, `ctx.llm.registerConfigurableProviders([...])` + `ctx.llm.registerAdapter(['bedrock'], adapter)`, with the route name `bedrock`.
- **Config**: a `Config` schemastery schema that doubles as the `llm-bedrock` user-settings section. Connection facts are re-resolved on every request, so a config change takes effect on the next request without a restart.

### Source files

| File | Responsibility |
|---|---|
| `src/index.ts` | Plugin entry, `Config` schema, registration, per-request config resolution (region/profile/endpoint/catalog) |
| `src/adapter.ts` | `BedrockAdapter`: `ConverseStreamCommand`, SDK client reuse, idle watchdog, AWS error → `LlmError` code mapping |
| `src/serialize.ts` | harness `Message[]` → Converse inputs (`messages` / `system` / `toolConfig` / thinking fields), merges consecutive same-role turns, folds tool results into the user's `toolResult` block |
| `src/translate.ts` | Converse stream events → harness `StreamChunk` (block-start/delta/end, usage, finish, stopReason mapping) |
| `src/model.ts` | Per-model capability judgement by model id (tools / images / reasoning), Claude first |
| `src/replay.ts` | Carries Claude extended-thinking `signature` back through a `ReplayEnvelope` for replay |
| `src/types.ts` | Wire types reused from `@aws-sdk/client-bedrock-runtime` |

## Key differences from the DeepSeek provider

| Aspect | DeepSeek | Bedrock (this package) |
|---|---|---|
| Protocol | OpenAI-compatible REST + SSE | AWS Bedrock Converse (SDK EventStream, already decoded into an object stream by the SDK) |
| Auth | Bearer token (credentials seam / env vars) | **AWS SigV4, SDK default credential chain** (does not use the dsh credentials seam) |
| Message content | `type` discriminated union | tagged union (`{text}` / `{toolUse}` / `{toolResult}` / `{reasoningContent}`), each member its own key |
| System prompt | one `role:system` message | dedicated top-level `system` field |
| Tool results | separate `role:tool` messages | a `toolResult` content block inside a user message |
| Reasoning replay | relies on `reasoning_content` text | needs the `signature` emitted by Bedrock, carried through replay state |
| Usage caching | `prompt_tokens` includes cache, must be subtracted | `inputTokens` already excludes cache, used directly |

## Capability matrix (by model family)

`modelCapabilities(modelId)` in `src/model.ts` decides the request mapping:

| Family | tools | images | reasoning |
|---|---|---|---|
| Claude (3.7 / 4+) | ✅ | ✅ | `claude-thinking` (`additionalModelRequestFields.thinking`) |
| Claude (3.0 / 3.5) | ✅ | ✅ | none |
| Nova | ✅ | ✅ | none |
| Llama | ✅ | ✅ | none |
| DeepSeek / Mistral | ✅ | ❌ | none |
| Titan | ❌ | ❌ | none |
| unknown | ✅ | ❌ | none (safe floor) |

> Note: image input serialization is not yet implemented in this first version (`serialize.ts` throws `UNSUPPORTED_CONTENT` on image content). The `images` column above indicates whether the model itself supports images, reserved for when image serialization is added.

reasoning effort (`off`/`low`/`high`/`max`) currently maps uniformly to "whether to enable thinking" plus a fixed `thinkingBudgetTokens`. To budget per effort tier, edit `resolveThinking` in `serialize.ts`.

## Configuration

Every field is optional. Credentials are **not** in this config — the AWS SDK resolves them from `AWS_PROFILE` / `AWS_ACCESS_KEY_ID`+`AWS_SECRET_ACCESS_KEY` / SSO / container / instance role.

| Field | Default | Description |
|---|---|---|
| `region` | `$AWS_REGION` → `$AWS_DEFAULT_REGION` → SDK resolution | AWS region |
| `profile` | SDK default | shared-config profile name |
| `endpoint` | regional default | override endpoint (VPC endpoint / gateway) |
| `maxTokens` | 8192 | default output cap; the model's own cap and an explicit per-request value win |
| `thinkingBudgetTokens` | 4096 | token budget for the Claude thinking channel |
| `defaultContextWindow` | 200000 | context capacity used when the catalog does not provide one |
| `models` | two Claude entries | advisory catalog; does not restrict which models can actually be used |
| `streamIdleTimeoutMs` | 300000 | idle timeout for a single stream read |
| `retryPolicy` | normal / 5 attempts | provider-level retry policy |

See [`examples/cordis.yml`](examples/cordis.yml) for an example.

## How to integrate into dsh

> **Prerequisite**: dsh's core packages are published to npm as standalone packages (`@deepseek-ai/dsh-llm`, `@deepseek-ai/dsh-settings`, `@deepseek-ai/dsh-launch-environment`, `@deepseek-ai/dsh-timeout`, `@deepseek-ai/cordis`, `@deepseek-ai/schemastery`, release lines `0.0.1-rc.x` / `0.1.1-rc.x`). This package can therefore be installed as a **third-party out-of-tree plugin** into a production dsh installed via `npm i -g @deepseek-ai/dsh`, without touching dsh source or the monorepo. This package's `peerDependencies` are declared against the published version numbers; a peer missing at install time (cordis, etc.) falls through to the dependency dsh's install already carries, sharing the same cordis instance as dsh.

### How dsh loads plugins (background)

- `dsh --profile <name>` starts up. A profile is the `$DSH_HOME/profiles/<name>/` directory, containing `package.json` (declares out-of-tree plugins) and `cordis.patch.yml` (a hot-reloaded user config patch layer).
- `dsh plugin --profile <name> <pnpm args...>` is a **pnpm forwarder**: it runs pnpm inside the profile directory, so it accepts any spec `pnpm add` accepts — npm package names, local paths, `file:`, `git+https://...`, tarballs, etc. **pnpm must be on PATH** (otherwise it fails with exit 127).
- An installed package only auto-joins the layer if it declares `dsh.bundle`. This package is a plain provider plugin (no `dsh.bundle`), so it installs as an ordinary dependency (a "declares no dsh.bundle" warning prints at install time — this is normal) and **must be mounted manually in `cordis.patch.yml`** (see below).

### Steps

Using the `headless` profile as an example (`web` is identical — replace `headless` with `web`).

> **Prerequisite: build toolchain.** `dsh plugin ... add` runs pnpm inside the profile directory, which compiles this package via `tsc` through its `prepare` script at install time. So you need:
> - `pnpm` on PATH (otherwise exit 127). If you only have Node, `corepack enable pnpm` provides it.
> - This package's `devDependencies` already include `@types/node` and `typescript`; the tsconfig's `types: ["node"]` depends on the former. When installed as an out-of-tree git/local package, pnpm builds it in an isolated directory against this package's own `devDependencies`, and a missing one makes `tsc` fail.

#### 1. Install this plugin into the profile

**From GitHub (recommended, pairs with this repo):**
```sh
dsh plugin --profile headless add git+https://github.com/<your-username>/<repo-name>.git
```
This repo's `prepare` script runs `tsc` at install time to produce `lib/`.

> pnpm ≥10 blocks the `prepare` (build) script of git dependencies by default. If the install errors saying a package's build was blocked, follow pnpm's output and add the exact key it gives under `allowBuilds:` in `$DSH_HOME/profiles/headless/pnpm-workspace.yaml`, then rerun. That key embeds the git tarball's commit SHA, so **the key changes on every new commit to the repo** and must be updated per the new error.

**Or from a local directory (clone this repo to the server first):**
```sh
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name> && npm install && npm run build
dsh plugin --profile headless add "$(pwd)"
```

**Or (in future) once this package is published to npm:**
```sh
dsh plugin --profile headless add @deepseek-ai/dsh-llm-bedrock
```

#### 2. Mount the adapter in the profile's cordis.patch.yml

Edit `$DSH_HOME/profiles/headless/cordis.patch.yml` (initial content is `[]`) to:
```yaml
- insert:
    - id: llm-bedrock
      name: '@deepseek-ai/dsh-llm-bedrock'
      config:
        region: us-east-1
```
It hot-reloads — saving takes effect immediately. See the "Configuration" table above for all fields.

#### 3. Point the default model at Bedrock Claude

Edit `$DSH_HOME/settings.yaml` (hot-reloaded). The section key is `agent-default-model`, with fields `provider` / `model` / optional `reasoningEffort`:
```yaml
agent-default-model:
  provider: bedrock
  model: us.anthropic.claude-sonnet-4-20250514-v1:0
  # reasoningEffort: high   # enables Claude extended thinking; omit to disable
```
> You can also override the plugin config in the `llm-bedrock:` section (e.g. region), which overrides the config in step 2's cordis.patch.yml:
> ```yaml
> llm-bedrock:
>   region: us-east-1
> ```

#### 4. Configure AWS credentials and model access

- **Credentials**: this package uses the AWS SDK default credential chain — any one on the server works: env vars `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (plus `AWS_SESSION_TOKEN` for temporary credentials), a `~/.aws/credentials` profile (set `profile:` or `AWS_PROFILE`), or an EC2/ECS IAM role.
- **region**: `config.region` > `AWS_REGION` / `AWS_DEFAULT_REGION` > SDK resolution.
- **Model access**: request access to the target Claude model under **Model access** in the Bedrock console for the target region, otherwise requests fail with `AccessDeniedException`.

#### 5. Run

```sh
dsh --profile headless "Introduce yourself"
# or web:
dsh --profile web
```

### Alternative: as a monorepo internal package (during source development)

If you develop inside the dsh source monorepo, you can copy this directory to `packages/llm/llm-bedrock`, change the `@deepseek-ai/*` versions in `peerDependencies` / `dependencies` back to `workspace:^`, switch the tsconfig to `extends ../../../tsconfig.base.json` + project references (references: `vendor/cordis`, `vendor/schemastery`, `llm/llm`, `settings/settings`, `util/launch-environment`, `util/timeout`), and declare this package in the target app's `package.json` dependencies (`verify-cordis-config` requires the bare plugin name to appear in the dependency list). Then run `pnpm install && pnpm typecheck && pnpm build`.

## Verification status

All source has been type-checked with the **real AWS SDK types** under strict mode (`strict` + `exactOptionalPropertyTypes` + `noUncheckedIndexedAccess`), with dsh packages substituted by type stubs transcribed from their real signatures, yielding **0 errors**. This covers: correctness of AWS SDK usage, internal type consistency of this package, and that calls into the dsh interface match its real signatures.

After installing into a real dsh, it is recommended to run `npm run build` once (this repo includes the script) for a final compile confirmation — the real schemastery `z<T>` schema and the real `installSettingsSection` generics can only be fully validated against the real dependencies.

## TODO / follow-ups

- Image input serialization (`toImage` content block), wired to the dsh attachment seam.
- Map reasoning effort tiers to different `budget_tokens`.
- Provider-level e2e / snapshot tests (requires real AWS credentials or a mock Bedrock endpoint).
