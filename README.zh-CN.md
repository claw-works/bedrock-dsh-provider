# @deepseek-ai/dsh-llm-bedrock

[English](README.md) | **中文**

一个面向 [DeepSeek Harness (dsh)](https://github.com/deepseek-ai/deepseek-harness) LLM capability seam 的 **AWS Bedrock Converse** model provider。用 `ConverseStreamCommand` 走 AWS SDK，认证由 SDK 默认凭证链完成，主要面向 **Anthropic Claude**，并按 model 名字对其它家族（Nova / Llama / DeepSeek 等）的能力差异做判断。

源码结构与 dsh 内置的 `llm-deepseek` provider 完全对齐（同样的 `LlmAdapter` 接口、per-request 配置解析、注册模式），因此可以零改动地放进 monorepo 作为内部包。

## 它做了什么

实现了 dsh LLM seam 的三个角色：

- **Service Provider**：`BedrockAdapter extends LlmAdapter`，唯一必需方法 `stream()` 返回 `AsyncIterable<StreamChunk>`；另实现 `resolveModel()` / `listModels()` / `providerRetryPolicy()`。
- **注册**：`apply()` 里 `ctx.llm.registerConfigurableProviders([...])` + `ctx.llm.registerAdapter(['bedrock'], adapter)`，路由名 `bedrock`。
- **配置**：`Config` schemastery schema，同时作为 `llm-bedrock` user-settings section；连接facts每次请求重新解析，配置变更无需重启即对下一次请求生效。

### 源码文件

| 文件 | 职责 |
|---|---|
| `src/index.ts` | 插件入口、`Config` schema、注册、per-request 配置解析（region/profile/endpoint/catalog） |
| `src/adapter.ts` | `BedrockAdapter`：`ConverseStreamCommand`、SDK 客户端复用、idle watchdog、AWS 错误 → `LlmError` code 映射 |
| `src/serialize.ts` | harness `Message[]` → Converse 入参（`messages` / `system` / `toolConfig` / thinking fields），合并连续同 role turn，tool-result 归入 user 的 `toolResult` block，image block 经 attachment seam 映射为 Converse `image` block |
| `src/translate.ts` | Converse 流事件 → harness `StreamChunk`（block-start/delta/end、usage、finish、stopReason 映射） |
| `src/model.ts` | 按 model id 判断能力（tools / images / reasoning），Claude 优先 |
| `src/replay.ts` | Claude extended-thinking 的 `signature` 通过 `ReplayEnvelope` 携带回放 |
| `src/types.ts` | 从 `@aws-sdk/client-bedrock-runtime` 复用的 wire 类型 |

## 与 DeepSeek provider 的关键差异

| 方面 | DeepSeek | Bedrock（本包） |
|---|---|---|
| 协议 | OpenAI 兼容 REST + SSE | AWS Bedrock Converse（SDK EventStream，SDK 已解码为对象流） |
| 认证 | Bearer token（credentials seam / 环境变量） | **AWS SigV4，SDK 默认凭证链**（不走 dsh credentials seam） |
| 消息内容 | `type` 判别 union | tagged union（`{text}` / `{toolUse}` / `{toolResult}` / `{reasoningContent}`），每成员独立键 |
| system prompt | 一条 `role:system` 消息 | 独立顶层 `system` 字段 |
| tool 结果 | 独立 `role:tool` 消息 | user 消息里的 `toolResult` content block |
| reasoning 回放 | 靠 `reasoning_content` 文本 | 需 Bedrock 下发的 `signature`，通过 replay state 携带 |
| usage 缓存 | `prompt_tokens` 含缓存需减去 | `inputTokens` 已不含缓存，直接用 |

## 能力矩阵（按 model 家族）

`src/model.ts` 里 `modelCapabilities(modelId)` 决定请求映射：

| 家族 | tools | images | reasoning |
|---|---|---|---|
| Claude（3.7 / 4+） | ✅ | ✅ | `claude-thinking`（`additionalModelRequestFields.thinking`） |
| Claude（3.0 / 3.5） | ✅ | ✅ | 无 |
| Nova | ✅ | ✅ | 无 |
| Llama | ✅ | ✅ | 无 |
| DeepSeek / Mistral | ✅ | ❌ | 无 |
| Titan | ❌ | ❌ | 无 |
| unknown | ✅ | ❌ | 无（安全下限） |

> 注：image 输入已实现序列化为 Converse `image` content block，按上表 images 列放行。图片字节从 dsh attachment seam（`ctx.attachments.readImage`）读取，harness 媒体类型（`image/png` · `image/jpeg` · `image/gif` · `image/webp`）与 Converse `format` 1:1 映射。若某家族 images 为 ❌ 却携带图片，请求以 `UNSUPPORTED_CONTENT` 拒绝，错误信息含 model id 与家族。

reasoning effort（`off`/`low`/`high`/`max`）目前统一映射为"是否开启 thinking"+固定 `thinkingBudgetTokens`；如需按 effort 分档预算，改 `serialize.ts` 的 `resolveThinking`。

## 配置

所有字段可选。凭证**不在此配置**——由 AWS SDK 从 `AWS_PROFILE` / `AWS_ACCESS_KEY_ID`+`AWS_SECRET_ACCESS_KEY` / SSO / 容器 / 实例角色解析。

| 字段 | 默认 | 说明 |
|---|---|---|
| `region` | `$AWS_REGION` → `$AWS_DEFAULT_REGION` → SDK 解析 | AWS 区域 |
| `profile` | SDK 默认 | 共享配置 profile 名 |
| `endpoint` | 区域默认 | 覆盖 endpoint（VPC endpoint / gateway） |
| `maxTokens` | 8192 | 默认输出上限，模型自身上限与请求显式值优先 |
| `thinkingBudgetTokens` | 4096 | Claude thinking 通道 token 预算 |
| `defaultContextWindow` | 200000 | 目录未给出时的上下文容量 |
| `models` | 两个 Claude 条目 | 咨询用目录，不限制实际可用 model |
| `streamIdleTimeoutMs` | 300000 | 单次流读空闲超时 |
| `retryPolicy` | normal / 5 次 | provider 级重试策略 |

示例见 [`examples/cordis.yml`](examples/cordis.yml)。

## 如何集成到 dsh

> **前提说明**：dsh 的核心包已作为独立包发布到 npm（`@deepseek-ai/dsh-llm`、`@deepseek-ai/dsh-settings`、`@deepseek-ai/dsh-launch-environment`、`@deepseek-ai/dsh-timeout`、`@deepseek-ai/cordis`、`@deepseek-ai/schemastery`，发布线 `0.0.1-rc.x` / `0.1.1-rc.x`）。因此本包可以作为**第三方 out-of-tree 插件**装进一个用 `npm i -g @deepseek-ai/dsh` 安装的产品版 dsh，无需改动 dsh 源码或 monorepo。本包的 `peerDependencies` 已按发布版本号声明，安装时缺失的 peer（cordis 等）会 fall through 到 dsh 安装自带的依赖，与 dsh 共用同一个 cordis 实例。

### dsh 的插件加载机制（背景）

- `dsh --profile <name>` 启动。profile 是 `$DSH_HOME/profiles/<name>/` 目录，含 `package.json`（声明 out-of-tree 插件）和 `cordis.patch.yml`（用户配置补丁层，热加载）。
- `dsh plugin --profile <name> <pnpm 参数...>` 是一个 **pnpm 转发器**：它在 profile 目录里跑 pnpm，因此支持 npm 包名、本地路径、`file:`、`git+https://...`、tarball 等一切 pnpm add 能接受的 spec。**要求 pnpm 在 PATH 上**（否则报 exit 127）。
- 装进来的包若声明了 `dsh.bundle` 才会自动进 layer；本包是普通 provider 插件（无 `dsh.bundle`），会作为普通依赖装入（安装时会打印一行 "declares no dsh.bundle" 的 warning，属正常），**需要手动在 `cordis.patch.yml` 挂载**（见下）。

### 步骤

以 `headless` profile 为例（`web` 同理，把 `headless` 换成 `web`）。

> **前置：构建工具链。** `dsh plugin ... add` 会在 profile 目录里跑 pnpm，安装时通过本包的 `prepare` 脚本执行 `tsc` 编译。因此需要：
> - `pnpm` 在 PATH 上（否则 exit 127）。若只有 Node，可用 `corepack enable pnpm` 提供。
> - 本包的 `devDependencies` 已包含 `@types/node` 与 `typescript`，`tsconfig` 的 `types: ["node"]` 依赖前者；作为 out-of-tree git/本地包安装时，pnpm 会在隔离目录里按本包自己的 `devDependencies` 构建，缺任一项都会导致 `tsc` 失败。

#### 1. 安装本插件到 profile

**从 GitHub 装（推荐，配合本仓库）：**
```sh
dsh plugin --profile headless add git+https://github.com/<你的用户名>/<仓库名>.git
```
本仓库的 `prepare` 脚本会在安装时自动 `tsc` 编译出 `lib/`。

> pnpm ≥10 默认拦截 git 依赖的 `prepare`（构建）脚本。若安装报错提示某个包的 build 被阻止，按 pnpm 输出的提示，在 `$DSH_HOME/profiles/headless/pnpm-workspace.yaml` 的 `allowBuilds:` 下加上它给出的确切 key，然后重跑。该 key 内含 git tarball 的 commit SHA，**每次仓库有新提交后 key 会变**，需按新的报错更新。

**或从本地目录装（先克隆本仓库到服务器）：**
```sh
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名> && npm install && npm run build
dsh plugin --profile headless add "$(pwd)"
```

**或（将来）本包发布到 npm 后：**
```sh
dsh plugin --profile headless add @deepseek-ai/dsh-llm-bedrock
```

#### 2. 在 profile 的 cordis.patch.yml 挂载 adapter

编辑 `$DSH_HOME/profiles/headless/cordis.patch.yml`（初始内容是 `[]`），改为：
```yaml
- insert:
    - id: llm-bedrock
      name: '@deepseek-ai/dsh-llm-bedrock'
      config:
        region: us-east-1
```
热加载，保存即生效。完整配置字段见上文"配置"表。

#### 3. 把默认模型指向 Bedrock Claude

编辑 `$DSH_HOME/settings.yaml`（热加载）。section key 为 `agent-default-model`，字段 `provider` / `model` / 可选 `reasoningEffort`：
```yaml
agent-default-model:
  provider: bedrock
  model: us.anthropic.claude-sonnet-4-20250514-v1:0
  # reasoningEffort: high   # 开启 Claude extended thinking；省略则关闭
```
> 也可在 `llm-bedrock:` 段覆盖插件配置（如 region），它会覆盖步骤 2 里 cordis.patch.yml 的 config：
> ```yaml
> llm-bedrock:
>   region: us-east-1
> ```

#### 4. 配置 AWS 凭证与模型开通

- **凭证**：本包走 AWS SDK 默认凭证链，服务器上任选其一即可 —— 环境变量 `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`（临时凭证再加 `AWS_SESSION_TOKEN`）、`~/.aws/credentials` 的 profile（配 `profile:` 或 `AWS_PROFILE`）、或 EC2/ECS 的 IAM role。
- **region**：`config.region` > `AWS_REGION` / `AWS_DEFAULT_REGION` > SDK 解析。
- **模型开通**：在目标 region 的 Bedrock 控制台 **Model access** 里申请开通目标 Claude 模型，否则请求会 `AccessDeniedException`。

#### 5. 运行

```sh
dsh --profile headless "介绍一下你自己"
# 或 web：
dsh --profile web
```

### 备选：作为 monorepo 内部包（源码开发时）

若你是在 dsh 源码 monorepo 里开发，也可把本目录拷到 `packages/llm/llm-bedrock`，把 `peerDependencies` / `dependencies` 里的 `@deepseek-ai/*` 版本号换回 `workspace:^`，tsconfig 换成 `extends ../../../tsconfig.base.json` + project references（references：`vendor/cordis`、`vendor/schemastery`、`llm/llm`、`settings/settings`、`util/launch-environment`、`util/timeout`），并在目标 app 的 `package.json` 依赖里声明本包（`verify-cordis-config` 要求裸插件名出现在依赖清单）。然后 `pnpm install && pnpm typecheck && pnpm build`。

## 验证状态

已用**真实 AWS SDK 类型**在严格模式（`strict` + `exactOptionalPropertyTypes` + `noUncheckedIndexedAccess`）下对全部源码做类型检查，dsh 包用照真实签名转写的类型 stub 顶替，结果 **0 error**。这覆盖了：AWS SDK 用法正确性、本包内部逻辑类型自洽、以及对 dsh 接口的调用与其真实签名一致。

装进真实 dsh 后建议先跑一次 `npm run build`（本仓库已含此脚本）做最终编译确认 —— 真实 schemastery `z<T>` schema、真实 `installSettingsSection` 泛型只有在真依赖下才能完全校验。

## 待办 / 后续

- image 输入序列化（`toImage` content block），并接 dsh attachment seam。
- reasoning effort 按档位映射不同 `budget_tokens`。
- provider 级 e2e / snapshot 测试（需真实 AWS 凭证或 mock Bedrock 端点）。
