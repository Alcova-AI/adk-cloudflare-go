# adk-cloudflare-go v0.1 — Design

A Go package implementing `google.golang.org/adk/model.LLM` for Cloudflare Workers AI's OpenAI-compatible chat completions endpoint. Mirrors the structure and feel of `github.com/Alcova-AI/adk-anthropic-go`.

The first internal caller is alcova-backend's docx-editor sub-agent (`pkg/domains/agents/internal/catalog/doceditor.go`), which today runs on Claude Haiku 4.5 and is structurally compatible with kimi-k2.6 (multi-turn tool calling, structured output via `OutputSchema`, no streaming UX). The integration into alcova-backend is a follow-up; this package is the dependency that work consumes.

## Scope

### In scope (v0.1)

- `model.LLM` implementation backed by the OpenAI-compat endpoint
- Non-streaming text generation
- Multi-turn tool calling with stable `tool_call_id` round-trip
- Structured output via `response_format: json_schema` (`strict: false`)
- System instructions and generation params (temperature, top_p, stop_sequences, max_output_tokens)
- Token usage extraction into `UsageMetadata`
- Request-side reasoning support via `ThinkingConfig.ThinkingBudget` → `reasoning_effort`
- Credential env-var fallback
- Optional `BaseURL` override (escape hatch for AI Gateway URLs in a future revision)

### Out of scope (v0.1)

- Streaming (SSE)
- Vision / image input
- Prompt caching
- Response-side reasoning decoding (request-side only; godoc warning documents the half-feature)
- Generic OpenAI-compat positioning — the package is CF-specific in repo, package, and `Config` field names
- Typed error sentinels (callers use `errors.As` against openai-go's typed errors)
- Live CF integration tests in CI

## Architecture

```
caller
   │  m, err := adkcloudflare.NewModel(ctx, modelName, &Config{...})
   ▼
adkcloudflare.cfModel  (implements google.golang.org/adk/model.LLM)
   │  Name() string                                            → returns modelName
   │  GenerateContent(ctx, *LLMRequest, stream bool) iter.Seq2 → yields exactly one (resp, err)
   │      stream argument is accepted for interface conformance
   │      and ignored — non-streaming only in v0.1
   │
   ├── converters.BuildRequest(modelName, req)  → openai.ChatCompletionNewParams
   │
   ├── openai-go client
   │     POST {Config.BaseURL or default}/chat/completions
   │     default: https://api.cloudflare.com/client/v4/accounts/{AccountID}/ai/v1/
   │     Authorization: Bearer {APIToken}
   │
   └── converters.CompletionToLLMResponse(*openai.ChatCompletion) → *model.LLMResponse
```

## Repo layout

```
~/work/adk-cloudflare-go/
├── cloudflare.go              # cfModel + NewModel + GenerateContent + generate()
├── config.go                  # Config struct, env-var fallback, validation
├── doc.go                     # package godoc / usage example / thinking-config warning
├── cloudflare_test.go         # public-API tests against a fake http.RoundTripper
├── converters/
│   ├── request.go             # genai.Content[]+Config → ChatCompletionNewParams
│   ├── response.go            # ChatCompletion → *model.LLMResponse
│   ├── tools.go               # FunctionDeclaration → tool param; tool_choice mapping; SchemaToMap
│   └── converters_test.go     # table-driven tests on every conversion function
├── go.mod
├── README.md
└── .github/
    └── workflows/
        └── ci.yaml            # go test + go vet on push and PR
```

No `LICENSE` / `NOTICE` / `CHANGELOG.md` for v0.1. Source files carry a bare `// Copyright 2026 Alcova AI` header (no license declaration). veadk-go attribution lives inline as a comment block in `converters/request.go` and `converters/response.go`.

## Component design

### `Config` (`config.go`)

```go
type Config struct {
    AccountID  string        // CF account ID (env: CLOUDFLARE_ACCOUNT_ID)
    APIToken   string        // CF API token (env: CLOUDFLARE_API_TOKEN)
    BaseURL    string        // optional override; defaults to api.cloudflare.com path
    MaxTokens  int           // default max_tokens when request omits it; defaults to 16384
    HTTPClient *http.Client  // optional; defaults to http.DefaultClient
}
```

Construction-time error rules:

| Condition | Behaviour |
|---|---|
| `Config == nil` | treat as `&Config{}`, fall through to env-var fallback |
| `AccountID` empty AND `CLOUDFLARE_ACCOUNT_ID` env empty | error: `"cloudflare: account ID is required (set Config.AccountID or CLOUDFLARE_ACCOUNT_ID)"` |
| `APIToken` empty AND `CLOUDFLARE_API_TOKEN` env empty | error: `"cloudflare: API token is required (set Config.APIToken or CLOUDFLARE_API_TOKEN)"` |
| `modelName` empty | error: `"cloudflare: model name is required"` |
| `Config.HTTPClient == nil` | use `http.DefaultClient` |
| `Config.MaxTokens == 0` | use default `16384` |
| `Config.BaseURL` empty | construct `https://api.cloudflare.com/client/v4/accounts/{AccountID}/ai/v1/` |
| `Config.BaseURL` set | use verbatim; openai-go appends `chat/completions` |

### `cfModel` (`cloudflare.go`)

Implements `model.LLM`. Holds the openai-go client, the model name, and the resolved default `MaxTokens`. `GenerateContent` returns an `iter.Seq2[*model.LLMResponse, error]` that yields exactly one `(resp, err)` regardless of the `stream` argument (parity with `adk-anthropic-go`'s non-stream branch).

Internal `generate(ctx, req)` is the real work:

1. `params, err := converters.BuildRequest(m.name, req)` — fail-fast on conversion errors before any HTTP call. Wrap as `"converting request: %w"`.
2. `completion, err := m.client.Chat.Completions.New(ctx, params)` — wrap as `"calling model: %w"`. openai-go's typed `*openai.Error` is preserved through `errors.As`.
3. `resp, err := converters.CompletionToLLMResponse(completion)` — wrap as `"converting response: %w"`.

### Request converter (`converters/request.go`)

`BuildRequest(modelName string, req *model.LLMRequest) (openai.ChatCompletionNewParams, error)`.

Genai `Content` → OpenAI message rules:

| Genai input | OpenAI output |
|---|---|
| `Config.SystemInstruction` | `role: "system"` message at index 0 (always first when present) |
| `Content{Role:"user", Parts:[Text]}` | `role:"user"`, `content: text` |
| `Content{Role:"user", Parts:[FunctionResponse(id1)]}` | `role:"tool"`, `tool_call_id: id1`, `content: json.Marshal(response)` |
| `Content{Role:"user", Parts:[FR1, FR2, FR3]}` | three separate `role:"tool"` messages, order preserved |
| `Content{Role:"model", Parts:[Text]}` | `role:"assistant"`, `content: text` |
| `Content{Role:"model", Parts:[FunctionCall(id1)]}` | `role:"assistant"`, `content: null`, `tool_calls: [{id:id1, type:"function", function:{name, arguments: json.Marshal(args)}}]` |
| `Content{Role:"model", Parts:[Text, FunctionCall, FunctionCall]}` | ONE assistant message with `content: text` AND both tool calls in `tool_calls[]`, ids preserved, order preserved |
| `Content{Role:""}` | default to `"user"`, no error |
| Multiple `Text` parts in one `Content` | concatenate with `\n` (avoids the silent-data-loss veadk-go has by taking only the first) |
| `Part.InlineData` / `Part.FileData` / `Part.Thought` | drop part, debug-level log, no error |

Generation parameter mapping:

| Genai field | OpenAI param | Notes |
|---|---|---|
| `Temperature` | `temperature` | |
| `TopP` | `top_p` | |
| `TopK` | — | dropped, godoc note (CF/OpenAI compat doesn't accept) |
| `StopSequences` | `stop` | |
| `MaxOutputTokens` | `max_tokens` | falls back to `Config.MaxTokens` (default 16384) |
| `SystemInstruction` | first `role:"system"` message | |
| `Tools` | `tools[]` | see Tool converter |
| `ToolConfig` | `tool_choice` | see Tool converter |
| `ResponseSchema` | `response_format` | `{type:"json_schema", json_schema:{name:"response", schema, strict:false}}` |
| `ThinkingConfig.ThinkingBudget` | `reasoning_effort` | thresholds below |
| `ThinkingConfig.IncludeThoughts` | — | silently ignored; godoc warns response-side reasoning is not decoded in v0.1 |

Thinking-budget threshold mapping:

```
ThinkingConfig nil             →  reasoning_effort unset (model default)
ThinkingBudget nil             →  "medium"
ThinkingBudget == 0            →  "low"
ThinkingBudget < 4096          →  "low"
ThinkingBudget < 16384         →  "medium"
ThinkingBudget >= 16384        →  "high"
```

### Response converter (`converters/response.go`)

`CompletionToLLMResponse(c *openai.ChatCompletion) (*model.LLMResponse, error)`.

OpenAI → genai mapping:

| OpenAI | genai |
|---|---|
| `Choice[0].Message.Content` | `Part{Text: content}` (omitted if empty) |
| `Choice[0].Message.ToolCalls[i]` | `Part{FunctionCall{ID: tc.ID, Name: tc.Function.Name, Args: json.Unmarshal(tc.Function.Arguments)}}` |
| `Usage.PromptTokens` | `UsageMetadata.PromptTokenCount` |
| `Usage.CompletionTokens` | `UsageMetadata.CandidatesTokenCount` |
| `Usage.TotalTokens` | `UsageMetadata.TotalTokenCount` |
| `Choice[0].FinishReason` | `FinishReason` (table below) |

Finish-reason mapping:

| OpenAI | genai |
|---|---|
| `stop` | `STOP` |
| `tool_calls` | `STOP` (matches adk-anthropic-go's treatment of `tool_use`) |
| `length` | `MAX_TOKENS` |
| `content_filter` | `SAFETY` |
| anything else / empty | `OTHER` |

Edge cases:

- Empty `choices[]` → error: `"converting response: no choices in response"`
- `Content == nil` AND `len(ToolCalls) == 0` → return `*LLMResponse{Content: &Content{Role:"model", Parts: nil}, ...}`. Empty content is valid.
- `tool_calls[i].Function.Arguments` not parseable JSON object → error: `"converting response: parsing tool call args for %q (id=%s): %w"`
- Mixed text + tool_calls in one response → Parts ordered `[Text, FunctionCall, ...]` (Text first, then function calls in order)
- `usage` field missing → leave `UsageMetadata` nil, no error
- Unknown `finish_reason` → map to `genai.FinishReasonOther`, no error

### Tool converter (`converters/tools.go`)

```go
FunctionDeclarationsToTools([]*genai.Tool) []openai.ChatCompletionToolUnionParam
ToolConfigToToolChoice(*genai.ToolConfig) openai.ChatCompletionToolChoiceOptionUnionParam
SchemaToMap(*genai.Schema) map[string]any
```

Tool-choice mapping:

| Genai mode | OpenAI |
|---|---|
| `AUTO` (or unset) | `"auto"` |
| `ANY` | `"required"` |
| `NONE` | `"none"` |
| `ANY` + single `AllowedFunctionNames` | `{type:"function", function:{name}}` |
| `ANY` + multiple allowed names | `"required"` (OpenAI doesn't support a name allow-list; widen to `"required"` and rely on the model picking from declared tools) |

`extractFunctionParams` follows adk-anthropic-go's pattern — supports both `Parameters *genai.Schema` and `ParametersJsonSchema any` (map or `*jsonschema.Schema`).

### `cloudflare_test.go`

End-to-end tests using a recording `http.RoundTripper` injected via `Config.HTTPClient`. No `httptest.Server` (no port management; cleaner request-body assertions).

Standard library `testing` + `github.com/google/go-cmp/cmp` for struct diff. No testify.

## Error handling

All errors return as Go errors via `fmt.Errorf("...: %w", err)`. No typed sentinels in v0.1.

```
NewModel construction errors   →  "cloudflare: <reason>"
Per-call generation errors     →  "<phase>: %w"
                                   converting request: ...
                                   calling model: ...           (openai-go errors bubble through)
                                   converting response: ...
```

We rely on openai-go for:
- `Authorization: Bearer` header
- Retry on 429 + 5xx with exponential backoff (default 2 retries; we don't override)
- Request body marshaling
- Response body parsing
- Context cancellation cancelling in-flight requests
- Per-request timeout (caller via `Config.HTTPClient` if needed)

We do not expose `Config.MaxRetries` in v0.1. Callers who need to disable retries pass an `HTTPClient` whose transport short-circuits.

## Things this design explicitly does not do

| Not doing | Why |
|---|---|
| Validate that `tool_call_id`s in `FunctionResponse` parts match prior `FunctionCall` IDs in the same conversation | The agent runtime's job; CF surfaces the error if invalid. We see one request at a time. |
| De-duplicate same-ID `tool_calls[]` in a response | Shouldn't happen; pass through. |
| Coerce non-object tool arguments | Errors per the table. Caller's prompt is the bug, not ours. |
| Mask raw error text from openai-go | The wrapped HTTP error often carries the CF response body, which is the most useful debugging info. |
| Auto-generate missing `tool_call_id`s | Strict by default; fail loudly. |

## Testing strategy

### Required test scenarios

**Converter tests (`converters/converters_test.go`)**

| Scenario | What it asserts |
|---|---|
| `system_instruction_becomes_first_message` | `Config.SystemInstruction` produces a `role:"system"` message at index 0 |
| `user_content_becomes_user_message` | Plain text `Content{Role:"user"}` → `{role:"user", content:text}` |
| `model_text_becomes_assistant_message` | `Content{Role:"model"}` → `{role:"assistant"}` |
| `function_call_in_model_content` | `Part.FunctionCall` with id → `assistant` message with `tool_calls[]`, id preserved, args JSON-stringified |
| `mixed_text_and_tool_calls` | `[Text, FunctionCall, FunctionCall]` → ONE assistant message with `content=text` AND both tool calls in `tool_calls[]` in order |
| `function_response_becomes_tool_message` | `Part.FunctionResponse` with id → `role:"tool"` message with matching `tool_call_id`, JSON-marshaled content |
| `parallel_function_responses_fan_out` | Two `FunctionResponse` parts in one `Content` → two separate tool messages, order preserved |
| `function_response_empty_id_errors` | Empty `ID` returns the documented error |
| `multiple_text_parts_concatenate` | `[Text, Text]` joined with `\n` (not silently dropped) |
| `unsupported_part_silently_dropped` | `InlineData` / `FileData` / `Thought` parts skipped, no error |
| `response_schema_emits_response_format` | `Config.ResponseSchema` set → `response_format.json_schema.strict == false`, schema converted via `SchemaToMap` |
| `response_schema_with_optional_subobject` | Schema mirroring the doc editor's optional `file_ref` round-trips (strict=false makes this valid) |
| `tool_choice_modes` | AUTO / ANY / NONE / single-named → `"auto"` / `"required"` / `"none"` / `{type:function, function:{name}}` |
| `thinking_config_budget_thresholds` | `nil` / `0` / `<4096` / `<16384` / `>=16384` → `unset` / `"low"` / `"low"` / `"medium"` / `"high"` |
| `topk_dropped` | `Config.TopK` set is silently ignored (no field in request) |
| `generation_params_round_trip` | Temperature / TopP / StopSequences / MaxOutputTokens map verbatim |
| `usage_extracted` | `usage` object → `UsageMetadata` |
| `finish_reason_mapping` | `stop` / `tool_calls` / `length` / `content_filter` / unknown → `STOP` / `STOP` / `MAX_TOKENS` / `SAFETY` / `OTHER` |
| `tool_call_args_unparseable_errors` | `arguments` field that isn't valid JSON object returns the documented error |
| `mixed_response_text_and_tool_calls` | Response with both `content` and `tool_calls[]` → Parts ordered `[Text, FunctionCall, ...]` |

**Top-level integration tests (`cloudflare_test.go`)**

| Scenario | What it asserts |
|---|---|
| `new_model_requires_account_id` | Missing AccountID + missing env var → error |
| `new_model_requires_api_token` | Missing APIToken + missing env var → error |
| `new_model_uses_env_var_fallback` | Env vars populate Config when fields empty |
| `new_model_custom_base_url` | `Config.BaseURL` overrides default |
| `name_returns_model_name` | `Name()` returns the constructor's `modelName` |
| `generate_content_yields_once` | The iter.Seq2 yields exactly one `(resp, err)` regardless of `stream` arg |
| `multi_turn_tool_loop_preserves_ids` | Gating test: turn 1 (user) → turn 2 (assistant with 2 parallel tool_calls) → turn 3 (tool results) → turn 4 (assistant final). Asserts request bodies on turn 2→3 round-trip both ids verbatim and the model receives `role:"tool"` messages in order. Uses fake transport scripted with three responses. |
| `error_wrapping_includes_phase` | A 4xx from CF → returned error matches `errors.As(&openai.Error)` AND error text contains `"calling model:"` prefix |
| `request_conversion_error_short_circuits` | `FunctionResponse` with empty ID → error before any HTTP call (fake transport never invoked) |

### Out of scope for tests

- Live CF API tests requiring a real `CLOUDFLARE_API_TOKEN` (the alcova-backend pilot covers end-to-end "does it actually work")
- openai-go internals (retries, header construction, JSON marshaling) — covered by openai-go's own tests
- Streaming (non-streaming only)

## CI

`.github/workflows/ci.yaml` runs `go test ./...` and `go vet ./...` on push and PR. Mirrors adk-anthropic-go's CI shape.

## Things lifted from veadk-go

Not the wire structs (we're using `openai-go`), but the *converter logic* in two specific places, marked with an inline comment block:

- `converters/request.go` — the genai.Content → OpenAI message decomposition, particularly the assistant-message-with-text-and-tool_calls case and the FunctionResponse → role:tool fan-out
- `converters/response.go` — the tool_calls reassembly into Parts with id preservation

Inline comment in each file: `// Portions adapted from github.com/volcengine/veadk-go (Apache 2.0).`

## Future work (not v0.1)

- Streaming support (SSE)
- Response-side reasoning decoding (when a specific reasoning model is targeted; see godoc warning in `doc.go`)
- AI Gateway URL form (the `Config.BaseURL` escape hatch already lets callers point at it manually)
- Vision / image input
- Prompt caching
- Generic `adk-openai-go` extraction if a non-CF caller emerges
