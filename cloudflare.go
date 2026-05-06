package adkcloudflare

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"google.golang.org/adk/model"

	"github.com/Alcova-AI/adk-cloudflare-go/converters"
)

// Compile-time assertion that *cfModel satisfies model.LLM.
var _ model.LLM = (*cfModel)(nil)

type cfModel struct {
	client    openai.Client
	name      string
	maxTokens int
}

// NewModel creates a new Cloudflare Workers AI model with the given name and configuration.
func NewModel(ctx context.Context, modelName string, cfg *Config) (model.LLM, error) {
	if modelName == "" {
		return nil, errors.New("cloudflare: model name is required")
	}
	if cfg == nil {
		cfg = &Config{}
	}
	resolved, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	base := resolved.BaseURL
	if base == "" {
		base = fmt.Sprintf("https://api.cloudflare.com/client/v4/accounts/%s/ai/v1/", resolved.AccountID)
	}
	client := openai.NewClient(
		option.WithBaseURL(base),
		option.WithAPIKey(resolved.APIToken),
		option.WithHTTPClient(resolved.HTTPClient),
	)
	return &cfModel{client: client, name: modelName, maxTokens: resolved.MaxTokens}, nil
}

func (m *cfModel) Name() string {
	return m.name
}

// GenerateContent calls the Cloudflare Workers AI chat completions endpoint and
// yields exactly one (response, error) pair. The stream argument is ignored —
// only non-streaming completions are supported in v0.1.
func (m *cfModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

func (m *cfModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	params, err := converters.BuildRequest(m.name, req)
	if err != nil {
		return nil, fmt.Errorf("converting request: %w", err)
	}

	// Per-instance max_tokens default. BuildRequest only sets max_tokens when
	// req.Config.MaxOutputTokens is explicitly set; if it didn't, fall back to
	// the value resolved at NewModel time (Config.MaxTokens or the 16384 default).
	if !params.MaxTokens.Valid() {
		params.MaxTokens = param.NewOpt(int64(m.maxTokens))
	}

	completion, err := m.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("calling model: %w", err)
	}

	resp, err := converters.CompletionToLLMResponse(completion)
	if err != nil {
		return nil, fmt.Errorf("converting response: %w", err)
	}
	return resp, nil
}
