package adkcloudflare

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"google.golang.org/adk/model"
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

// GenerateContent is a stub for this task. Replaced by Task 6 with real generate() pipeline.
// Yields nothing (no responses, no error) so the iterator interface stays valid;
// tests that exercise GenerateContent beyond construction live in Task 6.
func (m *cfModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {}
}
