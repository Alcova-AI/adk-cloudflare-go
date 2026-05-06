package adkcloudflare

import (
	"context"
	"strings"
	"testing"

	"google.golang.org/adk/model"
)

func TestNewModelRequiresAccountID(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	_, err := NewModel(context.Background(), "test-model", &Config{APIToken: "x"})

	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "account ID is required") {
		t.Fatalf("expected error containing 'account ID is required', got: %v", err)
	}
}

func TestNewModelRequiresAPIToken(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	_, err := NewModel(context.Background(), "test-model", &Config{AccountID: "x"})

	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "API token is required") {
		t.Fatalf("expected error containing 'API token is required', got: %v", err)
	}
}

func TestNewModelRequiresModelName(t *testing.T) {
	_, err := NewModel(context.Background(), "", &Config{
		AccountID: "account",
		APIToken:  "token",
	})

	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "model name is required") {
		t.Fatalf("expected error containing 'model name is required', got: %v", err)
	}
}

func TestNewModelUsesEnvVarFallback(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "env-account-id")
	t.Setenv("CLOUDFLARE_API_TOKEN", "env-api-token")

	m, err := NewModel(context.Background(), "test-model", &Config{})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m == nil {
		t.Fatal("expected non-nil model")
	}
}

func TestNewModelDefaultMaxTokens(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	m, err := NewModel(context.Background(), "test-model", &Config{
		AccountID: "account",
		APIToken:  "token",
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m == nil {
		t.Fatal("expected non-nil model")
	}

	// Cast to *cfModel to check unexported field
	cfm := m.(*cfModel)
	if cfm.maxTokens != 16384 {
		t.Fatalf("expected maxTokens=16384, got %d", cfm.maxTokens)
	}
}

func TestNewModelCustomMaxTokensHonoured(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	m, err := NewModel(context.Background(), "test-model", &Config{
		AccountID: "account",
		APIToken:  "token",
		MaxTokens: 4096,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m == nil {
		t.Fatal("expected non-nil model")
	}

	cfm := m.(*cfModel)
	if cfm.maxTokens != 4096 {
		t.Fatalf("expected maxTokens=4096, got %d", cfm.maxTokens)
	}
}

func TestNameReturnsModelName(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	m, err := NewModel(context.Background(), "my-test-model", &Config{
		AccountID: "account",
		APIToken:  "token",
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if m.Name() != "my-test-model" {
		t.Fatalf("expected name 'my-test-model', got %q", m.Name())
	}
}

func TestGenerateContentStubYieldsNothing(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	m, err := NewModel(context.Background(), "test-model", &Config{
		AccountID: "account",
		APIToken:  "token",
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Call GenerateContent and count yields
	yields := 0
	for resp, err := range m.GenerateContent(context.Background(), &model.LLMRequest{}, false) {
		_ = resp
		_ = err
		yields++
	}

	if yields != 0 {
		t.Fatalf("expected 0 yields from stub, got %d", yields)
	}
}
