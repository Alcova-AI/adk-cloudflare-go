package adkcloudflare

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
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

// fakeTransport records requests and returns canned HTTP responses in order.
type fakeTransport struct {
	requests  []*http.Request
	bodies    [][]byte
	responses []*http.Response // queued; dequeued in order on each RoundTrip
}

func (f *fakeTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(r.Body)
	f.requests = append(f.requests, r)
	f.bodies = append(f.bodies, body)
	if len(f.responses) == 0 {
		return nil, errors.New("no canned response")
	}
	resp := f.responses[0]
	f.responses = f.responses[1:]
	return resp, nil
}

// cannedJSONResponse creates a 200 response with JSON content-type for the given body.
func cannedJSONResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}
}

// minimalCompletion is a valid ChatCompletion JSON response body for testing.
const minimalCompletion = `{
	"id": "chatcmpl-test",
	"object": "chat.completion",
	"created": 1700000000,
	"model": "test-model",
	"choices": [{
		"index": 0,
		"message": {"role": "assistant", "content": "done"},
		"finish_reason": "stop"
	}],
	"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}`

// newModelWithFakeTransport creates a cfModel backed by the given fakeTransport.
func newModelWithFakeTransport(t *testing.T, ft *fakeTransport, cfg *Config) model.LLM {
	t.Helper()
	cfg.HTTPClient = &http.Client{Transport: ft}
	m, err := NewModel(context.Background(), "test-model", cfg)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	return m
}

func TestGenerateContentYieldsOnce(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
		},
	}

	for _, stream := range []bool{false, true} {
		stream := stream
		name := "stream=false"
		if stream {
			name = "stream=true"
		}
		t.Run(name, func(t *testing.T) {
			ft := &fakeTransport{
				responses: []*http.Response{cannedJSONResponse(minimalCompletion)},
			}
			m := newModelWithFakeTransport(t, ft, &Config{AccountID: "x", APIToken: "y"})

			type result struct {
				resp *model.LLMResponse
				err  error
			}
			var results []result
			for resp, err := range m.GenerateContent(context.Background(), req, stream) {
				results = append(results, result{resp, err})
			}

			if len(results) != 1 {
				t.Fatalf("expected exactly 1 yield, got %d", len(results))
			}
			if results[0].err != nil {
				t.Fatalf("unexpected error from GenerateContent: %v", results[0].err)
			}
			if results[0].resp == nil {
				t.Fatal("expected non-nil response")
			}
			if results[0].resp.Content == nil {
				t.Fatal("expected non-nil response content")
			}
			if len(results[0].resp.Content.Parts) == 0 {
				t.Fatal("expected at least one part in response content")
			}
			if results[0].resp.Content.Parts[0].Text != "done" {
				t.Fatalf("expected text 'done', got %q", results[0].resp.Content.Parts[0].Text)
			}
		})
	}
}

func TestMultiTurnToolLoopPreservesIDs(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	// Turn 3 of a conversation: user asked to edit, assistant issued two tool calls,
	// now we're providing both tool responses.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: "edit foo.docx"}},
			},
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "ok let me look"},
					{FunctionCall: &genai.FunctionCall{ID: "call_a", Name: "read_docx"}},
					{FunctionCall: &genai.FunctionCall{ID: "call_b", Name: "browse_docx"}},
				},
			},
			{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_a",
						Name:     "read_docx",
						Response: map[string]any{"content": "file content here"},
					}},
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_b",
						Name:     "browse_docx",
						Response: map[string]any{"result": "browse result"},
					}},
				},
			},
		},
	}

	ft := &fakeTransport{
		responses: []*http.Response{cannedJSONResponse(minimalCompletion)},
	}
	m := newModelWithFakeTransport(t, ft, &Config{AccountID: "x", APIToken: "y"})

	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if len(ft.bodies) != 1 {
		t.Fatalf("expected 1 captured request body, got %d", len(ft.bodies))
	}

	// Parse the captured request body as JSON and examine messages[].
	var body struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(ft.bodies[0], &body); err != nil {
		t.Fatalf("parsing captured request body: %v\nbody: %s", err, ft.bodies[0])
	}

	if len(body.Messages) != 4 {
		t.Fatalf("expected exactly 4 wire messages, got %d\nbody: %s", len(body.Messages), ft.bodies[0])
	}

	// Message 0: user
	var msg0 struct {
		Role string `json:"role"`
	}
	mustUnmarshal(t, body.Messages[0], &msg0)
	if msg0.Role != "user" {
		t.Errorf("messages[0].role: want %q, got %q", "user", msg0.Role)
	}

	// Message 1: assistant with tool_calls[0].id=="call_a" and tool_calls[1].id=="call_b"
	var msg1 struct {
		Role      string `json:"role"`
		ToolCalls []struct {
			ID string `json:"id"`
		} `json:"tool_calls"`
	}
	mustUnmarshal(t, body.Messages[1], &msg1)
	if msg1.Role != "assistant" {
		t.Errorf("messages[1].role: want %q, got %q", "assistant", msg1.Role)
	}
	if len(msg1.ToolCalls) != 2 {
		t.Fatalf("messages[1].tool_calls: want 2, got %d", len(msg1.ToolCalls))
	}
	if msg1.ToolCalls[0].ID != "call_a" {
		t.Errorf("messages[1].tool_calls[0].id: want %q, got %q", "call_a", msg1.ToolCalls[0].ID)
	}
	if msg1.ToolCalls[1].ID != "call_b" {
		t.Errorf("messages[1].tool_calls[1].id: want %q, got %q", "call_b", msg1.ToolCalls[1].ID)
	}

	// Message 2: tool with tool_call_id=="call_a"
	var msg2 struct {
		Role       string `json:"role"`
		ToolCallID string `json:"tool_call_id"`
	}
	mustUnmarshal(t, body.Messages[2], &msg2)
	if msg2.Role != "tool" {
		t.Errorf("messages[2].role: want %q, got %q", "tool", msg2.Role)
	}
	if msg2.ToolCallID != "call_a" {
		t.Errorf("messages[2].tool_call_id: want %q, got %q", "call_a", msg2.ToolCallID)
	}

	// Message 3: tool with tool_call_id=="call_b"
	var msg3 struct {
		Role       string `json:"role"`
		ToolCallID string `json:"tool_call_id"`
	}
	mustUnmarshal(t, body.Messages[3], &msg3)
	if msg3.Role != "tool" {
		t.Errorf("messages[3].role: want %q, got %q", "tool", msg3.Role)
	}
	if msg3.ToolCallID != "call_b" {
		t.Errorf("messages[3].tool_call_id: want %q, got %q", "call_b", msg3.ToolCallID)
	}
}

func mustUnmarshal(t *testing.T, data json.RawMessage, v any) {
	t.Helper()
	if err := json.Unmarshal(data, v); err != nil {
		t.Fatalf("unmarshal failed: %v\ndata: %s", err, data)
	}
}

func TestErrorWrappingIncludesPhase(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	// CF-shaped 401 error response.
	errBody := `{"error":{"code":"","message":"unauthorized","param":"","type":"auth_error"}}`
	ft := &fakeTransport{
		responses: []*http.Response{
			{
				StatusCode: 401,
				Body:       io.NopCloser(strings.NewReader(errBody)),
				Header:     http.Header{"Content-Type": []string{"application/json"}},
			},
		},
	}
	m := newModelWithFakeTransport(t, ft, &Config{AccountID: "x", APIToken: "y"})

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
		},
	}

	var gotErr error
	for _, err := range m.GenerateContent(context.Background(), req, false) {
		gotErr = err
	}

	if gotErr == nil {
		t.Fatal("expected an error from 401 response, got nil")
	}

	var apiErr *openai.Error
	if !errors.As(gotErr, &apiErr) {
		t.Fatalf("expected errors.As(err, *openai.Error) to be true; err=%v", gotErr)
	}
	if apiErr.StatusCode != 401 {
		t.Errorf("expected apiErr.StatusCode==401, got %d", apiErr.StatusCode)
	}
	if !strings.Contains(gotErr.Error(), "calling model:") {
		t.Errorf("expected error to contain %q, got: %v", "calling model:", gotErr)
	}
}

func TestRequestConversionErrorShortCircuits(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	ft := &fakeTransport{}
	m := newModelWithFakeTransport(t, ft, &Config{AccountID: "x", APIToken: "y"})

	// A FunctionResponse with empty ID will fail in BuildRequest.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "", // empty — should cause conversion error
						Name:     "some_tool",
						Response: map[string]any{"result": "ok"},
					}},
				},
			},
		},
	}

	var gotErr error
	for _, err := range m.GenerateContent(context.Background(), req, false) {
		gotErr = err
	}

	if gotErr == nil {
		t.Fatal("expected conversion error, got nil")
	}
	if len(ft.requests) != 0 {
		t.Errorf("expected 0 HTTP requests (short-circuit), got %d", len(ft.requests))
	}
	if !strings.Contains(gotErr.Error(), "converting request:") {
		t.Errorf("expected error to contain %q, got: %v", "converting request:", gotErr)
	}
	if !strings.Contains(gotErr.Error(), "tool response missing call ID") {
		t.Errorf("expected error to contain %q, got: %v", "tool response missing call ID", gotErr)
	}
}

func TestDefaultMaxTokensUsedWhenUnset(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	ft := &fakeTransport{
		responses: []*http.Response{cannedJSONResponse(minimalCompletion)},
	}
	// Config with no MaxTokens set → default of 16384 should be used.
	m := newModelWithFakeTransport(t, ft, &Config{AccountID: "x", APIToken: "y"})

	req := &model.LLMRequest{
		// No Config.MaxOutputTokens set.
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
		},
	}

	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if len(ft.bodies) != 1 {
		t.Fatalf("expected 1 captured request body, got %d", len(ft.bodies))
	}

	var body struct {
		MaxTokens int64 `json:"max_tokens"`
	}
	if err := json.Unmarshal(ft.bodies[0], &body); err != nil {
		t.Fatalf("parsing captured request body: %v", err)
	}
	if body.MaxTokens != 16384 {
		t.Errorf("expected max_tokens=16384 in wire body, got %d", body.MaxTokens)
	}
}

func TestRequestMaxOutputTokensOverridesDefault(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	ft := &fakeTransport{
		responses: []*http.Response{cannedJSONResponse(minimalCompletion)},
	}
	m := newModelWithFakeTransport(t, ft, &Config{AccountID: "x", APIToken: "y"})

	temp := float32(0.7)
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 4096,
			Temperature:     &temp,
		},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
		},
	}

	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if len(ft.bodies) != 1 {
		t.Fatalf("expected 1 captured request body, got %d", len(ft.bodies))
	}

	var body struct {
		MaxTokens int64 `json:"max_tokens"`
	}
	if err := json.Unmarshal(ft.bodies[0], &body); err != nil {
		t.Fatalf("parsing captured request body: %v", err)
	}
	if body.MaxTokens != 4096 {
		t.Errorf("expected max_tokens=4096 in wire body (per-request override), got %d", body.MaxTokens)
	}
}

func TestNewModelCustomBaseURL(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_TOKEN", "")

	ft := &fakeTransport{
		responses: []*http.Response{cannedJSONResponse(minimalCompletion)},
	}
	m := newModelWithFakeTransport(t, ft, &Config{
		AccountID: "x",
		APIToken:  "y",
		BaseURL:   "https://gateway.example/foo/",
	})

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
		},
	}

	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if len(ft.requests) != 1 {
		t.Fatalf("expected 1 captured request, got %d", len(ft.requests))
	}

	rawURL := ft.requests[0].URL.String()
	if !strings.HasPrefix(rawURL, "https://gateway.example/foo/") {
		t.Errorf("expected URL to start with %q, got %q", "https://gateway.example/foo/", rawURL)
	}
	if !strings.HasSuffix(rawURL, "chat/completions") {
		t.Errorf("expected URL to end with %q, got %q", "chat/completions", rawURL)
	}
}
