// Copyright 2026 Alcova AI

package converters

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// CompletionToLLMResponse converts a Cloudflare/OpenAI ChatCompletion to a model.LLMResponse.
//
// Portions adapted from github.com/volcengine/veadk-go (Apache 2.0).
func CompletionToLLMResponse(c *openai.ChatCompletion) (*model.LLMResponse, error) {
	if len(c.Choices) == 0 {
		return nil, errors.New("no choices in response")
	}
	choice := c.Choices[0]
	msg := choice.Message

	parts, err := messageToParts(msg)
	if err != nil {
		return nil, err
	}

	resp := &model.LLMResponse{
		Content:      &genai.Content{Role: "model", Parts: parts},
		FinishReason: mapFinishReason(choice.FinishReason),
	}

	// openai-go's Usage is a value type (not a pointer); treat all-zero as
	// "usage not reported by the provider" and leave UsageMetadata nil.
	if c.Usage.PromptTokens != 0 || c.Usage.CompletionTokens != 0 || c.Usage.TotalTokens != 0 {
		resp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(c.Usage.PromptTokens),
			CandidatesTokenCount: int32(c.Usage.CompletionTokens),
			TotalTokenCount:      int32(c.Usage.TotalTokens),
		}
	}

	return resp, nil
}

func messageToParts(msg openai.ChatCompletionMessage) ([]*genai.Part, error) {
	var parts []*genai.Part
	if msg.Content != "" {
		parts = append(parts, &genai.Part{Text: msg.Content})
	}
	for _, tc := range msg.ToolCalls {
		args := map[string]any{}
		if tc.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				return nil, fmt.Errorf("parsing tool call args for %q (id=%s): %w", tc.Function.Name, tc.ID, err)
			}
		}
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: args,
			},
		})
	}
	return parts, nil
}

func mapFinishReason(s string) genai.FinishReason {
	switch s {
	case "stop", "tool_calls":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonOther
	}
}
