// Copyright 2026 Alcova AI

package converters

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// BuildRequest converts an ADK LLMRequest to an openai-go ChatCompletionNewParams.
//
// It does NOT apply Config.MaxTokens fallback — that is per-model-instance state
// applied at the call site (Task 6).
//
// Portions of contentToMessages adapted from github.com/volcengine/veadk-go (Apache 2.0).
func BuildRequest(modelName string, req *model.LLMRequest) (openai.ChatCompletionNewParams, error) {
	var msgs []openai.ChatCompletionMessageParamUnion

	if req.Config != nil && req.Config.SystemInstruction != nil {
		msgs = append(msgs, systemMessage(req.Config.SystemInstruction))
	}

	for _, content := range req.Contents {
		if content == nil {
			continue
		}
		contentMsgs, err := contentToMessages(content)
		if err != nil {
			return openai.ChatCompletionNewParams{}, err
		}
		msgs = append(msgs, contentMsgs...)
	}

	if len(msgs) == 0 {
		return openai.ChatCompletionNewParams{}, errors.New("at least one message required")
	}

	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(modelName),
		Messages: msgs,
	}

	if req.Config != nil {
		applyGenerationParams(&params, req.Config)
		if len(req.Config.Tools) > 0 {
			params.Tools = FunctionDeclarationsToTools(req.Config.Tools)
		}
		if req.Config.ToolConfig != nil {
			params.ToolChoice = ToolConfigToToolChoice(req.Config.ToolConfig)
		}
		if req.Config.ResponseSchema != nil {
			params.ResponseFormat = buildResponseFormat(req.Config.ResponseSchema)
		}
		if effort, ok := thinkingBudgetToEffort(req.Config.ThinkingConfig); ok {
			params.ReasoningEffort = effort
		}
	}

	return params, nil
}

// contentToMessages decomposes a single genai.Content into one or more
// OpenAI messages, applying the role-and-parts rules described in the spec.
//
// Portions adapted from github.com/volcengine/veadk-go (Apache 2.0).
func contentToMessages(c *genai.Content) ([]openai.ChatCompletionMessageParamUnion, error) {
	role := c.Role
	if role == "" {
		role = "user"
	}

	// 1. FunctionResponse fanout: if any part is a FunctionResponse, emit one
	// role:"tool" message per FunctionResponse (in part order).
	hasFunctionResponse := false
	for _, p := range c.Parts {
		if p != nil && p.FunctionResponse != nil {
			hasFunctionResponse = true
			break
		}
	}
	if hasFunctionResponse {
		var msgs []openai.ChatCompletionMessageParamUnion
		for _, p := range c.Parts {
			if p == nil || p.FunctionResponse == nil {
				continue
			}
			msg, err := functionResponseToToolMessage(p.FunctionResponse)
			if err != nil {
				return nil, err
			}
			msgs = append(msgs, msg)
		}
		return msgs, nil
	}

	// 2. Assistant turn: role "model" or "assistant"
	if role == "model" || role == "assistant" {
		var asst openai.ChatCompletionAssistantMessageParam

		textContent := concatenateText(c.Parts)
		if textContent != "" {
			asst.Content.OfString = param.NewOpt(textContent)
		}

		for _, p := range c.Parts {
			if p == nil || p.FunctionCall == nil {
				continue
			}
			tc, err := functionCallToOpenAI(p.FunctionCall)
			if err != nil {
				return nil, err
			}
			asst.ToolCalls = append(asst.ToolCalls, tc)
		}

		return []openai.ChatCompletionMessageParamUnion{{OfAssistant: &asst}}, nil
	}

	// 3. User turn: concatenate text; drop InlineData, FileData, Thought parts.
	for _, p := range c.Parts {
		if p == nil {
			continue
		}
		if p.InlineData != nil {
			slog.Default().Debug("dropping unsupported part", "type", "InlineData")
			continue
		}
		if p.FileData != nil {
			slog.Default().Debug("dropping unsupported part", "type", "FileData")
			continue
		}
		if p.Thought {
			slog.Default().Debug("dropping unsupported part", "type", "Thought")
			continue
		}
	}

	textContent := concatenateTextNonThought(c.Parts)
	var user openai.ChatCompletionUserMessageParam
	user.Content.OfString = param.NewOpt(textContent)

	return []openai.ChatCompletionMessageParamUnion{{OfUser: &user}}, nil
}

// concatenateText joins all non-nil text parts with "\n", skipping empty strings.
func concatenateText(parts []*genai.Part) string {
	var texts []string
	for _, p := range parts {
		if p == nil || p.Text == "" {
			continue
		}
		texts = append(texts, p.Text)
	}
	return strings.Join(texts, "\n")
}

// concatenateTextNonThought joins text parts that are NOT thought parts.
func concatenateTextNonThought(parts []*genai.Part) string {
	var texts []string
	for _, p := range parts {
		if p == nil || p.Text == "" || p.Thought {
			continue
		}
		texts = append(texts, p.Text)
	}
	return strings.Join(texts, "\n")
}

// functionCallToOpenAI converts a genai FunctionCall to an openai tool call param.
func functionCallToOpenAI(fc *genai.FunctionCall) (openai.ChatCompletionMessageToolCallParam, error) {
	argsBytes, err := json.Marshal(fc.Args)
	if err != nil {
		return openai.ChatCompletionMessageToolCallParam{}, fmt.Errorf("marshaling tool call args: %w", err)
	}
	return openai.ChatCompletionMessageToolCallParam{
		ID: fc.ID,
		Function: openai.ChatCompletionMessageToolCallFunctionParam{
			Name:      fc.Name,
			Arguments: string(argsBytes),
		},
	}, nil
}

// functionResponseToToolMessage converts a genai FunctionResponse to a role:"tool" message.
func functionResponseToToolMessage(fr *genai.FunctionResponse) (openai.ChatCompletionMessageParamUnion, error) {
	if fr.ID == "" {
		return openai.ChatCompletionMessageParamUnion{}, errors.New("tool response missing call ID")
	}
	contentBytes, err := json.Marshal(fr.Response)
	if err != nil {
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("marshaling tool result: %w", err)
	}
	var tool openai.ChatCompletionToolMessageParam
	tool.ToolCallID = fr.ID
	tool.Content.OfString = param.NewOpt(string(contentBytes))
	return openai.ChatCompletionMessageParamUnion{OfTool: &tool}, nil
}

// systemMessage extracts the text from a SystemInstruction Content and wraps it as role:"system".
func systemMessage(c *genai.Content) openai.ChatCompletionMessageParamUnion {
	text := concatenateText(c.Parts)
	var sys openai.ChatCompletionSystemMessageParam
	sys.Content.OfString = param.NewOpt(text)
	return openai.ChatCompletionMessageParamUnion{OfSystem: &sys}
}

// applyGenerationParams sets temperature/top_p/stop/max_tokens on params
// only when the source fields are explicitly non-nil/non-zero.
// TopK is silently dropped (CF/OpenAI compat doesn't accept it).
func applyGenerationParams(p *openai.ChatCompletionNewParams, cfg *genai.GenerateContentConfig) {
	if cfg.Temperature != nil {
		p.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		p.TopP = param.NewOpt(float64(*cfg.TopP))
	}
	if len(cfg.StopSequences) > 0 {
		p.Stop.OfStringArray = cfg.StopSequences
	}
	if cfg.MaxOutputTokens != 0 {
		p.MaxTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	// TopK is intentionally not mapped — not supported by CF/OpenAI-compatible endpoints.
}

// buildResponseFormat assembles {type:"json_schema", json_schema:{name:"response", schema, strict:false}}
func buildResponseFormat(s *genai.Schema) openai.ChatCompletionNewParamsResponseFormatUnion {
	schemaMap := SchemaToMap(s)
	rf := &shared.ResponseFormatJSONSchemaParam{
		JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:   "response",
			Schema: schemaMap,
			// Strict intentionally omitted (not set to true) for maximum schema compatibility.
		},
	}
	return openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONSchema: rf}
}

// thinkingBudgetToEffort returns the reasoning_effort enum value and an ok flag.
// ok=false means do not set ReasoningEffort on the request.
func thinkingBudgetToEffort(tc *genai.ThinkingConfig) (shared.ReasoningEffort, bool) {
	if tc == nil {
		return "", false
	}
	b := tc.ThinkingBudget
	if b == nil {
		return shared.ReasoningEffortMedium, true
	}
	switch {
	case *b == 0:
		return shared.ReasoningEffortLow, true
	case *b < 4096:
		return shared.ReasoningEffortLow, true
	case *b < 16384:
		return shared.ReasoningEffortMedium, true
	default:
		return shared.ReasoningEffortHigh, true
	}
}
