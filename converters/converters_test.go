package converters

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// Test 1: tool_choice_modes - table-driven test
func TestToolConfigToToolChoice_Modes(t *testing.T) {
	tests := []struct {
		name            string
		cfg             *genai.ToolConfig
		wantAuto        bool
		wantRequiredStr bool
		wantNoneStr     bool
		wantNamed       bool
		wantName        string
	}{
		{
			name: "AUTO mode",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAuto,
				},
			},
			wantAuto: true,
		},
		{
			name:     "nil config defaults to auto",
			cfg:      nil,
			wantAuto: true,
		},
		{
			name: "ANY mode",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			},
			wantRequiredStr: true,
		},
		{
			name: "NONE mode",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeNone,
				},
			},
			wantNoneStr: true,
		},
		{
			name: "ANY mode with single allowed function",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"myFunc"},
				},
			},
			wantNamed: true,
			wantName:  "myFunc",
		},
		{
			name: "ANY mode with multiple allowed functions",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"func1", "func2"},
				},
			},
			wantRequiredStr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ToolConfigToToolChoice(tt.cfg)
			switch {
			case tt.wantAuto:
				if !result.OfAuto.Valid() || result.OfAuto.Value != "auto" {
					t.Errorf("expected OfAuto=\"auto\", got %v (valid=%v)", result.OfAuto.Value, result.OfAuto.Valid())
				}
			case tt.wantRequiredStr:
				if !result.OfAuto.Valid() || result.OfAuto.Value != "required" {
					t.Errorf("expected OfAuto=\"required\", got %v (valid=%v)", result.OfAuto.Value, result.OfAuto.Valid())
				}
			case tt.wantNoneStr:
				if !result.OfAuto.Valid() || result.OfAuto.Value != "none" {
					t.Errorf("expected OfAuto=\"none\", got %v (valid=%v)", result.OfAuto.Value, result.OfAuto.Valid())
				}
			case tt.wantNamed:
				if result.OfChatCompletionNamedToolChoice == nil {
					t.Fatal("expected OfChatCompletionNamedToolChoice to be non-nil")
				}
				if param.IsOmitted(result.OfAuto) {
					// good — no string variant set
				}
				got := result.OfChatCompletionNamedToolChoice.Function.Name
				if got != tt.wantName {
					t.Errorf("expected function name %q, got %q", tt.wantName, got)
				}
			}
		})
	}
}

// Test 2: function_declaration_to_tool - Parameters based
func TestFunctionDeclarationsToTools_WithParameters(t *testing.T) {
	decl := &genai.FunctionDeclaration{
		Name:        "testFunc",
		Description: "A test function",
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"name": {Type: genai.TypeString},
				"age":  {Type: genai.TypeInteger},
			},
			Required: []string{"name"},
		},
	}

	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{decl},
		},
	}

	result := FunctionDeclarationsToTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	fn := result[0].Function
	if fn.Name != "testFunc" {
		t.Errorf("expected function name %q, got %q", "testFunc", fn.Name)
	}
	if fn.Description.Value != "A test function" {
		t.Errorf("expected description %q, got %q", "A test function", fn.Description.Value)
	}
	params := map[string]any(fn.Parameters)
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties to be map[string]any, got %T", params["properties"])
	}
	if _, ok := props["name"]; !ok {
		t.Error("expected properties to contain \"name\"")
	}
	if _, ok := props["age"]; !ok {
		t.Error("expected properties to contain \"age\"")
	}
	req, ok := params["required"].([]string)
	if !ok {
		t.Fatalf("expected required to be []string, got %T", params["required"])
	}
	if len(req) != 1 || req[0] != "name" {
		t.Errorf("expected required=[\"name\"], got %v", req)
	}
}

// Test 3: function_declaration_with_jsonschema_map - ParametersJsonSchema as map
func TestFunctionDeclarationsToTools_WithJsonSchemaMap(t *testing.T) {
	decl := &genai.FunctionDeclaration{
		Name:        "testFunc",
		Description: "A test function",
		ParametersJsonSchema: map[string]any{
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
				"age":  map[string]any{"type": "integer"},
			},
			"required": []any{"name"},
		},
	}

	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{decl},
		},
	}

	result := FunctionDeclarationsToTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	fn := result[0].Function
	if fn.Name != "testFunc" {
		t.Errorf("expected function name %q, got %q", "testFunc", fn.Name)
	}
	params := map[string]any(fn.Parameters)
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties to be map[string]any, got %T", params["properties"])
	}
	if _, ok := props["name"]; !ok {
		t.Error("expected properties to contain \"name\"")
	}
	if _, ok := props["age"]; !ok {
		t.Error("expected properties to contain \"age\"")
	}
	// required arrives as []any from the map (JSON-style)
	req, ok := params["required"].([]string)
	if !ok {
		t.Fatalf("expected required to be []string after extraction, got %T", params["required"])
	}
	if len(req) != 1 || req[0] != "name" {
		t.Errorf("expected required=[\"name\"], got %v", req)
	}
}

// Test 4: function_declaration_with_jsonschema_struct - ParametersJsonSchema as *jsonschema.Schema
func TestFunctionDeclarationsToTools_WithJsonSchemaStruct(t *testing.T) {
	// Build a *jsonschema.Schema with top-level properties, required, and a nested Items
	// to exercise the *jsonschema.Schema branch of extractFunctionParams and
	// jsonSchemaPropToMap recursion.
	schema := &jsonschema.Schema{
		Properties: map[string]*jsonschema.Schema{
			"label": {Type: "string"},
			"tags": {
				Type:  "array",
				Items: &jsonschema.Schema{Type: "string"},
			},
		},
		Required: []string{"label"},
	}

	decl := &genai.FunctionDeclaration{
		Name:                 "testFunc",
		Description:          "A test function",
		ParametersJsonSchema: schema,
	}

	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{decl},
		},
	}

	result := FunctionDeclarationsToTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	fn := result[0].Function
	if fn.Name != "testFunc" {
		t.Errorf("expected function name %q, got %q", "testFunc", fn.Name)
	}
	params := map[string]any(fn.Parameters)

	// properties must contain "label" and "tags"
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties to be map[string]any, got %T", params["properties"])
	}
	labelProp, ok := props["label"].(map[string]any)
	if !ok {
		t.Fatalf("expected label property to be map[string]any, got %T", props["label"])
	}
	if labelProp["type"] != "string" {
		t.Errorf("expected label.type=\"string\", got %v", labelProp["type"])
	}
	tagsProp, ok := props["tags"].(map[string]any)
	if !ok {
		t.Fatalf("expected tags property to be map[string]any, got %T", props["tags"])
	}
	if tagsProp["type"] != "array" {
		t.Errorf("expected tags.type=\"array\", got %v", tagsProp["type"])
	}
	// Verify Items recursion
	itemsProp, ok := tagsProp["items"].(map[string]any)
	if !ok {
		t.Fatalf("expected tags.items to be map[string]any, got %T", tagsProp["items"])
	}
	if itemsProp["type"] != "string" {
		t.Errorf("expected tags.items.type=\"string\", got %v", itemsProp["type"])
	}

	// required must contain "label"
	req, ok := params["required"].([]string)
	if !ok {
		t.Fatalf("expected required to be []string, got %T", params["required"])
	}
	if len(req) != 1 || req[0] != "label" {
		t.Errorf("expected required=[\"label\"], got %v", req)
	}
}

// Test 5: schema_to_map_basic
func TestSchemaToMap_Basic(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString},
			"age":  {Type: genai.TypeInteger},
		},
		Required: []string{"name"},
	}

	result := SchemaToMap(schema)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result["type"] != "object" {
		t.Errorf("expected type=object, got %v", result["type"])
	}
	// required must be []string{"name"}
	req, ok := result["required"].([]string)
	if !ok {
		t.Fatalf("expected required to be []string, got %T", result["required"])
	}
	if len(req) != 1 || req[0] != "name" {
		t.Errorf("expected required=[\"name\"], got %v", req)
	}
	// properties must contain "age" as map[string]any with type=integer
	props, ok := result["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties to be map[string]any, got %T", result["properties"])
	}
	ageProp, ok := props["age"].(map[string]any)
	if !ok {
		t.Fatalf("expected age property to be map[string]any, got %T", props["age"])
	}
	if ageProp["type"] != "integer" {
		t.Errorf("expected age.type=\"integer\", got %v", ageProp["type"])
	}
}

// Test 6: schema_to_map_optional_subobject - doc editor shape
func TestSchemaToMap_OptionalSubobject(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"message": {
				Type: genai.TypeString,
			},
			"file_ref": {
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"uri":       {Type: genai.TypeString},
					"name":      {Type: genai.TypeString},
					"path":      {Type: genai.TypeString},
					"mime_type": {Type: genai.TypeString},
					"size":      {Type: genai.TypeInteger},
					"version":   {Type: genai.TypeString},
				},
			},
		},
		Required: []string{"message"},
	}

	result := SchemaToMap(schema)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result["type"] != "object" {
		t.Errorf("expected type=object, got %v", result["type"])
	}

	// Check that message is in required but file_ref is not.
	// Accept either []string or []any to be resilient to implementation changes.
	required := result["required"]
	var reqNames []string
	switch r := required.(type) {
	case []string:
		reqNames = r
	case []any:
		for _, v := range r {
			if s, ok := v.(string); ok {
				reqNames = append(reqNames, s)
			}
		}
	default:
		t.Fatalf("required field has unexpected type %T", required)
	}
	if !contains(reqNames, "message") {
		t.Error("expected message in required list")
	}
	if contains(reqNames, "file_ref") {
		t.Error("expected file_ref NOT in required list")
	}

	// Check nested properties structure
	if propsMap, ok := result["properties"].(map[string]any); !ok {
		t.Errorf("expected properties to be map[string]any, got %T", result["properties"])
	} else {
		if fileRefMap, ok := propsMap["file_ref"].(map[string]any); !ok {
			t.Errorf("expected file_ref to be map[string]any, got %T", propsMap["file_ref"])
		} else if fileRefMap["type"] != "object" {
			t.Errorf("expected nested file_ref type=object, got %v", fileRefMap["type"])
		}
	}
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// Test 7: schema_to_map_nested_arrays
func TestSchemaToMap_NestedArrays(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"id":   {Type: genai.TypeString},
				"name": {Type: genai.TypeString},
			},
		},
	}

	result := SchemaToMap(schema)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result["type"] != "array" {
		t.Errorf("expected type=array, got %v", result["type"])
	}
	itemsRaw := result["items"]
	if itemsRaw == nil {
		t.Fatal("expected items to be set")
	}
	items, ok := itemsRaw.(map[string]any)
	if !ok {
		t.Fatalf("expected items to be map[string]any, got %T", itemsRaw)
	}
	if items["type"] != "object" {
		t.Errorf("expected items.type=\"object\", got %v", items["type"])
	}
	itemsProps, ok := items["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected items.properties to be map[string]any, got %T", items["properties"])
	}
	if _, ok := itemsProps["id"]; !ok {
		t.Error("expected items.properties to contain \"id\"")
	}
	if _, ok := itemsProps["name"]; !ok {
		t.Error("expected items.properties to contain \"name\"")
	}
}

// Test 8: schema_to_map_nullable
func TestSchemaToMap_Nullable(t *testing.T) {
	nullable := true
	schema := &genai.Schema{
		Type:     genai.TypeString,
		Nullable: &nullable,
	}

	result := SchemaToMap(schema)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result["nullable"] != true {
		t.Errorf("expected nullable=true, got %v", result["nullable"])
	}
}

// Test 9: schema_to_map_enum
func TestSchemaToMap_Enum(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeString,
		Enum: []string{"a", "b", "c"},
	}

	result := SchemaToMap(schema)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	enumVal, ok := result["enum"]
	if !ok {
		t.Error("expected enum key in result")
	} else if enumList, ok := enumVal.([]any); !ok {
		t.Errorf("expected enum to be []any, got %T", enumVal)
	} else if len(enumList) != 3 {
		t.Errorf("expected 3 enum values, got %d", len(enumList))
	}
}

// ============================================================
// Request converter tests (Task 4)
// ============================================================

func ptr32(v int32) *int32 { return &v }

// helper: build a minimal LLMRequest with one user text message
func userTextReq(text string) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: text}}},
		},
	}
}

// Test: system_instruction_becomes_first_message
func TestBuildRequest_SystemInstructionBecomesFirstMessage(t *testing.T) {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "You are a helpful assistant."}},
			},
		},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "Hello"}}},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) < 2 {
		t.Fatalf("expected at least 2 messages, got %d", len(params.Messages))
	}
	first := params.Messages[0]
	if first.OfSystem == nil {
		t.Fatal("expected first message to be system message (OfSystem non-nil)")
	}
	if !first.OfSystem.Content.OfString.Valid() {
		t.Fatal("expected system message content to be a string")
	}
	if first.OfSystem.Content.OfString.Value != "You are a helpful assistant." {
		t.Errorf("expected system content %q, got %q", "You are a helpful assistant.", first.OfSystem.Content.OfString.Value)
	}
}

// Test: user_content_becomes_user_message
func TestBuildRequest_UserContentBecomesUserMessage(t *testing.T) {
	params, err := BuildRequest("test-model", userTextReq("Hello"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	msg := params.Messages[0]
	if msg.OfUser == nil {
		t.Fatal("expected user message (OfUser non-nil)")
	}
	if !msg.OfUser.Content.OfString.Valid() {
		t.Fatal("expected user content to be a string")
	}
	if msg.OfUser.Content.OfString.Value != "Hello" {
		t.Errorf("expected content %q, got %q", "Hello", msg.OfUser.Content.OfString.Value)
	}
}

// Test: model_text_becomes_assistant_message
func TestBuildRequest_ModelTextBecomesAssistantMessage(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "model", Parts: []*genai.Part{{Text: "I can help with that."}}},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	msg := params.Messages[0]
	if msg.OfAssistant == nil {
		t.Fatal("expected assistant message (OfAssistant non-nil)")
	}
	if !msg.OfAssistant.Content.OfString.Valid() {
		t.Fatal("expected assistant content to be a string")
	}
	if msg.OfAssistant.Content.OfString.Value != "I can help with that." {
		t.Errorf("expected content %q, got %q", "I can help with that.", msg.OfAssistant.Content.OfString.Value)
	}
}

// Test: unset_role_defaults_to_user
func TestBuildRequest_UnsetRoleDefaultsToUser(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "", Parts: []*genai.Part{{Text: "No role set"}}},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	msg := params.Messages[0]
	if msg.OfUser == nil {
		t.Fatalf("expected user message (OfUser non-nil) for empty role, got OfAssistant=%v", msg.OfAssistant)
	}
	if msg.OfUser.Content.OfString.Value != "No role set" {
		t.Errorf("expected content %q, got %q", "No role set", msg.OfUser.Content.OfString.Value)
	}
}

// Test: function_call_in_model_content
func TestBuildRequest_FunctionCallInModelContent(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "call-123", Name: "myFunc", Args: map[string]any{"key": "value"}}},
				},
			},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	msg := params.Messages[0]
	if msg.OfAssistant == nil {
		t.Fatal("expected assistant message")
	}
	if len(msg.OfAssistant.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(msg.OfAssistant.ToolCalls))
	}
	tc := msg.OfAssistant.ToolCalls[0]
	if tc.ID != "call-123" {
		t.Errorf("expected tool call ID %q, got %q", "call-123", tc.ID)
	}
	if tc.Function.Name != "myFunc" {
		t.Errorf("expected function name %q, got %q", "myFunc", tc.Function.Name)
	}
	// args should be JSON-stringified
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatalf("expected valid JSON args, got %q: %v", tc.Function.Arguments, err)
	}
	if args["key"] != "value" {
		t.Errorf("expected args[key]=value, got %v", args["key"])
	}
}

// Test: mixed_text_and_tool_calls
func TestBuildRequest_MixedTextAndToolCalls(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "I'll call two functions."},
					{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "funcA", Args: map[string]any{"x": 1.0}}},
					{FunctionCall: &genai.FunctionCall{ID: "call-2", Name: "funcB", Args: map[string]any{"y": 2.0}}},
				},
			},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message (assistant), got %d", len(params.Messages))
	}
	msg := params.Messages[0]
	if msg.OfAssistant == nil {
		t.Fatal("expected assistant message")
	}
	// text content
	if !msg.OfAssistant.Content.OfString.Valid() {
		t.Fatal("expected assistant content to have text")
	}
	if msg.OfAssistant.Content.OfString.Value != "I'll call two functions." {
		t.Errorf("expected text content %q, got %q", "I'll call two functions.", msg.OfAssistant.Content.OfString.Value)
	}
	// two tool calls in order
	if len(msg.OfAssistant.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(msg.OfAssistant.ToolCalls))
	}
	if msg.OfAssistant.ToolCalls[0].ID != "call-1" {
		t.Errorf("expected first tool call ID %q, got %q", "call-1", msg.OfAssistant.ToolCalls[0].ID)
	}
	if msg.OfAssistant.ToolCalls[1].ID != "call-2" {
		t.Errorf("expected second tool call ID %q, got %q", "call-2", msg.OfAssistant.ToolCalls[1].ID)
	}
	if msg.OfAssistant.ToolCalls[0].Function.Name != "funcA" {
		t.Errorf("expected first tool name %q, got %q", "funcA", msg.OfAssistant.ToolCalls[0].Function.Name)
	}
	if msg.OfAssistant.ToolCalls[1].Function.Name != "funcB" {
		t.Errorf("expected second tool name %q, got %q", "funcB", msg.OfAssistant.ToolCalls[1].Function.Name)
	}
}

// Test: function_response_becomes_tool_message
func TestBuildRequest_FunctionResponseBecomesToolMessage(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call-123",
						Name:     "myFunc",
						Response: map[string]any{"result": "ok"},
					}},
				},
			},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	msg := params.Messages[0]
	if msg.OfTool == nil {
		t.Fatal("expected tool message (OfTool non-nil)")
	}
	if msg.OfTool.ToolCallID != "call-123" {
		t.Errorf("expected tool_call_id %q, got %q", "call-123", msg.OfTool.ToolCallID)
	}
	if !msg.OfTool.Content.OfString.Valid() {
		t.Fatal("expected tool content to be a string")
	}
	// content should be JSON-marshaled response
	var content map[string]any
	if err := json.Unmarshal([]byte(msg.OfTool.Content.OfString.Value), &content); err != nil {
		t.Fatalf("expected valid JSON tool content, got %q: %v", msg.OfTool.Content.OfString.Value, err)
	}
	if content["result"] != "ok" {
		t.Errorf("expected content[result]=ok, got %v", content["result"])
	}
}

// Test: parallel_function_responses_fan_out
// Also tests multi-turn: [user(text), assistant(text+2FunctionCalls), user(2FunctionResponses), user(text)]
func TestBuildRequest_ParallelFunctionResponsesFanOut(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			// Turn 1: user text
			{Role: "user", Parts: []*genai.Part{{Text: "Call two functions please."}}},
			// Turn 2: assistant with text + 2 function calls
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "Calling both functions."},
					{FunctionCall: &genai.FunctionCall{ID: "tc-aaa", Name: "funcA", Args: map[string]any{"a": 1.0}}},
					{FunctionCall: &genai.FunctionCall{ID: "tc-bbb", Name: "funcB", Args: map[string]any{"b": 2.0}}},
				},
			},
			// Turn 3: two function responses in one Content → fan out to 2 tool messages
			{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{ID: "tc-aaa", Name: "funcA", Response: map[string]any{"r": "a"}}},
					{FunctionResponse: &genai.FunctionResponse{ID: "tc-bbb", Name: "funcB", Response: map[string]any{"r": "b"}}},
				},
			},
			// Turn 4: user text
			{Role: "user", Parts: []*genai.Part{{Text: "Thanks!"}}},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Expected: user(text), assistant(text+2calls), tool(tc-aaa), tool(tc-bbb), user(text) = 5 messages
	if len(params.Messages) != 5 {
		t.Fatalf("expected 5 messages, got %d", len(params.Messages))
	}
	// [0] user text
	if params.Messages[0].OfUser == nil {
		t.Error("expected message[0] to be user")
	}
	// [1] assistant with 2 tool calls
	if params.Messages[1].OfAssistant == nil {
		t.Error("expected message[1] to be assistant")
	} else if len(params.Messages[1].OfAssistant.ToolCalls) != 2 {
		t.Errorf("expected 2 tool calls in assistant message, got %d", len(params.Messages[1].OfAssistant.ToolCalls))
	}
	// [2] tool message for tc-aaa
	if params.Messages[2].OfTool == nil {
		t.Error("expected message[2] to be tool message")
	} else if params.Messages[2].OfTool.ToolCallID != "tc-aaa" {
		t.Errorf("expected tool_call_id %q, got %q", "tc-aaa", params.Messages[2].OfTool.ToolCallID)
	}
	// [3] tool message for tc-bbb
	if params.Messages[3].OfTool == nil {
		t.Error("expected message[3] to be tool message")
	} else if params.Messages[3].OfTool.ToolCallID != "tc-bbb" {
		t.Errorf("expected tool_call_id %q, got %q", "tc-bbb", params.Messages[3].OfTool.ToolCallID)
	}
	// [4] user text
	if params.Messages[4].OfUser == nil {
		t.Error("expected message[4] to be user")
	} else if params.Messages[4].OfUser.Content.OfString.Value != "Thanks!" {
		t.Errorf("expected final user text %q, got %q", "Thanks!", params.Messages[4].OfUser.Content.OfString.Value)
	}
}

// Test: function_response_empty_id_errors
func TestBuildRequest_FunctionResponseEmptyIDErrors(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "", // empty — should error
						Name:     "myFunc",
						Response: map[string]any{"result": "ok"},
					}},
				},
			},
		},
	}
	_, err := BuildRequest("test-model", req)
	if err == nil {
		t.Fatal("expected error for empty function response ID, got nil")
	}
	if !strings.Contains(err.Error(), "tool response missing call ID") {
		t.Errorf("expected error to contain %q, got %q", "tool response missing call ID", err.Error())
	}
}

// Test: multiple_text_parts_concatenate
func TestBuildRequest_MultipleTextPartsConcatenate(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "a"}, {Text: "b"}}},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	msg := params.Messages[0]
	if msg.OfUser == nil {
		t.Fatal("expected user message")
	}
	expected := "a\nb"
	if msg.OfUser.Content.OfString.Value != expected {
		t.Errorf("expected concatenated text %q, got %q", expected, msg.OfUser.Content.OfString.Value)
	}
}

// Test: unsupported_part_silently_dropped
// InlineData, FileData, Thought parts are dropped with a debug log, no error.
func TestBuildRequest_UnsupportedPartSilentlyDropped(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "keep this"},
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}}},
					{Thought: true, Text: "thinking..."},
				},
			},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	msg := params.Messages[0]
	if msg.OfUser == nil {
		t.Fatal("expected user message")
	}
	// Only the first non-Thought text part is kept. Thought part Text is also dropped.
	// The concatenated text should be "keep this" only.
	if msg.OfUser.Content.OfString.Value != "keep this" {
		t.Errorf("expected content %q, got %q", "keep this", msg.OfUser.Content.OfString.Value)
	}
}

// Test: response_schema_emits_response_format
func TestBuildRequest_ResponseSchemaEmitsResponseFormat(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString},
		},
		Required: []string{"name"},
	}
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			ResponseSchema: schema,
		},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.ResponseFormat.OfJSONSchema == nil {
		t.Fatal("expected response_format.OfJSONSchema to be non-nil")
	}
	rf := params.ResponseFormat.OfJSONSchema
	// type marshals as "json_schema" (zero value via Default() during marshal)
	marshaled, err := rf.MarshalJSON()
	if err != nil {
		t.Fatalf("MarshalJSON failed: %v", err)
	}
	if !strings.Contains(string(marshaled), `"json_schema"`) {
		t.Errorf("expected marshaled response_format to contain json_schema, got: %s", string(marshaled))
	}
	// strict must be false (omitted, not set to true)
	if rf.JSONSchema.Strict.Valid() && rf.JSONSchema.Strict.Value {
		t.Error("expected strict to not be true")
	}
	// schema must be non-nil and match SchemaToMap output
	if rf.JSONSchema.Schema == nil {
		t.Fatal("expected json_schema.schema to be non-nil")
	}
	schemaMap, ok := rf.JSONSchema.Schema.(map[string]any)
	if !ok {
		t.Fatalf("expected schema to be map[string]any, got %T", rf.JSONSchema.Schema)
	}
	if schemaMap["type"] != "object" {
		t.Errorf("expected schema.type=%q, got %v", "object", schemaMap["type"])
	}
}

// Test: response_schema_with_optional_subobject
// Uses the doc editor OutputSchema pattern (object with required "message", optional "file_ref")
func TestBuildRequest_ResponseSchemaWithOptionalSubobject(t *testing.T) {
	docEditorOutputSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"message": {Type: genai.TypeString, Description: "Human-readable summary of the edit."},
			"file_ref": {
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"uri":       {Type: genai.TypeString},
					"name":      {Type: genai.TypeString},
					"path":      {Type: genai.TypeString},
					"mime_type": {Type: genai.TypeString},
					"size":      {Type: genai.TypeInteger},
					"version":   {Type: genai.TypeInteger},
				},
			},
		},
		Required: []string{"message"}, // file_ref is intentionally optional
	}
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			ResponseSchema: docEditorOutputSchema,
		},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "edit the doc"}}},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.ResponseFormat.OfJSONSchema == nil {
		t.Fatal("expected OfJSONSchema to be non-nil")
	}
	schemaAny := params.ResponseFormat.OfJSONSchema.JSONSchema.Schema
	schemaMap, ok := schemaAny.(map[string]any)
	if !ok {
		t.Fatalf("expected schema to be map[string]any, got %T", schemaAny)
	}
	// top-level required = ["message"]
	req2, ok := schemaMap["required"].([]string)
	if !ok {
		t.Fatalf("expected required to be []string, got %T", schemaMap["required"])
	}
	if len(req2) != 1 || req2[0] != "message" {
		t.Errorf("expected required=[\"message\"], got %v", req2)
	}
	// properties.file_ref.type == "object"
	propsMap, ok := schemaMap["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties to be map[string]any, got %T", schemaMap["properties"])
	}
	fileRefMap, ok := propsMap["file_ref"].(map[string]any)
	if !ok {
		t.Fatalf("expected file_ref to be map[string]any, got %T", propsMap["file_ref"])
	}
	if fileRefMap["type"] != "object" {
		t.Errorf("expected file_ref.type=object, got %v", fileRefMap["type"])
	}
}

// Test: thinking_config_budget_thresholds
func TestBuildRequest_ThinkingConfigBudgetThresholds(t *testing.T) {
	tests := []struct {
		name         string
		config       *genai.ThinkingConfig
		wantEffort   shared.ReasoningEffort
		wantOK       bool
	}{
		{name: "nil config", config: nil, wantEffort: "", wantOK: false},
		{name: "nil budget", config: &genai.ThinkingConfig{ThinkingBudget: nil}, wantEffort: shared.ReasoningEffortMedium, wantOK: true},
		{name: "budget=0", config: &genai.ThinkingConfig{ThinkingBudget: ptr32(0)}, wantEffort: shared.ReasoningEffortLow, wantOK: true},
		{name: "budget=4095", config: &genai.ThinkingConfig{ThinkingBudget: ptr32(4095)}, wantEffort: shared.ReasoningEffortLow, wantOK: true},
		{name: "budget=4096", config: &genai.ThinkingConfig{ThinkingBudget: ptr32(4096)}, wantEffort: shared.ReasoningEffortMedium, wantOK: true},
		{name: "budget=16383", config: &genai.ThinkingConfig{ThinkingBudget: ptr32(16383)}, wantEffort: shared.ReasoningEffortMedium, wantOK: true},
		{name: "budget=16384", config: &genai.ThinkingConfig{ThinkingBudget: ptr32(16384)}, wantEffort: shared.ReasoningEffortHigh, wantOK: true},
		{name: "budget=100000", config: &genai.ThinkingConfig{ThinkingBudget: ptr32(100000)}, wantEffort: shared.ReasoningEffortHigh, wantOK: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotEffort, gotOK := thinkingBudgetToEffort(tt.config)
			if gotOK != tt.wantOK {
				t.Errorf("ok: expected %v, got %v", tt.wantOK, gotOK)
			}
			if gotOK && gotEffort != tt.wantEffort {
				t.Errorf("effort: expected %q, got %q", tt.wantEffort, gotEffort)
			}
		})
	}
}

// Test: topk_dropped
func TestBuildRequest_TopKDropped(t *testing.T) {
	topk := float32(40)
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			TopK: &topk,
		},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Marshal to JSON and verify there is no top_k field
	data, err := params.MarshalJSON()
	if err != nil {
		t.Fatalf("MarshalJSON failed: %v", err)
	}
	if strings.Contains(string(data), "top_k") {
		t.Errorf("expected no top_k in marshaled request, got: %s", string(data))
	}
}

// Test: generation_params_mapped_to_request
func TestBuildRequest_GenerationParamsMappedToRequest(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.9)
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Temperature:     &temp,
			TopP:            &topP,
			StopSequences:   []string{"END", "STOP"},
			MaxOutputTokens: 512,
		},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		},
	}
	params, err := BuildRequest("test-model", req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !params.Temperature.Valid() {
		t.Fatal("expected Temperature to be set")
	}
	if params.Temperature.Value != float64(temp) {
		t.Errorf("expected temperature=%v, got %v", float64(temp), params.Temperature.Value)
	}
	if !params.TopP.Valid() {
		t.Fatal("expected TopP to be set")
	}
	if params.TopP.Value != float64(topP) {
		t.Errorf("expected top_p=%v, got %v", float64(topP), params.TopP.Value)
	}
	// Stop sequences
	if param.IsOmitted(params.Stop.OfStringArray) {
		t.Fatal("expected stop sequences to be set as array")
	}
	if len(params.Stop.OfStringArray) != 2 {
		t.Errorf("expected 2 stop sequences, got %d", len(params.Stop.OfStringArray))
	}
	// MaxTokens
	if !params.MaxTokens.Valid() {
		t.Fatal("expected MaxTokens to be set")
	}
	if params.MaxTokens.Value != 512 {
		t.Errorf("expected max_tokens=512, got %d", params.MaxTokens.Value)
	}
}

// Test: empty_contents_and_no_system_errors
func TestBuildRequest_EmptyContentsAndNoSystemErrors(t *testing.T) {
	req := &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{SystemInstruction: nil},
		Contents: nil,
	}
	_, err := BuildRequest("test-model", req)
	if err == nil {
		t.Fatal("expected error for empty contents, got nil")
	}
	if !strings.Contains(err.Error(), "at least one message required") {
		t.Errorf("expected error to contain %q, got %q", "at least one message required", err.Error())
	}
}

// ============================================================
// Response converter tests (Task 5)
// ============================================================

// helper: build a minimal ChatCompletion with one choice
func makeCompletion(content string, toolCalls []openai.ChatCompletionMessageToolCall, finishReason string) openai.ChatCompletion {
	return openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				FinishReason: finishReason,
				Message: openai.ChatCompletionMessage{
					Content:   content,
					ToolCalls: toolCalls,
				},
			},
		},
	}
}

// helper: build a tool call
func makeToolCall(id, name, args string) openai.ChatCompletionMessageToolCall {
	return openai.ChatCompletionMessageToolCall{
		ID: id,
		Function: openai.ChatCompletionMessageToolCallFunction{
			Name:      name,
			Arguments: args,
		},
	}
}

// Test: usage_extracted
func TestCompletionToLLMResponse_UsageExtracted(t *testing.T) {
	c := makeCompletion("hi", nil, "stop")
	c.Usage = openai.CompletionUsage{
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
	}
	resp, err := CompletionToLLMResponse(&c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.UsageMetadata == nil {
		t.Fatal("expected UsageMetadata to be non-nil")
	}
	if resp.UsageMetadata.PromptTokenCount != 10 {
		t.Errorf("expected PromptTokenCount=10, got %d", resp.UsageMetadata.PromptTokenCount)
	}
	if resp.UsageMetadata.CandidatesTokenCount != 20 {
		t.Errorf("expected CandidatesTokenCount=20, got %d", resp.UsageMetadata.CandidatesTokenCount)
	}
	if resp.UsageMetadata.TotalTokenCount != 30 {
		t.Errorf("expected TotalTokenCount=30, got %d", resp.UsageMetadata.TotalTokenCount)
	}
}

// Test: usage_zero_leaves_metadata_nil
func TestCompletionToLLMResponse_UsageZeroLeavesMetadataNil(t *testing.T) {
	c := makeCompletion("hi", nil, "stop")
	// Usage is zero-valued (all token counts are 0)
	resp, err := CompletionToLLMResponse(&c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.UsageMetadata != nil {
		t.Errorf("expected UsageMetadata to be nil for all-zero usage, got %+v", resp.UsageMetadata)
	}
}

// Test: finish_reason_mapping (table-driven, 6 rows)
func TestCompletionToLLMResponse_FinishReasonMapping(t *testing.T) {
	tests := []struct {
		input string
		want  genai.FinishReason
	}{
		{"stop", genai.FinishReasonStop},
		{"tool_calls", genai.FinishReasonStop},
		{"length", genai.FinishReasonMaxTokens},
		{"content_filter", genai.FinishReasonSafety},
		{"weird", genai.FinishReasonOther},
		{"", genai.FinishReasonOther},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			c := makeCompletion("hi", nil, tt.input)
			resp, err := CompletionToLLMResponse(&c)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp.FinishReason != tt.want {
				t.Errorf("finish reason: expected %q, got %q", tt.want, resp.FinishReason)
			}
		})
	}
}

// Test: mixed_response_text_and_tool_calls
func TestCompletionToLLMResponse_MixedResponseTextAndToolCalls(t *testing.T) {
	tc := makeToolCall("call_xyz", "myFunc", `{"a":1}`)
	c := makeCompletion("hi", []openai.ChatCompletionMessageToolCall{tc}, "tool_calls")
	resp, err := CompletionToLLMResponse(&c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content == nil {
		t.Fatal("expected non-nil Content")
	}
	parts := resp.Content.Parts
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts (text+functioncall), got %d", len(parts))
	}
	// First part must be text
	if parts[0].Text != "hi" {
		t.Errorf("expected parts[0].Text=%q, got %q", "hi", parts[0].Text)
	}
	// Second part must be function call
	if parts[1].FunctionCall == nil {
		t.Fatal("expected parts[1] to be a FunctionCall")
	}
	if parts[1].FunctionCall.Name != "myFunc" {
		t.Errorf("expected FunctionCall.Name=%q, got %q", "myFunc", parts[1].FunctionCall.Name)
	}
}

// Test: response_only_tool_calls
func TestCompletionToLLMResponse_ResponseOnlyToolCalls(t *testing.T) {
	tcs := []openai.ChatCompletionMessageToolCall{
		makeToolCall("call_1", "funcA", `{"x":1}`),
		makeToolCall("call_2", "funcB", `{"y":2}`),
	}
	c := makeCompletion("", tcs, "tool_calls")
	resp, err := CompletionToLLMResponse(&c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts := resp.Content.Parts
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}
	for i, p := range parts {
		if p.Text != "" {
			t.Errorf("parts[%d]: expected no text, got %q", i, p.Text)
		}
		if p.FunctionCall == nil {
			t.Errorf("parts[%d]: expected FunctionCall to be non-nil", i)
		}
	}
}

// Test: response_only_text
func TestCompletionToLLMResponse_ResponseOnlyText(t *testing.T) {
	c := makeCompletion("hi", nil, "stop")
	resp, err := CompletionToLLMResponse(&c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts := resp.Content.Parts
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}
	if parts[0].Text != "hi" {
		t.Errorf("expected parts[0].Text=%q, got %q", "hi", parts[0].Text)
	}
	if parts[0].FunctionCall != nil {
		t.Error("expected no FunctionCall in parts[0]")
	}
}

// Test: response_empty
func TestCompletionToLLMResponse_ResponseEmpty(t *testing.T) {
	c := makeCompletion("", nil, "stop")
	resp, err := CompletionToLLMResponse(&c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content == nil {
		t.Fatal("expected non-nil Content")
	}
	if len(resp.Content.Parts) != 0 {
		t.Errorf("expected 0 parts for empty message, got %d", len(resp.Content.Parts))
	}
}

// Test: response_no_choices_errors
func TestCompletionToLLMResponse_ResponseNoChoicesErrors(t *testing.T) {
	c := openai.ChatCompletion{
		Choices: nil,
	}
	_, err := CompletionToLLMResponse(&c)
	if err == nil {
		t.Fatal("expected error for no choices, got nil")
	}
	// Error must be bare — no "converting response:" prefix
	want := "no choices in response"
	if err.Error() != want {
		t.Errorf("expected error %q, got %q", want, err.Error())
	}
}

// Test: tool_call_args_unparseable_errors
func TestCompletionToLLMResponse_ToolCallArgsUnparseableErrors(t *testing.T) {
	// "[1,2,3]" is valid JSON but not an object — unmarshal into map[string]any fails
	tc := makeToolCall("call_bad", "myFunc", "[1,2,3]")
	c := makeCompletion("", []openai.ChatCompletionMessageToolCall{tc}, "tool_calls")
	_, err := CompletionToLLMResponse(&c)
	if err == nil {
		t.Fatal("expected error for unparseable tool call args, got nil")
	}
	msg := err.Error()
	if !strings.Contains(msg, "parsing tool call args") {
		t.Errorf("expected error to contain %q, got %q", "parsing tool call args", msg)
	}
	if !strings.Contains(msg, "myFunc") {
		t.Errorf("expected error to contain function name %q, got %q", "myFunc", msg)
	}
	if !strings.Contains(msg, "call_bad") {
		t.Errorf("expected error to contain tool call id %q, got %q", "call_bad", msg)
	}
}

// Test: tool_call_id_preserved
func TestCompletionToLLMResponse_ToolCallIDPreserved(t *testing.T) {
	const wantID = "call_abc"
	tc := makeToolCall(wantID, "myFunc", `{}`)
	c := makeCompletion("", []openai.ChatCompletionMessageToolCall{tc}, "tool_calls")
	resp, err := CompletionToLLMResponse(&c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Content.Parts) == 0 {
		t.Fatal("expected at least one part")
	}
	fc := resp.Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatal("expected FunctionCall in parts[0]")
	}
	if fc.ID != wantID {
		t.Errorf("tool call ID: expected %q, got %q", wantID, fc.ID)
	}
}
