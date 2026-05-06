package converters

import (
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go/packages/param"
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
