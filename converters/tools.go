package converters

import (
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"google.golang.org/genai"
)

// FunctionDeclarationsToTools converts genai Tools to openai-go tool params.
// Iterates each Tool's FunctionDeclarations; tools without declarations are skipped.
func FunctionDeclarationsToTools(tools []*genai.Tool) []openai.ChatCompletionToolParam {
	var result []openai.ChatCompletionToolParam
	for _, tool := range tools {
		if tool == nil || tool.FunctionDeclarations == nil {
			continue
		}
		for _, decl := range tool.FunctionDeclarations {
			if decl == nil {
				continue
			}
			properties, required := extractFunctionParams(decl)
			paramsSchema := shared.FunctionParameters{"type": "object"}
			if len(properties) > 0 {
				paramsSchema["properties"] = properties
			}
			if len(required) > 0 {
				paramsSchema["required"] = required
			}
			funcDef := shared.FunctionDefinitionParam{
				Name:        decl.Name,
				Description: openai.String(decl.Description),
				Parameters:  paramsSchema,
			}
			toolParam := openai.ChatCompletionToolParam{
				Function: funcDef,
			}
			result = append(result, toolParam)
		}
	}
	return result
}

// ToolConfigToToolChoice maps genai's FunctionCallingConfig.Mode + AllowedFunctionNames
// to OpenAI's tool_choice.
func ToolConfigToToolChoice(cfg *genai.ToolConfig) openai.ChatCompletionToolChoiceOptionUnionParam {
	// Default to AUTO
	if cfg == nil || cfg.FunctionCallingConfig == nil {
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("auto"),
		}
	}

	fcc := cfg.FunctionCallingConfig
	switch fcc.Mode {
	case genai.FunctionCallingConfigModeAuto, genai.FunctionCallingConfigModeUnspecified:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("auto"),
		}
	case genai.FunctionCallingConfigModeNone:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("none"),
		}
	case genai.FunctionCallingConfigModeAny:
		// If single allowed function, use named choice
		if len(fcc.AllowedFunctionNames) == 1 {
			return openai.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(
				openai.ChatCompletionNamedToolChoiceFunctionParam{
					Name: fcc.AllowedFunctionNames[0],
				},
			)
		}
		// Otherwise return required
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("required"),
		}
	default:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("auto"),
		}
	}
}

// SchemaToMap converts a genai.Schema to a JSON Schema map[string]any suitable
// for the function `parameters` field or the response_format json_schema.schema field.
func SchemaToMap(s *genai.Schema) map[string]any {
	if s == nil {
		return nil
	}
	result := make(map[string]any)

	// Type field
	if s.Type != "" {
		result["type"] = strings.ToLower(string(s.Type))
	}

	// Description field
	if s.Description != "" {
		result["description"] = s.Description
	}

	// Format field
	if s.Format != "" {
		result["format"] = s.Format
	}

	// Enum field
	if len(s.Enum) > 0 {
		enumAny := make([]any, len(s.Enum))
		for i, e := range s.Enum {
			enumAny[i] = e
		}
		result["enum"] = enumAny
	}

	// Properties field (recursive)
	if len(s.Properties) > 0 {
		props := make(map[string]any)
		for name, prop := range s.Properties {
			props[name] = SchemaToMap(prop)
		}
		result["properties"] = props
	}

	// Required field
	if len(s.Required) > 0 {
		result["required"] = s.Required
	}

	// Items field (recursive)
	if s.Items != nil {
		result["items"] = SchemaToMap(s.Items)
	}

	// Nullable field
	if s.Nullable != nil && *s.Nullable {
		result["nullable"] = true
	}

	// Additional fields that may be present
	if s.Title != "" {
		result["title"] = s.Title
	}

	if s.Default != nil {
		result["default"] = s.Default
	}

	if s.Example != nil {
		result["example"] = s.Example
	}

	if s.MinLength != nil {
		result["minLength"] = *s.MinLength
	}

	if s.MaxLength != nil {
		result["maxLength"] = *s.MaxLength
	}

	if s.Minimum != nil {
		result["minimum"] = *s.Minimum
	}

	if s.Maximum != nil {
		result["maximum"] = *s.Maximum
	}

	if s.MinItems != nil {
		result["minItems"] = *s.MinItems
	}

	if s.MaxItems != nil {
		result["maxItems"] = *s.MaxItems
	}

	if s.Pattern != "" {
		result["pattern"] = s.Pattern
	}

	return result
}

// extractFunctionParams returns (properties, required) extracted from a FunctionDeclaration.
// Parameters takes precedence over ParametersJsonSchema.
func extractFunctionParams(fd *genai.FunctionDeclaration) (map[string]any, []string) {
	properties := make(map[string]any)
	var required []string

	if fd.Parameters != nil {
		// Walk fd.Parameters.Properties and convert each genai.Schema → map via SchemaToMap.
		for name, prop := range fd.Parameters.Properties {
			properties[name] = SchemaToMap(prop)
		}
		required = fd.Parameters.Required
		return properties, required
	}

	switch schema := fd.ParametersJsonSchema.(type) {
	case map[string]any:
		if props, ok := schema["properties"].(map[string]any); ok {
			for k, v := range props {
				properties[k] = v
			}
		}
		// "required" can arrive as []string (manual construction) or []any (JSON).
		switch req := schema["required"].(type) {
		case []string:
			required = req
		case []any:
			required = make([]string, 0, len(req))
			for _, r := range req {
				if s, ok := r.(string); ok {
					required = append(required, s)
				}
			}
		}
	case *jsonschema.Schema:
		// Convert via jsonschema's own walker to map[string]any.
		if schema.Properties != nil {
			for name, prop := range schema.Properties {
				properties[name] = jsonSchemaPropToMap(prop)
			}
		}
		required = schema.Required
	}

	return properties, required
}

// jsonSchemaPropToMap converts a *jsonschema.Schema to map[string]any,
// copying Type, Description, Properties, Items, Required, Enum fields.
func jsonSchemaPropToMap(s *jsonschema.Schema) map[string]any {
	if s == nil {
		return nil
	}
	result := make(map[string]any)

	// Type field
	// TODO: handle jsonschema.Schema.Types union when needed
	if s.Type != "" {
		result["type"] = s.Type
	}

	// Description field
	if s.Description != "" {
		result["description"] = s.Description
	}

	// Properties field (recursive)
	if len(s.Properties) > 0 {
		props := make(map[string]any)
		for name, prop := range s.Properties {
			props[name] = jsonSchemaPropToMap(prop)
		}
		result["properties"] = props
	}

	// Items field (recursive)
	if s.Items != nil {
		result["items"] = jsonSchemaPropToMap(s.Items)
	}

	// Required field
	if len(s.Required) > 0 {
		result["required"] = s.Required
	}

	// Enum field
	if len(s.Enum) > 0 {
		enumAny := make([]any, len(s.Enum))
		for i, e := range s.Enum {
			enumAny[i] = e
		}
		result["enum"] = enumAny
	}

	return result
}
