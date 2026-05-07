// Copyright 2026 Alcova AI

// Package adkcloudflare implements the model.LLM interface from
// google.golang.org/adk/model for Cloudflare Workers AI's
// OpenAI-compatible chat completions endpoint.
//
// Basic usage:
//
//	m, err := adkcloudflare.NewModel(ctx, "@cf/moonshotai/kimi-k2.6", &adkcloudflare.Config{
//	    AccountID: os.Getenv("CLOUDFLARE_ACCOUNT_ID"),
//	    APIToken:  os.Getenv("CLOUDFLARE_API_TOKEN"),
//	})
//	if err != nil { log.Fatal(err) }
//
// Reasoning support is request-side only in v0.1. Setting ThinkingConfig
// on the request influences model behaviour (via reasoning_effort) on
// models that support it, but reasoning content returned by the model
// is NOT decoded back into genai.Part{Thought: true}. If a future caller
// needs reasoning output surfaced, the response-side decoder will be
// added with knowledge of the specific model's wire shape.
//
// response_format is suppressed when the request declares tools. Some
// Workers AI models (notably kimi-k2.6) collapse tool-calling into the
// schema's text fields when both are sent — emitting tool-call XML
// inside the message string instead of structured tool_calls. ADK's
// agenttool.ValidateOutputSchema still enforces the schema after the
// run returns, so omitting the wire-level constraint here only relaxes
// the per-turn JSON guarantee, not the eventual validation.
package adkcloudflare
