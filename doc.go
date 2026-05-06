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
package adkcloudflare
