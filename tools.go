//go:build tools

// Copyright 2026 Alcova AI

// This file exists solely to anchor module dependencies in go.mod
// before any source file imports them. The build tag excludes it
// from normal builds.

package adkcloudflare

import (
	_ "github.com/openai/openai-go"
	_ "google.golang.org/adk/model"
	_ "google.golang.org/genai"
)
