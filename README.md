# adk-cloudflare-go

Cloudflare Workers AI model provider for the ADK (AI Development Kit).

## Installation

```bash
go get github.com/Alcova-AI/adk-cloudflare-go
```

## Features

- Non-streaming chat completions via Cloudflare Workers AI's OpenAI-compatible endpoint
- Tool calling support
- Structured output support
- Request-side reasoning (ThinkingConfig)

## Usage

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	adkcloudflare "github.com/Alcova-AI/adk-cloudflare-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	m, err := adkcloudflare.NewModel(ctx, "@cf/moonshotai/kimi-k2.6", &adkcloudflare.Config{
		AccountID: os.Getenv("CLOUDFLARE_ACCOUNT_ID"),
		APIToken:  os.Getenv("CLOUDFLARE_API_TOKEN"),
	})
	if err != nil {
		log.Fatal(err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", "user"),
		},
	}
	for resp, err := range m.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Fatal(err)
		}
		if len(resp.Content.Parts) > 0 {
			fmt.Println(resp.Content.Parts[0].Text)
		}
	}
}
```
