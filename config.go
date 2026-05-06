package adkcloudflare

import (
	"errors"
	"net/http"
	"os"
)

type Config struct {
	AccountID  string
	APIToken   string
	BaseURL    string
	MaxTokens  int
	HTTPClient *http.Client
}

// resolve returns a fully-populated copy of the config with env-var fallback and defaults applied.
func (c *Config) resolve() (*Config, error) {
	out := &Config{
		AccountID:  c.AccountID,
		APIToken:   c.APIToken,
		BaseURL:    c.BaseURL,
		MaxTokens:  c.MaxTokens,
		HTTPClient: c.HTTPClient,
	}
	if out.AccountID == "" {
		out.AccountID = os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	}
	if out.APIToken == "" {
		out.APIToken = os.Getenv("CLOUDFLARE_API_TOKEN")
	}
	if out.AccountID == "" {
		return nil, errors.New("cloudflare: account ID is required (set Config.AccountID or CLOUDFLARE_ACCOUNT_ID)")
	}
	if out.APIToken == "" {
		return nil, errors.New("cloudflare: API token is required (set Config.APIToken or CLOUDFLARE_API_TOKEN)")
	}
	if out.MaxTokens == 0 {
		out.MaxTokens = 16384
	}
	if out.HTTPClient == nil {
		out.HTTPClient = http.DefaultClient
	}
	return out, nil
}
