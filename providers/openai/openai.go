package openai

import (
	"context"
	"fmt"
	"strings"

	"github.com/lyricat/goutils/structs"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
	"github.com/quailyquaily/uniai/chat"
	"github.com/quailyquaily/uniai/internal/diag"
	"github.com/quailyquaily/uniai/internal/oaicompat"
)

type Config struct {
	APIKey       string
	BaseURL      string
	DefaultModel string
	Debug        bool
}

type Provider struct {
	client       openai.Client
	defaultModel string
	debug        bool
}

func New(cfg Config) (*Provider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("openai api key is required")
	}

	opts := []option.RequestOption{option.WithAPIKey(cfg.APIKey)}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	return &Provider{
		client:       openai.NewClient(opts...),
		defaultModel: cfg.DefaultModel,
		debug:        cfg.Debug,
	}, nil
}

func (p *Provider) Chat(ctx context.Context, req *chat.Request) (*chat.Result, error) {
	debugFn := req.Options.DebugFn
	params, err := buildParams(req, p.defaultModel)
	if err != nil {
		return nil, err
	}
	diag.LogJSON(p.debug, debugFn, "openai.chat.request", params)
	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}
	if raw := resp.RawJSON(); raw != "" {
		diag.LogText(p.debug, debugFn, "openai.chat.response", raw)
	} else {
		diag.LogJSON(p.debug, debugFn, "openai.chat.response", resp)
	}
	return toResult(resp), nil
}

func buildParams(req *chat.Request, defaultModel string) (openai.ChatCompletionNewParams, error) {
	model := req.Model
	if model == "" {
		model = defaultModel
	}
	if model == "" {
		return openai.ChatCompletionNewParams{}, fmt.Errorf("model is required")
	}

	messages, err := oaicompat.ToMessages(req.Messages)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(model),
		Messages: messages,
	}

	if req.Options.Temperature != nil {
		params.Temperature = openai.Float(*req.Options.Temperature)
	}
	if req.Options.TopP != nil {
		params.TopP = openai.Float(*req.Options.TopP)
	}
	if req.Options.MaxTokens != nil {
		maxTokens := int64(*req.Options.MaxTokens)
		if useMaxCompletionTokens(model) {
			params.MaxCompletionTokens = openai.Int(maxTokens)
		} else {
			params.MaxTokens = openai.Int(maxTokens)
		}
	}
	if len(req.Options.Stop) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: append([]string{}, req.Options.Stop...),
		}
	}
	if req.Options.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(*req.Options.PresencePenalty)
	}
	if req.Options.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(*req.Options.FrequencyPenalty)
	}
	if req.Options.User != nil {
		params.User = openai.String(*req.Options.User)
	}

	if len(req.Tools) > 0 {
		tools, err := oaicompat.ToToolParams(req.Tools)
		if err != nil {
			return openai.ChatCompletionNewParams{}, err
		}
		params.Tools = tools
	}

	if req.ToolChoice != nil {
		params.ToolChoice = oaicompat.ToToolChoice(req.ToolChoice)
	}

	applyOpenAIOptions(&params, req.Options.OpenAI)

	return params, nil
}

func toResult(resp *openai.ChatCompletion) *chat.Result {
	if resp == nil {
		return &chat.Result{Warnings: []string{"openai response is nil"}}
	}
	text := ""
	var toolCalls []chat.ToolCall
	for _, choice := range resp.Choices {
		text += choice.Message.Content
		if len(choice.Message.ToolCalls) > 0 && len(toolCalls) == 0 {
			toolCalls = oaicompat.ToToolCalls(choice.Message.ToolCalls)
		}
	}

	return &chat.Result{
		Text:      text,
		Model:     resp.Model,
		ToolCalls: toolCalls,
		Usage: chat.Usage{
			InputTokens:  int(resp.Usage.PromptTokens),
			OutputTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
		},
		Raw: resp,
	}
}

func useMaxCompletionTokens(model string) bool {
	model = strings.ToLower(model)
	return strings.HasPrefix(model, "gpt") ||
		strings.HasPrefix(model, "o1") ||
		strings.HasPrefix(model, "o3") ||
		strings.HasPrefix(model, "o4")
}

func applyOpenAIOptions(params *openai.ChatCompletionNewParams, opts structs.JSONMap) {
	if params == nil || len(opts) == 0 {
		return
	}
	opt := &opts
	if opt.HasKey("n") {
		if n := opt.GetInt64("n"); n > 0 {
			params.N = openai.Int(n)
		}
	}
	if opt.HasKey("seed") {
		params.Seed = openai.Int(opt.GetInt64("seed"))
	}
	if opt.HasKey("logprobs") {
		params.Logprobs = openai.Bool(opt.GetBool("logprobs"))
	}
	if opt.HasKey("top_logprobs") {
		if top := opt.GetInt64("top_logprobs"); top > 0 {
			params.TopLogprobs = openai.Int(top)
		}
	}
	if opt.HasKey("parallel_tool_calls") {
		params.ParallelToolCalls = openai.Bool(opt.GetBool("parallel_tool_calls"))
	}
	if opt.HasKey("store") {
		params.Store = openai.Bool(opt.GetBool("store"))
	}
	if opt.HasKey("prompt_cache_key") {
		if val := strings.TrimSpace(opt.GetString("prompt_cache_key")); val != "" {
			params.PromptCacheKey = openai.String(val)
		}
	}
	if opt.HasKey("safety_identifier") {
		if val := strings.TrimSpace(opt.GetString("safety_identifier")); val != "" {
			params.SafetyIdentifier = openai.String(val)
		}
	}
	if opt.HasKey("reasoning_effort") {
		if val := strings.TrimSpace(opt.GetString("reasoning_effort")); val != "" {
			params.ReasoningEffort = shared.ReasoningEffort(val)
		}
	}
	if opt.HasKey("verbosity") {
		if val := strings.TrimSpace(opt.GetString("verbosity")); val != "" {
			params.Verbosity = openai.ChatCompletionNewParamsVerbosity(val)
		}
	}
	if opt.HasKey("service_tier") {
		if val := strings.TrimSpace(opt.GetString("service_tier")); val != "" {
			params.ServiceTier = openai.ChatCompletionNewParamsServiceTier(val)
		}
	}
	if opt.HasKey("modalities") {
		if modalities := opt.GetStringArray("modalities"); len(modalities) > 0 {
			params.Modalities = append([]string{}, modalities...)
		}
	}
	if opt.HasKey("logit_bias") {
		if bias := oaicompat.ParseLogitBias((*opt)["logit_bias"]); len(bias) > 0 {
			params.LogitBias = bias
		}
	}
	if opt.HasKey("metadata") {
		if meta := oaicompat.ParseStringMap((*opt)["metadata"]); len(meta) > 0 {
			params.Metadata = shared.Metadata(meta)
		}
	}
	if opt.HasKey("response_format") {
		oaicompat.ApplyResponseFormat(params, (*opt)["response_format"])
	}
}
