use std::time::Instant;

use animus_plugin_protocol::{HealthCheckResult, HealthStatus};
use animus_provider_protocol::{
    AgentNotification, AgentResumeRequest, AgentRunRequest, AgentRunResponse, BackendError,
    NotificationSink, ProviderBackend, ProviderCapabilities, ProviderManifest, TokenUsage,
};
use async_trait::async_trait;

use crate::client::{ChatMessage, ChatRequest, OaiClient, OaiError, StreamEvent};
use crate::config::{OaiConfig, MISSING_API_KEY_MESSAGE};

pub struct OaiBackend {
    client: OaiClient,
    config: OaiConfig,
}

impl OaiBackend {
    pub fn new(config: OaiConfig) -> Self {
        let client = OaiClient::new(&config);
        Self { client, config }
    }

    fn build_chat_request(&self, request: &AgentRunRequest) -> ChatRequest {
        // Strip a leading `openrouter/` routing prefix. Animus uses that prefix to
        // route a model to the oai-runner tool, but OpenRouter's API expects the
        // bare `<vendor>/<model>` id (e.g. `minimax/minimax-m2.7`). No-op for ids
        // that don't carry the prefix, so direct-OpenAI/other bases are unaffected.
        let model = request
            .model
            .clone()
            .map(|m| m.strip_prefix("openrouter/").map(str::to_string).unwrap_or(m))
            .unwrap_or_else(|| self.config.default_model.clone());
        let mut messages = Vec::new();
        if let Some(system) = &request.system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: request.prompt.clone(),
        });

        ChatRequest {
            model,
            messages,
            temperature: None,
            max_tokens: None,
            response_format: request.response_schema.clone().map(|schema| {
                serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "animus_response",
                        // strict: true makes OpenAI/OpenRouter ENFORCE the schema
                        // (the model is constrained to emit valid, fully-populated
                        // JSON matching it). Without strict the schema is only a
                        // hint and weak models return a null/partial skeleton.
                        // Requires strict-compatible schemas: additionalProperties
                        // false, all properties required, no minLength/format.
                        "strict": true,
                        "schema": schema,
                    }
                })
            }),
            stream: None,
            stream_options: None,
        }
    }
}

fn map_oai_error(error: OaiError) -> BackendError {
    match error {
        OaiError::Api { status, message } if status == 401 || status == 403 => {
            BackendError::Unavailable(format!("oai auth failed ({status}): {message}"))
        }
        OaiError::Api { status, message } if (500..600).contains(&status) => {
            BackendError::Unavailable(format!("oai upstream {status}: {message}"))
        }
        OaiError::Api { status, message } => {
            BackendError::RunFailed(format!("oai api {status}: {message}"))
        }
        OaiError::Http(error) => BackendError::RunFailed(format!("oai http error: {error}")),
        OaiError::MissingApiKey => BackendError::Unavailable(MISSING_API_KEY_MESSAGE.to_string()),
    }
}

#[async_trait]
impl ProviderBackend for OaiBackend {
    fn manifest(&self) -> ProviderManifest {
        ProviderManifest {
            name: env!("CARGO_PKG_NAME").to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: env!("CARGO_PKG_DESCRIPTION").to_string(),
            supported_models: vec![
                "gpt-5".to_string(),
                "gpt-5-mini".to_string(),
                "gpt-4o".to_string(),
                "gpt-4o-mini".to_string(),
                "gpt-4.1".to_string(),
                "gpt-4.1-mini".to_string(),
            ],
            tool: "oai".to_string(),
            capabilities: ProviderCapabilities {
                streaming: true,
                resume: false,
                cancellation: false,
                write_capable: false,
                mcp: false,
            },
        }
    }

    async fn run_agent(&self, request: AgentRunRequest) -> Result<AgentRunResponse, BackendError> {
        self.run_agent_streaming(request, NotificationSink::noop())
            .await
    }

    async fn run_agent_streaming(
        &self,
        request: AgentRunRequest,
        sink: NotificationSink,
    ) -> Result<AgentRunResponse, BackendError> {
        if !self.client.has_api_key() {
            return Err(BackendError::Unavailable(
                MISSING_API_KEY_MESSAGE.to_string(),
            ));
        }
        let started = Instant::now();
        let chat_request = self.build_chat_request(&request);
        let model_label = chat_request.model.clone();

        let mut rx = self
            .client
            .chat_stream(&chat_request)
            .await
            .map_err(map_oai_error)?;

        let mut output_text = String::new();
        let mut errors: Vec<String> = Vec::new();
        let mut session_id: Option<String> = None;
        let mut response_model: Option<String> = None;
        let mut tokens_used: Option<TokenUsage> = None;
        let mut pending_session_id: Option<String> = None;
        let mut tool_calls: Vec<serde_json::Value> = Vec::new();

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Delta(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    output_text.push_str(&text);
                    sink.emit(AgentNotification::Output {
                        session_id: session_id.clone().unwrap_or_default(),
                        text,
                        is_final: false,
                    });
                }
                StreamEvent::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    // OpenAI sends `function.arguments` as a JSON-encoded
                    // string. Try to parse it back so downstream consumers
                    // get structured args; fall back to a raw string blob
                    // if the model emitted invalid JSON.
                    let arguments_value: serde_json::Value = if arguments.is_empty() {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str(&arguments)
                            .unwrap_or_else(|_| serde_json::Value::String(arguments.clone()))
                    };
                    sink.emit(AgentNotification::ToolCall {
                        session_id: session_id.clone().unwrap_or_default(),
                        name: name.clone(),
                        arguments: arguments_value.clone(),
                        server: None,
                    });
                    tool_calls.push(serde_json::json!({
                        "id": id,
                        "name": name,
                        "arguments": arguments_value,
                    }));
                }
                StreamEvent::Done {
                    session_id: sid,
                    model: m,
                    usage,
                } => {
                    pending_session_id = sid;
                    response_model = m;
                    tokens_used = usage.map(|u| TokenUsage {
                        input: u.prompt_tokens,
                        output: u.completion_tokens,
                        cached: None,
                        cache_writes: None,
                    });
                }
                StreamEvent::Error(message) => {
                    sink.emit(AgentNotification::Error {
                        session_id: session_id.clone().unwrap_or_default(),
                        message: message.clone(),
                        recoverable: true,
                    });
                    errors.push(message);
                }
            }
        }

        if session_id.is_none() {
            session_id = pending_session_id;
        }
        let session_id = session_id
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        sink.emit(AgentNotification::Output {
            session_id: session_id.clone(),
            text: output_text.clone(),
            is_final: true,
        });

        let backend_label = match response_model {
            Some(m) if !m.is_empty() => format!("oai:{m}"),
            _ => format!("oai:{model_label}"),
        };

        Ok(AgentRunResponse {
            session_id,
            exit_code: 0,
            output: output_text,
            metadata: Vec::new(),
            tool_calls,
            tool_results: Vec::new(),
            thinking: Vec::new(),
            errors,
            duration_ms: started.elapsed().as_millis() as u64,
            backend: backend_label,
            tokens_used,
            decision_verdict: None,
        })
    }

    async fn resume_agent(
        &self,
        request: AgentResumeRequest,
    ) -> Result<AgentRunResponse, BackendError> {
        self.resume_agent_streaming(request, NotificationSink::noop())
            .await
    }

    async fn resume_agent_streaming(
        &self,
        _request: AgentResumeRequest,
        _sink: NotificationSink,
    ) -> Result<AgentRunResponse, BackendError> {
        if !self.client.has_api_key() {
            return Err(BackendError::Unavailable(
                MISSING_API_KEY_MESSAGE.to_string(),
            ));
        }
        Err(BackendError::Other(anyhow::anyhow!(
            "oai: resume not supported (stateless HTTP API)"
        )))
    }

    async fn cancel_agent(&self, _session_id: &str) -> Result<(), BackendError> {
        Err(BackendError::Other(anyhow::anyhow!(
            "oai: cancel not supported (synchronous HTTP API)"
        )))
    }

    async fn health(&self) -> Result<HealthCheckResult, BackendError> {
        if !self.client.has_api_key() {
            return Ok(HealthCheckResult {
                status: HealthStatus::Unhealthy,
                uptime_ms: None,
                memory_usage_bytes: None,
                last_error: Some(MISSING_API_KEY_MESSAGE.to_string()),
            });
        }
        match self.client.models().await {
            Ok(_) => Ok(HealthCheckResult {
                status: HealthStatus::Healthy,
                uptime_ms: None,
                memory_usage_bytes: None,
                last_error: None,
            }),
            Err(error) => Ok(HealthCheckResult {
                status: HealthStatus::Unhealthy,
                uptime_ms: None,
                memory_usage_bytes: None,
                last_error: Some(error.to_string()),
            }),
        }
    }
}
