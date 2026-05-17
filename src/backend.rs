use std::time::Instant;

use animus_plugin_protocol::{HealthCheckResult, HealthStatus};
use animus_provider_protocol::{
    AgentResumeRequest, AgentRunRequest, AgentRunResponse, BackendError, ProviderBackend,
    ProviderCapabilities, ProviderManifest, TokenUsage,
};
use async_trait::async_trait;

use crate::client::{ChatMessage, ChatRequest, OaiClient, OaiError};
use crate::config::OaiConfig;

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
        let model = request
            .model
            .clone()
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
                        "schema": schema,
                    }
                })
            }),
        }
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
                streaming: false,
                resume: false,
                cancellation: false,
                write_capable: false,
                mcp: false,
            },
        }
    }

    async fn run_agent(&self, request: AgentRunRequest) -> Result<AgentRunResponse, BackendError> {
        let started = Instant::now();
        let chat_request = self.build_chat_request(&request);
        let model_label = chat_request.model.clone();

        let response = self
            .client
            .chat(&chat_request)
            .await
            .map_err(|error| match error {
                OaiError::Api { status, message } if status == 401 || status == 403 => {
                    BackendError::Unavailable(format!("oai auth failed ({status}): {message}"))
                }
                OaiError::Api { status, message } if (500..600).contains(&status) => {
                    BackendError::Unavailable(format!("oai upstream {status}: {message}"))
                }
                OaiError::Api { status, message } => {
                    BackendError::RunFailed(format!("oai api {status}: {message}"))
                }
                OaiError::Http(error) => {
                    BackendError::RunFailed(format!("oai http error: {error}"))
                }
            })?;

        let output = response
            .choices
            .iter()
            .find_map(|choice| choice.message.content.clone())
            .unwrap_or_default();

        let session_id = if !response.id.is_empty() {
            response.id.clone()
        } else {
            uuid::Uuid::new_v4().to_string()
        };

        let tokens_used = response.usage.as_ref().map(|usage| TokenUsage {
            input: usage.prompt_tokens,
            output: usage.completion_tokens,
            cached: None,
            cache_writes: None,
        });

        let backend_label = if response.model.is_empty() {
            format!("oai:{model_label}")
        } else {
            format!("oai:{}", response.model)
        };

        Ok(AgentRunResponse {
            session_id,
            exit_code: 0,
            output,
            metadata: Vec::new(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            thinking: Vec::new(),
            errors: Vec::new(),
            duration_ms: started.elapsed().as_millis() as u64,
            backend: backend_label,
            tokens_used,
            decision_verdict: None,
        })
    }

    async fn resume_agent(
        &self,
        _request: AgentResumeRequest,
    ) -> Result<AgentRunResponse, BackendError> {
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
