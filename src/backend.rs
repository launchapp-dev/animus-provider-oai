use std::time::Instant;

use animus_plugin_protocol::{HealthCheckResult, HealthStatus};
use animus_provider_protocol::{
    AgentNotification, AgentResumeRequest, AgentRunRequest, AgentRunResponse, BackendError,
    NotificationSink, ProviderBackend, ProviderCapabilities, ProviderManifest, TokenUsage,
};
use async_trait::async_trait;

use crate::client::{ChatMessage, ChatRequest, OaiClient, OaiError, StreamEvent};
use crate::config::{OaiConfig, MISSING_API_KEY_MESSAGE};
use crate::gating::{self, GateContext};

/// Default human-escalation timeout (seconds) for the approve-hook when the
/// request carries no explicit `timeout_secs`. Matches the ACP provider.
const DEFAULT_APPROVAL_TIMEOUT_SECS: u64 = 300;

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

/// Build the approval gate context from an `AgentRunRequest`.
///
/// Adapted to this provider protocol where `runtime_contract` is a SEPARATE
/// field (not nested inside `extras`, unlike the ACP `SessionRequest`).
fn build_gate_context(request: &AgentRunRequest) -> GateContext {
    let approvals_enabled = approvals_enabled(request);
    let agent_id = resolve_agent_id(request).unwrap_or_else(|| "default".to_string());
    let project_root = request
        .project_root
        .clone()
        .unwrap_or_else(|| request.cwd.clone());
    let animus_bin = std::env::var("ANIMUS_BIN")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "animus".to_string());
    let timeout_secs = request
        .timeout_secs
        .unwrap_or(DEFAULT_APPROVAL_TIMEOUT_SECS);

    GateContext {
        approvals_enabled,
        agent_id,
        project_root,
        animus_bin,
        timeout_secs,
    }
}

/// Whether the run opted into human-in-the-loop approvals.
///
/// Fails SAFE: this can only turn gating ON, never off. Approvals are enabled
/// when EITHER:
/// - top-level `extras.approvals == true`, or
/// - the kernel pinned an approval identity at `runtime_contract.mcp.agent_id`.
fn approvals_enabled(request: &AgentRunRequest) -> bool {
    if request
        .extras
        .get("approvals")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return true;
    }
    request
        .runtime_contract
        .as_ref()
        .and_then(|rc| rc.pointer("/mcp/agent_id"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| !s.trim().is_empty())
}

/// Resolve the agent profile id from `runtime_contract.mcp.agent_id`, falling
/// back to a top-level `extras.agent_id`.
fn resolve_agent_id(request: &AgentRunRequest) -> Option<String> {
    if let Some(id) = request
        .runtime_contract
        .as_ref()
        .and_then(|rc| rc.pointer("/mcp/agent_id"))
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.trim().is_empty())
    {
        return Some(id.to_string());
    }
    request
        .extras
        .get("agent_id")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(ToString::to_string)
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
        let gate_ctx = build_gate_context(&request);
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
        let mut tool_results: Vec<serde_json::Value> = Vec::new();

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
                    // Approval gate: route the call through the Animus
                    // approve-hook before recording it. Pass-through when
                    // approvals are disabled; fails CLOSED on any error. An
                    // `allow` may carry an edited input we MUST honor instead of
                    // the original arguments.
                    match gating::gate_tool_call(&gate_ctx, &name, &arguments_value).await {
                        gating::GateOutcome::Allow {
                            input: approved_input,
                        } => {
                            sink.emit(AgentNotification::ToolCall {
                                session_id: session_id.clone().unwrap_or_default(),
                                name: name.clone(),
                                arguments: approved_input.clone(),
                                server: None,
                            });
                            tool_calls.push(serde_json::json!({
                                "id": id,
                                "name": name,
                                "arguments": approved_input,
                            }));
                        }
                        gating::GateOutcome::Deny => {
                            // Denied: do NOT emit the call or record it. Surface
                            // a tool error to the model and a visible run error.
                            tool_results.push(serde_json::json!({
                                "id": id,
                                "name": name,
                                "status": "denied",
                                "error": "denied by approval policy",
                            }));
                            let denial_message =
                                format!("tool call `{name}` denied by approval policy");
                            sink.emit(AgentNotification::Error {
                                session_id: session_id.clone().unwrap_or_default(),
                                message: denial_message.clone(),
                                recoverable: true,
                            });
                            errors.push(denial_message);
                        }
                    }
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
            tool_results,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn minimal_request() -> AgentRunRequest {
        AgentRunRequest {
            session_id: None,
            prompt: "do a thing".to_string(),
            model: None,
            system_prompt: None,
            cwd: PathBuf::from("/tmp"),
            project_root: None,
            permission_mode: None,
            timeout_secs: None,
            env: HashMap::new(),
            mcp_servers: None,
            tools: None,
            response_schema: None,
            runtime_contract: None,
            extras: HashMap::new(),
        }
    }

    #[test]
    fn approvals_enabled_via_runtime_contract_agent_id() {
        let mut req = minimal_request();
        req.runtime_contract = Some(serde_json::json!({ "mcp": { "agent_id": "swe" } }));
        let ctx = build_gate_context(&req);
        assert!(ctx.approvals_enabled);
        assert_eq!(ctx.agent_id, "swe");
    }

    #[test]
    fn approvals_enabled_via_extras_flag() {
        let mut req = minimal_request();
        req.extras
            .insert("approvals".to_string(), serde_json::json!(true));
        let ctx = build_gate_context(&req);
        assert!(ctx.approvals_enabled);
        // No pinned agent id → falls back to "default".
        assert_eq!(ctx.agent_id, "default");
    }

    #[test]
    fn agent_id_falls_back_to_extras() {
        let mut req = minimal_request();
        req.extras
            .insert("approvals".to_string(), serde_json::json!(true));
        req.extras
            .insert("agent_id".to_string(), serde_json::json!("reviewer"));
        let ctx = build_gate_context(&req);
        assert!(ctx.approvals_enabled);
        assert_eq!(ctx.agent_id, "reviewer");
    }

    #[test]
    fn runtime_contract_agent_id_wins_over_extras() {
        let mut req = minimal_request();
        req.runtime_contract = Some(serde_json::json!({ "mcp": { "agent_id": "swe" } }));
        req.extras
            .insert("agent_id".to_string(), serde_json::json!("reviewer"));
        let ctx = build_gate_context(&req);
        assert_eq!(ctx.agent_id, "swe");
    }

    #[test]
    fn approvals_disabled_when_neither_present() {
        let req = minimal_request();
        let ctx = build_gate_context(&req);
        assert!(!ctx.approvals_enabled);
        assert_eq!(ctx.agent_id, "default");
    }

    #[test]
    fn project_root_falls_back_to_cwd() {
        let req = minimal_request();
        let ctx = build_gate_context(&req);
        assert_eq!(ctx.project_root, PathBuf::from("/tmp"));
    }

    #[test]
    fn default_timeout_applied_when_unset() {
        let req = minimal_request();
        let ctx = build_gate_context(&req);
        assert_eq!(ctx.timeout_secs, DEFAULT_APPROVAL_TIMEOUT_SECS);
    }
}
