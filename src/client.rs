use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::config::OaiConfig;

#[derive(Debug, Error)]
pub enum OaiError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("openai api error ({status}): {message}")]
    Api { status: u16, message: String },
    #[error("OPENAI_API_KEY is required for animus-provider-oai run/resume calls")]
    MissingApiKey,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<serde_json::Value>,
}

/// SSE event surfaced by the streaming completions API.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Incremental text delta from the assistant message.
    Delta(String),
    /// One tool invocation requested by the assistant. OpenAI streams
    /// `delta.tool_calls[]` in fragments — `function.name` arrives once,
    /// `function.arguments` chunks across multiple deltas — so the client
    /// accumulates them and surfaces a single event per tool call once the
    /// stream signals `finish_reason: tool_calls`.
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    /// Final `[DONE]` sentinel followed by aggregated metadata. The client
    /// emits this after the SSE stream closes and carries the synthesized
    /// final transcript plus best-effort id / model / usage.
    Done {
        session_id: Option<String>,
        model: Option<String>,
        usage: Option<ChatUsage>,
    },
    /// Recoverable mid-stream API error (e.g. server-side abort).
    Error(String),
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<StreamChunkChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
struct StreamChunkChoice {
    #[serde(default)]
    delta: StreamChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct StreamChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<StreamChunkToolCall>,
}

#[derive(Debug, Deserialize)]
struct StreamChunkToolCall {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamChunkToolCallFunction>,
}

#[derive(Debug, Deserialize)]
struct StreamChunkToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    #[serde(default)]
    pub index: u32,
    pub message: ChatChoiceMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoiceMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Deserialize)]
pub struct ModelsResponse {
    #[serde(default)]
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
pub struct ModelEntry {
    #[serde(default)]
    pub id: String,
}

pub struct OaiClient {
    inner: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    org: Option<String>,
}

impl OaiClient {
    pub fn new(config: &OaiConfig) -> Self {
        let inner = reqwest::Client::builder()
            .user_agent(concat!("animus-provider-oai/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("reqwest client");
        Self {
            inner,
            base_url: config.base_url.clone(),
            api_key: config.api_key.clone(),
            org: config.org.clone(),
        }
    }

    pub fn has_api_key(&self) -> bool {
        self.api_key.as_deref().is_some_and(|key| !key.is_empty())
    }

    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, OaiError> {
        let api_key = self.api_key.as_deref().ok_or(OaiError::MissingApiKey)?;
        let url = format!("{}/chat/completions", self.base_url);
        let mut req = self.inner.post(&url).bearer_auth(api_key).json(request);
        if let Some(org) = &self.org {
            req = req.header("OpenAI-Organization", org);
        }
        let response = req.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(OaiError::Api {
                status: status.as_u16(),
                message: body,
            });
        }
        Ok(response.json::<ChatResponse>().await?)
    }

    /// Open an SSE-streaming chat completions request. The returned receiver
    /// emits one `StreamEvent::Delta` per token-delta chunk, terminating with
    /// either a `StreamEvent::Done` (graceful) or a `StreamEvent::Error`
    /// (mid-stream failure). HTTP-level errors (auth, 5xx, transport) are
    /// returned eagerly as the function result so callers can surface them as
    /// `BackendError::Unavailable` / `BackendError::RunFailed` before any
    /// notifications are emitted.
    pub async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<mpsc::Receiver<StreamEvent>, OaiError> {
        let api_key = self.api_key.as_deref().ok_or(OaiError::MissingApiKey)?;
        let url = format!("{}/chat/completions", self.base_url);

        let mut body = serde_json::to_value(request).map_err(|e| OaiError::Api {
            status: 0,
            message: format!("serialize request: {e}"),
        })?;
        if let Some(map) = body.as_object_mut() {
            map.insert("stream".to_string(), serde_json::Value::Bool(true));
            map.insert(
                "stream_options".to_string(),
                serde_json::json!({ "include_usage": true }),
            );
        }

        let mut req = self
            .inner
            .post(&url)
            .bearer_auth(api_key)
            .header("Accept", "text/event-stream")
            .json(&body);
        if let Some(org) = &self.org {
            req = req.header("OpenAI-Organization", org);
        }

        let response = req.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(OaiError::Api {
                status: status.as_u16(),
                message: body_text,
            });
        }

        let (tx, rx) = mpsc::channel::<StreamEvent>(32);
        tokio::spawn(async move {
            let mut stream = response.bytes_stream();
            let mut buffer: Vec<u8> = Vec::new();
            let mut session_id: Option<String> = None;
            let mut model: Option<String> = None;
            let mut usage: Option<ChatUsage> = None;
            let mut pending_tool_calls: Vec<PendingToolCall> = Vec::new();

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.extend_from_slice(&bytes);
                        while let Some(event_end) = find_event_boundary(&buffer) {
                            let raw = buffer.drain(..event_end).collect::<Vec<u8>>();
                            let frame = String::from_utf8_lossy(&raw).to_string();
                            for line in frame.lines() {
                                let line = line.trim_start_matches('\r');
                                let Some(data) = line.strip_prefix("data:") else {
                                    continue;
                                };
                                let data = data.trim();
                                if data.is_empty() {
                                    continue;
                                }
                                if data == "[DONE]" {
                                    let _ = tx
                                        .send(StreamEvent::Done {
                                            session_id: session_id.clone(),
                                            model: model.clone(),
                                            usage: usage.clone(),
                                        })
                                        .await;
                                    return;
                                }
                                match serde_json::from_str::<StreamChunk>(data) {
                                    Ok(parsed) => {
                                        if session_id.is_none() {
                                            session_id = parsed.id;
                                        }
                                        if model.is_none() {
                                            model = parsed.model;
                                        }
                                        if let Some(u) = parsed.usage {
                                            usage = Some(u);
                                        }
                                        for choice in parsed.choices {
                                            if let Some(text) = choice.delta.content {
                                                if !text.is_empty()
                                                    && tx
                                                        .send(StreamEvent::Delta(text))
                                                        .await
                                                        .is_err()
                                                {
                                                    return;
                                                }
                                            }
                                            for tc in choice.delta.tool_calls {
                                                accumulate_tool_call(&mut pending_tool_calls, tc);
                                            }
                                            if matches!(
                                                choice.finish_reason.as_deref(),
                                                Some("tool_calls")
                                            ) {
                                                for pending in pending_tool_calls.drain(..) {
                                                    if tx
                                                        .send(StreamEvent::ToolCall {
                                                            id: pending.id,
                                                            name: pending.name,
                                                            arguments: pending.arguments,
                                                        })
                                                        .await
                                                        .is_err()
                                                    {
                                                        return;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx
                                            .send(StreamEvent::Error(format!(
                                                "malformed sse chunk: {e}: {data}"
                                            )))
                                            .await;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(StreamEvent::Error(format!("sse transport error: {e}")))
                            .await;
                        return;
                    }
                }
            }
            // Stream closed without an explicit `[DONE]` — flush any
            // accumulated tool_calls before signalling completion so callers
            // don't lose tool-call deltas on truncated streams.
            for pending in pending_tool_calls.drain(..) {
                if tx
                    .send(StreamEvent::ToolCall {
                        id: pending.id,
                        name: pending.name,
                        arguments: pending.arguments,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }
            let _ = tx
                .send(StreamEvent::Done {
                    session_id,
                    model,
                    usage,
                })
                .await;
        });

        Ok(rx)
    }

    pub async fn models(&self) -> Result<ModelsResponse, OaiError> {
        let api_key = self.api_key.as_deref().ok_or(OaiError::MissingApiKey)?;
        let url = format!("{}/models", self.base_url);
        let mut req = self.inner.get(&url).bearer_auth(api_key);
        if let Some(org) = &self.org {
            req = req.header("OpenAI-Organization", org);
        }
        let response = req.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(OaiError::Api {
                status: status.as_u16(),
                message: body,
            });
        }
        Ok(response.json::<ModelsResponse>().await?)
    }
}

// OpenAI streams tool_calls as fragmented deltas: the first delta carries
// `id` + `function.name`, subsequent deltas append `function.arguments`
// shards. We key by `index` (positional, stable per call) and accumulate
// name + arguments across chunks until `finish_reason: tool_calls`.
fn accumulate_tool_call(pending: &mut Vec<PendingToolCall>, fragment: StreamChunkToolCall) {
    let idx = fragment.index as usize;
    if pending.len() <= idx {
        pending.resize_with(idx + 1, PendingToolCall::default);
    }
    let slot = &mut pending[idx];
    if let Some(id) = fragment.id {
        if !id.is_empty() {
            slot.id = id;
        }
    }
    if let Some(function) = fragment.function {
        if let Some(name) = function.name {
            if !name.is_empty() {
                slot.name = name;
            }
        }
        if let Some(arguments) = function.arguments {
            slot.arguments.push_str(&arguments);
        }
    }
}

fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    for i in 0..buf.len().saturating_sub(1) {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some(i + 2);
        }
        if i + 3 < buf.len()
            && buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some(i + 4);
        }
    }
    None
}
