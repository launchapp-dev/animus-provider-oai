use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::OaiConfig;

#[derive(Debug, Error)]
pub enum OaiError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("openai api error ({status}): {message}")]
    Api { status: u16, message: String },
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

#[derive(Debug, Deserialize)]
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
    api_key: String,
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

    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, OaiError> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut req = self
            .inner
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(request);
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

    pub async fn models(&self) -> Result<ModelsResponse, OaiError> {
        let url = format!("{}/models", self.base_url);
        let mut req = self.inner.get(&url).bearer_auth(&self.api_key);
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
