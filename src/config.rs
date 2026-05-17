use anyhow::{anyhow, Result};

#[derive(Debug, Clone)]
pub struct OaiConfig {
    pub api_key: String,
    pub base_url: String,
    pub org: Option<String>,
    pub default_model: String,
}

impl OaiConfig {
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow!("OPENAI_API_KEY is required for animus-provider-oai"))?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string())
            .trim_end_matches('/')
            .to_string();
        let org = std::env::var("OPENAI_ORG").ok().filter(|s| !s.is_empty());
        let default_model =
            std::env::var("OPENAI_DEFAULT_MODEL").unwrap_or_else(|_| "gpt-5".to_string());

        Ok(Self {
            api_key,
            base_url,
            org,
            default_model,
        })
    }

    /// Helper for integration tests / embedders that want to construct a
    /// config without going through env vars.
    pub fn for_testing(base_url: impl Into<String>) -> Self {
        Self {
            api_key: "test-key".to_string(),
            base_url: base_url.into(),
            org: None,
            default_model: "gpt-5".to_string(),
        }
    }
}
