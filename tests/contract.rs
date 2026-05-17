use std::collections::HashMap;
use std::path::PathBuf;

use animus_plugin_protocol::HealthStatus;
use animus_provider_oai::backend::OaiBackend;
use animus_provider_oai::config::OaiConfig;
use animus_provider_protocol::{AgentRunRequest, BackendError, ProviderBackend};
use serde_json::json;

fn run_request(model: Option<&str>, prompt: &str) -> AgentRunRequest {
    AgentRunRequest {
        session_id: None,
        prompt: prompt.to_string(),
        model: model.map(|s| s.to_string()),
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

#[tokio::test]
async fn run_agent_returns_assistant_message() {
    let mut server = mockito::Server::new_async().await;
    let body = json!({
        "id": "chatcmpl-test-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-5",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello from oai"
            },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13 }
    })
    .to_string();
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(body)
        .create_async()
        .await;

    let backend = OaiBackend::new(OaiConfig::for_testing(server.url()));
    let response = backend
        .run_agent(run_request(Some("gpt-5"), "ping"))
        .await
        .expect("run_agent should succeed");

    mock.assert_async().await;
    assert_eq!(response.output, "hello from oai");
    assert_eq!(response.session_id, "chatcmpl-test-1");
    assert_eq!(response.exit_code, 0);
    let tokens = response.tokens_used.expect("tokens reported");
    assert_eq!(tokens.input, 9);
    assert_eq!(tokens.output, 4);
}

#[tokio::test]
async fn run_agent_propagates_model() {
    let mut server = mockito::Server::new_async().await;
    let body = json!({
        "id": "chatcmpl-test-2",
        "model": "custom-model-xyz",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let mock = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(
            json!({ "model": "custom-model-xyz" }),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(body)
        .create_async()
        .await;

    let backend = OaiBackend::new(OaiConfig::for_testing(server.url()));
    let response = backend
        .run_agent(run_request(Some("custom-model-xyz"), "what model?"))
        .await
        .expect("run_agent should succeed");

    mock.assert_async().await;
    assert!(response.backend.contains("custom-model-xyz"));
}

#[tokio::test]
async fn health_returns_healthy_on_models_200() {
    let mut server = mockito::Server::new_async().await;
    let body = json!({ "object": "list", "data": [ { "id": "gpt-5" } ] }).to_string();
    let _mock = server
        .mock("GET", "/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(body)
        .create_async()
        .await;

    let backend = OaiBackend::new(OaiConfig::for_testing(server.url()));
    let health = backend.health().await.expect("health should not error");
    assert_eq!(health.status, HealthStatus::Healthy);
    assert!(health.last_error.is_none());
}

#[tokio::test]
async fn health_returns_unhealthy_on_401() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("GET", "/models")
        .with_status(401)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error": {"message": "invalid api key"}}"#)
        .create_async()
        .await;

    let backend = OaiBackend::new(OaiConfig::for_testing(server.url()));
    let health = backend.health().await.expect("health should not error");
    assert_eq!(health.status, HealthStatus::Unhealthy);
    assert!(health.last_error.is_some());
}

#[tokio::test]
async fn resume_agent_returns_unsupported() {
    let server = mockito::Server::new_async().await;
    let backend = OaiBackend::new(OaiConfig::for_testing(server.url()));
    let err = backend
        .resume_agent(run_request(Some("gpt-5"), "resume"))
        .await
        .expect_err("resume should be unsupported");
    let msg = format!("{err}");
    assert!(
        msg.contains("not supported"),
        "error message did not mention 'not supported': {msg}"
    );
}

/// Regression: `OaiConfig::from_env()` must succeed even when
/// `OPENAI_API_KEY` is unset so that credential-free plugin lifecycle
/// calls (notably `--manifest` and `health`) can run in shells without
/// secrets. The previous behavior aborted in `main()` before the runtime
/// could honor `--manifest`, which made the plugin invisible to
/// `animus plugin list` whenever the listing shell lacked credentials.
#[test]
fn from_env_succeeds_without_credentials() {
    let _guard = EnvKeyGuard::clear();
    let config = OaiConfig::from_env().expect("from_env must not require OPENAI_API_KEY");
    assert!(
        config.api_key.is_none(),
        "api_key should be None when OPENAI_API_KEY is unset, got {:?}",
        config.api_key
    );
}

/// Regression: a credential-free backend must report itself as
/// `Unhealthy` with a clear `last_error` instead of either panicking
/// or making a doomed network request.
#[tokio::test]
async fn health_reports_unhealthy_when_credentials_missing() {
    let backend = OaiBackend::new(OaiConfig {
        api_key: None,
        base_url: "http://127.0.0.1:1".to_string(),
        org: None,
        default_model: "gpt-5".to_string(),
    });
    let health = backend.health().await.expect("health should not error");
    assert_eq!(health.status, HealthStatus::Unhealthy);
    let last_error = health.last_error.expect("last_error should be set");
    assert!(
        last_error.contains("OPENAI_API_KEY"),
        "last_error did not mention OPENAI_API_KEY: {last_error}"
    );
}

/// Regression: when credentials are missing, `run_agent` must surface a
/// clear `BackendError::Unavailable` *before* attempting any network
/// call, so the host can advertise the plugin in `plugin list` and only
/// fail at actual use sites.
#[tokio::test]
async fn run_agent_requires_credentials() {
    let backend = OaiBackend::new(OaiConfig {
        api_key: None,
        base_url: "http://127.0.0.1:1".to_string(),
        org: None,
        default_model: "gpt-5".to_string(),
    });
    let err = backend
        .run_agent(run_request(Some("gpt-5"), "hi"))
        .await
        .expect_err("run_agent should require OPENAI_API_KEY");
    match err {
        BackendError::Unavailable(msg) => assert!(
            msg.contains("OPENAI_API_KEY"),
            "Unavailable message did not mention OPENAI_API_KEY: {msg}"
        ),
        other => panic!("expected BackendError::Unavailable, got {other:?}"),
    }
}

/// Regression: the binary entrypoint must print manifest JSON without
/// requiring any environment variables. This invokes the real built
/// binary so we exercise `main()` end-to-end, mirroring how the plugin
/// host discovers plugins in `animus plugin list`.
#[test]
fn main_emits_manifest_without_credentials() {
    let binary = std::path::PathBuf::from(env!("CARGO_BIN_EXE_animus-provider-oai"));
    let output = std::process::Command::new(&binary)
        .arg("--manifest")
        .env_remove("OPENAI_API_KEY")
        .output()
        .expect("failed to spawn animus-provider-oai");
    assert!(
        output.status.success(),
        "--manifest exited with {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let manifest: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("stdout should be JSON");
    assert_eq!(manifest["name"], "animus-provider-oai");
    assert_eq!(manifest["plugin_kind"], "provider");
}

/// Test helper: temporarily clear `OPENAI_API_KEY` and restore it on
/// drop so tests touching env vars stay isolated.
struct EnvKeyGuard {
    previous: Option<String>,
}

impl EnvKeyGuard {
    fn clear() -> Self {
        let previous = std::env::var("OPENAI_API_KEY").ok();
        // SAFETY: tests in this file are scoped, but env mutation is
        // process-global. We only flip a single var and restore it.
        std::env::remove_var("OPENAI_API_KEY");
        Self { previous }
    }
}

impl Drop for EnvKeyGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => std::env::set_var("OPENAI_API_KEY", value),
            None => std::env::remove_var("OPENAI_API_KEY"),
        }
    }
}
