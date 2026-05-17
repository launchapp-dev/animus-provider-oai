use std::collections::HashMap;
use std::path::PathBuf;

use animus_plugin_protocol::HealthStatus;
use animus_provider_oai::backend::OaiBackend;
use animus_provider_oai::config::OaiConfig;
use animus_provider_protocol::{AgentRunRequest, ProviderBackend};
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
