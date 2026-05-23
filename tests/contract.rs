use std::collections::HashMap;
use std::path::PathBuf;

use std::sync::{Arc, Mutex};

use animus_plugin_protocol::HealthStatus;
use animus_provider_oai::backend::OaiBackend;
use animus_provider_oai::config::OaiConfig;
use animus_provider_protocol::{
    AgentNotification, AgentRunRequest, BackendError, NotificationSink, ProviderBackend,
};
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

fn sse_script(chunks: &[serde_json::Value]) -> String {
    let mut s = String::new();
    for chunk in chunks {
        s.push_str("data: ");
        s.push_str(&chunk.to_string());
        s.push_str("\n\n");
    }
    s.push_str("data: [DONE]\n\n");
    s
}

#[tokio::test]
async fn run_agent_returns_assistant_message() {
    let mut server = mockito::Server::new_async().await;
    let body = sse_script(&[
        json!({
            "id": "chatcmpl-test-1",
            "object": "chat.completion.chunk",
            "model": "gpt-5",
            "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "hello " } }]
        }),
        json!({
            "id": "chatcmpl-test-1",
            "model": "gpt-5",
            "choices": [{ "index": 0, "delta": { "content": "from oai" }, "finish_reason": "stop" }]
        }),
        json!({
            "id": "chatcmpl-test-1",
            "model": "gpt-5",
            "choices": [],
            "usage": { "prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13 }
        }),
    ]);
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
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
    let body = sse_script(&[json!({
        "id": "chatcmpl-test-2",
        "model": "custom-model-xyz",
        "choices": [{ "index": 0, "delta": { "content": "ok" }, "finish_reason": "stop" }]
    })]);
    let mock = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(
            json!({ "model": "custom-model-xyz" }),
        ))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
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
async fn run_agent_streaming_emits_per_delta_notifications() {
    let mut server = mockito::Server::new_async().await;
    let body = sse_script(&[
        json!({
            "id": "chatcmpl-stream",
            "model": "gpt-5",
            "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "alpha " } }]
        }),
        json!({
            "id": "chatcmpl-stream",
            "model": "gpt-5",
            "choices": [{ "index": 0, "delta": { "content": "beta " } }]
        }),
        json!({
            "id": "chatcmpl-stream",
            "model": "gpt-5",
            "choices": [{ "index": 0, "delta": { "content": "gamma" }, "finish_reason": "stop" }]
        }),
    ]);
    let _mock = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({ "stream": true })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
        .create_async()
        .await;

    let recorder: Arc<Mutex<Vec<AgentNotification>>> = Arc::new(Mutex::new(Vec::new()));
    let r2 = Arc::clone(&recorder);
    let sink = NotificationSink::new(move |n| r2.lock().unwrap().push(n));

    let backend = OaiBackend::new(OaiConfig::for_testing(server.url()));
    let response = backend
        .run_agent_streaming(run_request(Some("gpt-5"), "stream"), sink)
        .await
        .expect("streaming run should succeed");

    assert_eq!(response.output, "alpha beta gamma");
    assert_eq!(response.session_id, "chatcmpl-stream");

    let notifications = recorder.lock().unwrap().clone();
    assert_eq!(
        notifications.len(),
        4,
        "expected 3 deltas + 1 final aggregate = 4, got {notifications:?}"
    );

    for (idx, expected_text, expected_final) in [
        (0usize, "alpha ", false),
        (1, "beta ", false),
        (2, "gamma", false),
        (3, "alpha beta gamma", true),
    ] {
        match &notifications[idx] {
            AgentNotification::Output { text, is_final, .. } => {
                assert_eq!(text, expected_text, "delta {idx}");
                assert_eq!(*is_final, expected_final, "is_final at {idx}");
            }
            other => panic!("expected Output at {idx}, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn run_agent_noop_sink_path_matches_streaming_response() {
    let make_body = || {
        sse_script(&[
            json!({
                "id": "chatcmpl-parity",
                "model": "gpt-5",
                "choices": [{ "index": 0, "delta": { "content": "abc" } }]
            }),
            json!({
                "id": "chatcmpl-parity",
                "model": "gpt-5",
                "choices": [{ "index": 0, "delta": { "content": "!" }, "finish_reason": "stop" }]
            }),
        ])
    };

    let mut bulk_server = mockito::Server::new_async().await;
    let _bulk_mock = bulk_server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(make_body())
        .create_async()
        .await;
    let bulk_backend = OaiBackend::new(OaiConfig::for_testing(bulk_server.url()));
    let bulk = bulk_backend
        .run_agent(run_request(None, "x"))
        .await
        .expect("bulk run");

    let mut stream_server = mockito::Server::new_async().await;
    let _stream_mock = stream_server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(make_body())
        .create_async()
        .await;
    let stream_backend = OaiBackend::new(OaiConfig::for_testing(stream_server.url()));
    let stream = stream_backend
        .run_agent_streaming(run_request(None, "x"), NotificationSink::noop())
        .await
        .expect("stream run");

    assert_eq!(bulk.output, stream.output);
    assert_eq!(bulk.session_id, stream.session_id);
    assert_eq!(bulk.exit_code, stream.exit_code);
    assert_eq!(bulk.tool_calls, stream.tool_calls);
    assert_eq!(bulk.tool_results, stream.tool_results);
    assert_eq!(bulk.thinking, stream.thinking);
    assert_eq!(bulk.errors, stream.errors);
}

#[tokio::test]
async fn run_agent_streaming_emits_error_notification_on_malformed_chunk() {
    let mut server = mockito::Server::new_async().await;
    let body = "data: not-valid-json\n\ndata: [DONE]\n\n".to_string();
    let _mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
        .create_async()
        .await;

    let recorder: Arc<Mutex<Vec<AgentNotification>>> = Arc::new(Mutex::new(Vec::new()));
    let r2 = Arc::clone(&recorder);
    let sink = NotificationSink::new(move |n| r2.lock().unwrap().push(n));

    let backend = OaiBackend::new(OaiConfig::for_testing(server.url()));
    let response = backend
        .run_agent_streaming(run_request(Some("gpt-5"), "hi"), sink)
        .await
        .expect("streaming run should succeed even with malformed chunk");

    assert_eq!(response.errors.len(), 1, "errors: {:?}", response.errors);
    let notifications = recorder.lock().unwrap().clone();
    let saw_error = notifications
        .iter()
        .any(|n| matches!(n, AgentNotification::Error { .. }));
    assert!(
        saw_error,
        "no Error notification emitted: {notifications:?}"
    );
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
