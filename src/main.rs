use animus_plugin_protocol::{PluginInfo, PLUGIN_KIND_PROVIDER};
use animus_plugin_runtime::provider_main_with_capabilities;
use animus_provider_oai::backend::OaiBackend;
use animus_provider_oai::config::OaiConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    emit_manifest_if_requested();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let config = OaiConfig::from_env()?;
    let backend = OaiBackend::new(config);

    let info = PluginInfo {
        name: env!("CARGO_PKG_NAME").into(),
        version: env!("CARGO_PKG_VERSION").into(),
        plugin_kind: PLUGIN_KIND_PROVIDER.into(),
        description: Some(env!("CARGO_PKG_DESCRIPTION").into()),
    };

    // oai is a thin wrapper over the stateless OpenAI Chat Completions API.
    // The host (daemon) owns tool execution and follow-up turns, so this
    // plugin emits exactly one ToolCall per turn and never a ToolResult.
    // Advertising `$harness/oai-style` lets the testkit gate the
    // `tool-call-*-oai` conformance scenarios on this plugin and skip the
    // generic scenarios that assert a ToolResult notification.
    //
    // We deliberately do NOT advertise `$harness/cancellation-loop-v2`:
    // OAI completions are synchronous HTTP — cancel_agent returns an error
    // ("oai: cancel not supported (synchronous HTTP API)") and there is no
    // mid-flight subprocess to terminate.
    let extra_capabilities = vec!["$harness/oai-style".to_string()];

    provider_main_with_capabilities(info, backend, extra_capabilities).await
}

fn emit_manifest_if_requested() {
    if !std::env::args()
        .skip(1)
        .any(|arg| arg == "--manifest" || arg == "-m")
    {
        return;
    }

    let manifest = serde_json::json!({
        "name": env!("CARGO_PKG_NAME"),
        "version": env!("CARGO_PKG_VERSION"),
        "plugin_kind": "provider",
        "description": env!("CARGO_PKG_DESCRIPTION"),
        "protocol_version": animus_plugin_protocol::PROTOCOL_VERSION,
        "capabilities": [
            "agent/run",
            "health/check",
            "$harness/oai-style"
        ],
        "env_required": [
            {
                "name": "OPENAI_API_KEY",
                "description": "OpenAI-compatible API key required for agent run/resume calls.",
                "sensitive": true,
                "required": true
            },
            {
                "name": "OPENAI_BASE_URL",
                "description": "Override the OpenAI-compatible API base URL.",
                "required": false
            },
            {
                "name": "OPENAI_ORG",
                "description": "OpenAI organization identifier.",
                "required": false
            },
            {
                "name": "OPENAI_DEFAULT_MODEL",
                "description": "Fallback model used when the request omits a model.",
                "required": false
            }
        ]
    });
    println!(
        "{}",
        serde_json::to_string(&manifest).expect("serialize manifest")
    );
    std::process::exit(0);
}
