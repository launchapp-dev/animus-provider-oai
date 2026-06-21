//! Translate a model-emitted tool call into an Animus approval decision via the
//! committed `animus agent approve-hook` verb, then map that decision onto an
//! allow/deny outcome for the OpenAI Chat Completions harness.
//!
//! This provider is a stateless Chat Completions client: it does NOT execute
//! tool calls (the host does). The gate point is therefore "before the provider
//! emits/records a tool call". A `deny` suppresses the `ToolCall` notification
//! and the `tool_calls` entry and instead surfaces a tool error back to the
//! model via `tool_results`.
//!
//! Design invariants:
//! - Gating only runs when approvals are enabled (ONLY when
//!   `extras.approvals == true`). The `runtime_contract.mcp.agent_id` pin is
//!   for scoping, not a gate trigger. Otherwise we pass through and never spawn
//!   the binary.
//! - A verdict of `allow` allows the tool call, honoring any `updated_input`
//!   the hook returned (a human/policy edit MUST be executed, not the original).
//! - A verdict of `deny` blocks it.
//! - ANY failure (verb missing, nonzero exit, unparseable stdout) FAILS CLOSED:
//!   the tool call is blocked. We NEVER default to allow.

use std::path::PathBuf;
use std::process::Stdio;

use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;

/// Everything the gate needs to reach the Animus approval core.
#[derive(Debug, Clone)]
pub struct GateContext {
    /// True when this run opted into human-in-the-loop approvals. When false,
    /// gating is a pass-through.
    pub approvals_enabled: bool,
    /// Agent profile id whose `approval_policy` governs the decision. Sourced
    /// from `runtime_contract.mcp.agent_id` (fallback `extras.agent_id`).
    pub agent_id: String,
    /// Project root passed to the verb (`--project-root`).
    pub project_root: PathBuf,
    /// Path/name of the `animus` binary (env `ANIMUS_BIN`, else `"animus"`).
    pub animus_bin: String,
    /// Human-escalation timeout in seconds passed to the verb.
    pub timeout_secs: u64,
}

/// Normalized verdict from the approve-hook verb.
///
/// `Allow` carries an optional `updated_input`: when a human (or policy) edits
/// the tool arguments before approving, the caller MUST execute the edited
/// input, not the original. Dropping it would let an unapproved input run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Allow { updated_input: Option<Value> },
    Deny,
}

/// Outcome of gating one tool call, returned to the caller.
///
/// - `Allow` carries the input to actually use (the hook's `updated_input` when
///   present, else the original arguments).
/// - `Deny` blocks the call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateOutcome {
    Allow { input: Value },
    Deny,
}

/// Gate one model-emitted tool call.
///
/// - Approvals disabled → allow with the original input (pass-through, never
///   spawns the binary).
/// - Approvals enabled → route through approve-hook; allow ONLY on an explicit
///   `allow` verdict, honoring any `updated_input` the hook returned. Any
///   deny/error fails CLOSED (returns `Deny`).
pub async fn gate_tool_call(ctx: &GateContext, tool_name: &str, input: &Value) -> GateOutcome {
    if !ctx.approvals_enabled {
        return GateOutcome::Allow {
            input: input.clone(),
        };
    }
    match run_approve_hook(ctx, tool_name, input).await {
        Ok(Verdict::Allow { updated_input }) => GateOutcome::Allow {
            input: updated_input.unwrap_or_else(|| input.clone()),
        },
        Ok(Verdict::Deny) => GateOutcome::Deny,
        Err(reason) => {
            tracing::warn!(
                tool = tool_name,
                agent_id = %ctx.agent_id,
                %reason,
                "approve-hook failed for tool call; failing closed (deny)"
            );
            GateOutcome::Deny
        }
    }
}

/// Shell out to `animus agent approve-hook --format generic`, writing the tool
/// call as JSON to stdin and parsing the decision from stdout.
async fn run_approve_hook(
    ctx: &GateContext,
    tool_name: &str,
    raw_input: &Value,
) -> Result<Verdict, String> {
    let stdin_payload = json!({
        "tool_name": tool_name,
        "input": raw_input,
    });
    let stdin_bytes = serde_json::to_vec(&stdin_payload)
        .map_err(|e| format!("failed to encode approve-hook stdin: {e}"))?;

    let mut cmd = tokio::process::Command::new(&ctx.animus_bin);
    cmd.arg("agent")
        .arg("approve-hook")
        .arg("--format")
        .arg("generic")
        .arg("--agent-id")
        .arg(&ctx.agent_id)
        .arg("--project-root")
        .arg(&ctx.project_root)
        .arg("--timeout-secs")
        .arg(ctx.timeout_secs.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Kill the hook if our future is dropped (turn cancel / timeout) so we
        // don't leak a hook process waiting on a human decision.
        .kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn `{}`: {e}", ctx.animus_bin))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&stdin_bytes)
            .await
            .map_err(|e| format!("failed to write approve-hook stdin: {e}"))?;
        // Close stdin so the verb's reader sees EOF.
        drop(stdin);
    }

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| format!("approve-hook wait failed: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "approve-hook exited with {:?}: {}",
            output.status.code(),
            stderr.trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_generic_decision(&stdout)
}

/// Parse the generic approve-hook stdout contract:
/// `{ "decision": "allow"|"deny", "reason": ..., "updated_input"? }`.
pub fn parse_generic_decision(stdout: &str) -> Result<Verdict, String> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Err("approve-hook produced empty stdout".to_string());
    }
    let value: Value = serde_json::from_str(trimmed)
        .map_err(|e| format!("approve-hook stdout was not JSON: {e}; raw={trimmed:?}"))?;
    match value.get("decision").and_then(Value::as_str) {
        Some("allow") => {
            // Honor a human/policy edit to the tool input, if present. Treat an
            // explicit `null` the same as an absent field (no edit) so we never
            // replace the model's original arguments with JSON null.
            let updated_input = value.get("updated_input").filter(|v| !v.is_null()).cloned();
            Ok(Verdict::Allow { updated_input })
        }
        Some("deny") => Ok(Verdict::Deny),
        other => Err(format!(
            "approve-hook stdout had unexpected decision {other:?}; raw={trimmed:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_allow_decision() {
        let v = parse_generic_decision(r#"{"decision":"allow","source":"policy","reason":""}"#);
        assert_eq!(
            v.unwrap(),
            Verdict::Allow {
                updated_input: None
            }
        );
    }

    #[test]
    fn parse_allow_carries_updated_input() {
        let v = parse_generic_decision(r#"{"decision":"allow","updated_input":{"path":"/safe"}}"#);
        assert_eq!(
            v.unwrap(),
            Verdict::Allow {
                updated_input: Some(serde_json::json!({"path": "/safe"}))
            }
        );
    }

    #[test]
    fn parse_allow_null_updated_input_is_no_edit() {
        let v = parse_generic_decision(r#"{"decision":"allow","updated_input":null}"#);
        assert_eq!(
            v.unwrap(),
            Verdict::Allow {
                updated_input: None
            }
        );
    }

    #[test]
    fn parse_deny_decision() {
        let v = parse_generic_decision(r#"{"decision":"deny","source":"policy","reason":"nope"}"#);
        assert_eq!(v.unwrap(), Verdict::Deny);
    }

    #[test]
    fn parse_empty_is_error() {
        assert!(parse_generic_decision("   ").is_err());
    }

    #[test]
    fn parse_garbage_is_error() {
        assert!(parse_generic_decision("not json").is_err());
        assert!(parse_generic_decision(r#"{"decision":"maybe"}"#).is_err());
        assert!(parse_generic_decision(r#"{"other":"allow"}"#).is_err());
    }

    #[tokio::test]
    async fn gate_tool_call_passes_through_when_disabled() {
        // Approvals off: must NOT spawn the (missing) binary; allows the call.
        let ctx = GateContext {
            approvals_enabled: false,
            agent_id: "swe".to_string(),
            project_root: PathBuf::from("/tmp"),
            animus_bin: "definitely-not-a-real-binary-xyz".to_string(),
            timeout_secs: 5,
        };
        let input = serde_json::json!({"path": "x"});
        // Allows with the original input unchanged; never spawns the binary.
        assert_eq!(
            gate_tool_call(&ctx, "read_file", &input).await,
            GateOutcome::Allow { input }
        );
    }

    #[tokio::test]
    async fn gate_tool_call_fails_closed_on_missing_binary() {
        // Approvals on but the binary is missing: the error path must DENY.
        let ctx = GateContext {
            approvals_enabled: true,
            agent_id: "swe".to_string(),
            project_root: PathBuf::from("/tmp"),
            animus_bin: "definitely-not-a-real-binary-xyz".to_string(),
            timeout_secs: 5,
        };
        assert_eq!(
            gate_tool_call(&ctx, "write_file", &serde_json::json!({"path": "x"})).await,
            GateOutcome::Deny
        );
    }
}
