# animus-provider-oai

An OpenAI-compatible API provider plugin for [Animus](https://github.com/launchapp-dev/animus-cli).

> **Status:** Under construction — landing in Animus v0.4.x. This crate currently lives in the Animus core workspace at `crates/animus-provider-oai/`; v0.4.x extracts it to this standalone repository.

## What this is

Animus v0.4.0 makes providers (LLM backends) pluggable. This repository will ship `animus-provider-oai`, a stdio plugin that talks directly to OpenAI-compatible HTTP APIs (the `/v1/chat/completions` shape) — useful for OpenAI itself, Together AI, Anyscale, on-prem inference servers, vLLM deployments, OpenRouter, and any other OpenAI-shaped backend.

Unlike CLI-wrapper providers (`animus-provider-claude`, `animus-provider-codex`, etc.) which spawn an external CLI process per phase, `animus-provider-oai` makes HTTP calls directly. Useful for backends that don't have a first-class CLI.

## Install (planned)

```bash
animus plugin install animus-provider-oai
```

## Workflow YAML usage

```yaml
agents:
  custom-llm:
    model: my-on-prem-llama
    tool: oai
    mcp_servers: ["animus"]
```

Configure the API endpoint and auth via env vars (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, etc.) or per-agent overrides.

## Roadmap

- [ ] Extract from Animus core workspace at v0.4.x cut
- [ ] Publish `animus-provider-oai` crate to crates.io
- [ ] Release binaries on tag
- [ ] Independent semver track
- [ ] Configurable base URL + auth header per-agent
- [ ] CI exercises the contract test from `animus-protocol`

## Design

- **Protocol:** [`animus-plugin-protocol`](https://github.com/launchapp-dev/animus-protocol) (provider variant)
- **Naming:** repo, crate, and binary all named `animus-provider-oai`
- **Core repo:** [Animus](https://github.com/launchapp-dev/animus-cli)

## License

MIT — see [LICENSE](LICENSE).
