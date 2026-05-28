import logging

from src.core.observability.response_trace import log_response_trace


def test_response_trace_redacts_secrets_and_counts_memory(caplog) -> None:
    caplog.set_level(logging.INFO, logger="src.core.observability.response_trace")

    log_response_trace(
        route="maestro",
        owner_id=123,
        memory_context="- [recall_context] loves coffee\nplain line",
        tools_proposed=[{"tool": "mcp_system"}],
        tools_executed=["mcp_system"],
        tools_blocked=[],
        guardrail_decision={"risk": "low", "api_token": "super-secret"},
        humanizer_mode="fix",
        humanizer_changed=True,
        extra={"nested": {"password": "hidden"}},
    )

    payload = caplog.records[0].response_trace

    assert payload["route"] == "maestro"
    assert payload["memory_facts_count"] == 1
    assert payload["tools_proposed"] == ["mcp_system"]
    assert payload["humanizer"] == {"mode": "fix", "changed": True}
    assert payload["guardrail_decision"]["api_token"] == "***"
    assert payload["extra"]["nested"]["password"] == "***"
