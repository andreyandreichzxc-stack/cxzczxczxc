from src.core.context.runtime_bundle import build_runtime_context
from src.core.context.spec import ContextChunk


def test_runtime_context_bundle_deduplicates_chunks_by_source_and_text() -> None:
    bundle = build_runtime_context(
        memory_context="- stable fact",
        self_profile="[self] concise",
        contact_context="[contact] prefers voice",
        chunks=[
            ContextChunk(source="memory", text="Likes coffee", reason="recall"),
            ContextChunk(source="memory", text="  likes   coffee  ", reason="vector"),
            ContextChunk(source="wiki", text="Project Alpha", reason="context"),
        ],
    )

    assert len(bundle.chunks) == 2
    assert "- stable fact" in bundle.memory_context
    assert "[memory:recall] Likes coffee" in bundle.memory_context
    assert "[wiki:context] Project Alpha" in bundle.memory_context
    assert bundle.self_profile == "[self] concise"
    assert "contact_context" in bundle.source_trace
    assert "self_profile" in bundle.source_trace
