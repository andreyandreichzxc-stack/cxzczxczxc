from src.core.actions.action_guard import guard_intent
from src.core.intelligence.context_compressor import compress_maestro_context
from src.core.intelligence.prompt_assembler import AssemblyContext, prompt_assembler


def test_action_guard_blocks_unknown_intent():
    result = guard_intent({"intent": "does_not_exist"})
    assert result.allowed is False
    assert "Неизвестное" in result.reason


def test_action_guard_sanitizes_extra_fields():
    result = guard_intent(
        {"intent": "chat", "reply": "ok", "unexpected": "drop me"}
    )
    assert result.allowed is True
    assert "unexpected" not in result.intent


def test_prompt_assembler_injects_skill_index():
    prompt = prompt_assembler.assemble(
        AssemblyContext(
            target="agent",
            user_id=1,
            skill_index="<skill_index>\n- test skill\n</skill_index>",
        )
    )
    assert "<skill_index>" in prompt
    assert "test skill" in prompt


def test_context_compressor_keeps_budget():
    history = "\n".join(f"turn {i}" for i in range(100))
    result = compress_maestro_context(history_block=history, budget_chars=500)
    assert len(result.compressed_context) <= 500
    assert result.dropped_sections

