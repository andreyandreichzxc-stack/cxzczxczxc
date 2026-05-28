from src.core.actions.mcp_network import _validate_public_host


def test_network_guard_blocks_localhost() -> None:
    result = _validate_public_host("localhost")

    assert result is not None
    assert "Refusing to probe" in result["error"]


def test_network_guard_blocks_private_ip() -> None:
    result = _validate_public_host("192.168.1.1")

    assert result is not None
    assert "Refusing to probe" in result["error"]


def test_network_guard_allows_public_literal_ip() -> None:
    result = _validate_public_host("8.8.8.8")

    assert result is None
