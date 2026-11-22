"""Provider selection does not need a network or an API key."""

import json

import pytest

from labs._shared.llm import LocalFallbackClient, get_llm


def test_default_is_local_when_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = get_llm()
    assert isinstance(client, LocalFallbackClient)
    assert client.complete([{"role": "user", "content": "What is 2 + 2?"}]) == "4"


def test_forced_local_ignores_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    client = get_llm()
    assert client.name == "local-fallback"


def test_openai_without_package_or_key_is_explicit(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(Exception, match="OPENAI_API_KEY"):
        get_llm("openai")


def test_unknown_provider():
    with pytest.raises(Exception, match="Unknown"):
        get_llm("not-a-provider")


def test_huggingface_provider_runs_offline():
    client = get_llm("huggingface")
    assert client.name == "huggingface"
    text = client.complete([{"role": "user", "content": "hello"}])
    assert isinstance(text, str) and text


def test_tool_loop_is_multi_step():
    client = get_llm("local")
    first = json.loads(client.complete([{"role": "user", "content": "fee: wire calc: {fee}*2\nTOOL_LOOP"}], mode="tools"))
    assert first == {"action": "lookup_fee", "input": "wire"}
    transcript = (
        'fee: wire calc: {fee}*2\n'
        '{"action": "lookup_fee", "input": "wire"}\n'
        "Observation (lookup_fee): 15\nTOOL_LOOP"
    )
    second = json.loads(client.complete([{"role": "user", "content": transcript}], mode="tools"))
    assert second["action"] == "calculator"
    assert second["input"].replace(" ", "") == "15*2"


def test_json_repair_demo_is_deterministic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    client = get_llm()
    first = client.complete(
        [{"role": "user", "content": "repair-demo RETURN_JSON Paris"}],
        mode="json",
    )
    second = client.complete(
        [{"role": "user", "content": "repair-demo RETURN_JSON Paris"}],
        mode="json",
    )
    assert "country" not in first
    assert "France" in second
