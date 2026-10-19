"""Unit tests for ChatAbstract."""

import pytest
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_abstract import ChatAbstract
from langchain_abstract.chat_models import DEFAULT_API_BASE

MODEL = "glm-5.3:superfast"


class TestChatAbstractStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatAbstract]:
        return ChatAbstract

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL, "api_key": "test-key"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "ABSTRACT_API_KEY": "env-key",
                "ABSTRACT_API_BASE": "https://example.test/v1",
            },
            {"model": MODEL},
            {"openai_api_key": "env-key", "openai_api_base": "https://example.test/v1"},
        )


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ABSTRACT_API_BASE", raising=False)
    model = ChatAbstract(model=MODEL, api_key="test-key")
    assert model.openai_api_base == DEFAULT_API_BASE
    assert model.use_responses_api is True
    assert model._llm_type == "abstract-chat"
    assert model._get_ls_params()["ls_provider"] == "abstract"


def test_missing_key_never_falls_back_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ABSTRACT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    with pytest.raises(ValueError, match="ABSTRACT_API_KEY"):
        ChatAbstract(model=MODEL)
