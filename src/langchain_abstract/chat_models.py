"""Abstract chat models."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import LangSmithParams
from langchain_core.utils import from_env
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import Field, model_validator

DEFAULT_API_BASE = "https://api.abstract.inc/v1"


class ChatAbstract(ChatOpenAI):
    """Chat model served by [Abstract](https://abstract.inc).

    Abstract serves `glm-5.3:superfast`, `deepseek-v4-flash:superfast` and
    `kimi-k3:superfast` through an OpenAI Responses-compatible API. `ChatAbstract`
    is `ChatOpenAI` pointed at that API, with the Responses API turned on.

    Setup:
        Install `langchain-abstract` and set the `ABSTRACT_API_KEY` environment
        variable. Create a key with `abstract login` or at
        https://platform.abstract.inc/keys.

        ```bash
        pip install -U langchain-abstract
        export ABSTRACT_API_KEY="your-api-key"
        ```

    Key init args:
        model: Abstract model id, for example `glm-5.3:superfast`.
        api_key: Abstract API key. Read from `ABSTRACT_API_KEY` if not passed.
        base_url: API base URL. Read from `ABSTRACT_API_BASE` if not passed,
            otherwise `https://api.abstract.inc/v1`.
        max_tokens: Maximum number of output tokens (at most 65,536).

    Instantiate:
        ```python
        from langchain_abstract import ChatAbstract

        model = ChatAbstract(model="glm-5.3:superfast")
        ```

    Invoke:
        ```python
        model.invoke("What is the capital of France?")
        ```

    Stream:
        ```python
        for chunk in model.stream("Count from 1 to 5."):
            print(chunk.text, end="")
        ```
    """

    model_name: str = Field(alias="model")
    """Abstract model id, for example `glm-5.3:superfast`."""

    openai_api_base: str | None = Field(
        alias="base_url",
        default_factory=from_env("ABSTRACT_API_BASE", default=DEFAULT_API_BASE),
    )
    """Base URL of the Abstract API."""

    use_responses_api: bool | None = True
    """Abstract serves its language models through the Responses API."""

    @model_validator(mode="before")
    @classmethod
    def _require_abstract_api_key(cls, values: Any) -> Any:
        """Read the key from `ABSTRACT_API_KEY`, never from `OPENAI_API_KEY`."""
        if not isinstance(values, dict):
            return values
        if (
            values.get("api_key") is not None
            or values.get("openai_api_key") is not None
        ):
            return values
        api_key = os.environ.get("ABSTRACT_API_KEY")
        if not api_key:
            msg = (
                "Abstract API key not found. Set the ABSTRACT_API_KEY environment "
                "variable or pass api_key."
            )
            raise ValueError(msg)
        return {**values, "api_key": api_key}

    @property
    def _llm_type(self) -> str:
        return "abstract-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        # ChatOpenAI's secret mapping wins during serialization, so a loaded model
        # would read OPENAI_API_KEY and send it to Abstract. Refuse instead.
        return False

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "abstract"
        return params
