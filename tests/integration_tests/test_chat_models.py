"""Live tests for ChatAbstract. They need ABSTRACT_API_KEY and spend a little credit."""

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_abstract import ChatAbstract


class TestChatAbstractStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatAbstract]:
        return ChatAbstract

    @property
    def chat_model_params(self) -> dict:
        return {"model": "deepseek-v4-flash:superfast", "max_tokens": 2048}

    @property
    def supports_json_mode(self) -> bool:
        return False

    @property
    def has_structured_output(self) -> bool:
        return False

    @property
    def supports_image_inputs(self) -> bool:
        return False
