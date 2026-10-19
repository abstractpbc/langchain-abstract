# langchain-abstract

LangChain chat model for [Abstract](https://abstract.inc).

`ChatAbstract` serves `glm-5.3:superfast`, `deepseek-v4-flash:superfast` and
`kimi-k3:superfast` through Abstract's OpenAI Responses-compatible API at
`https://api.abstract.inc/v1`. It is `ChatOpenAI` with Abstract's base URL and
key, and the Responses API turned on.

## Install

```bash
pip install -U langchain-abstract
```

Create an API key with `abstract login` or at
<https://platform.abstract.inc/keys>, then set it:

```bash
export ABSTRACT_API_KEY="your-api-key"
```

`ChatAbstract` reads only `ABSTRACT_API_KEY` (or `api_key=`). It never falls
back to `OPENAI_API_KEY`.

## Use

```python
from langchain_abstract import ChatAbstract

model = ChatAbstract(model="glm-5.3:superfast")

print(model.invoke("What is the capital of France?").text)

for chunk in model.stream("Count from 1 to 5."):
    print(chunk.text, end="")
```

Tool calling works through `bind_tools`, as with `ChatOpenAI`.

## Models

| Model ID                      | Input | Max output tokens |
| ----------------------------- | ----- | ----------------- |
| `glm-5.3:superfast`           | Text  | 65,536            |
| `deepseek-v4-flash:superfast` | Text  | 65,536            |
| `kimi-k3:superfast`           | Text  | 65,536            |

Prices: <https://abstract.inc/pricing>. API docs: <https://abstract.inc/docs>.

## Development

```bash
uv sync
uv run pytest tests/unit_tests
ABSTRACT_API_KEY=... uv run pytest tests/integration_tests  # live, spends credit
```
