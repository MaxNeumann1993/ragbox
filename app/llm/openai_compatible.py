"""LLM provider for any OpenAI-compatible endpoint.

This covers a hosted cloud API and a self-hosted vLLM server the same way:
vLLM's OpenAI-compatible server implements the same /chat/completions
contract, so only `base_url` and `model` change between the two. Swapping
providers is a config change, not a code change.
"""

from openai import OpenAI

from app.llm.base import LLMProvider

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below. If the context does not contain the answer, "
    "say clearly that you don't know based on the available documents. "
    "Never rely on outside knowledge, and never make up a source."
)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._model = model

    def answer(self, question: str, context: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
