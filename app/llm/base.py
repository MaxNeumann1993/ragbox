"""The one abstraction that decouples this app from any specific LLM.

Anything that can answer a question given a context block (a cloud API or a
locally hosted vLLM server) implements this interface. `rag.py` never knows
or cares which one it's talking to.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def answer(self, question: str, context: str) -> str:
        """Answer the question using ONLY the given context.

        Implementations must be instructed to say when the context is not
        sufficient, rather than falling back on the model's own knowledge.
        """
        ...
