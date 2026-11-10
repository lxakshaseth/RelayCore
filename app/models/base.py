"""Abstract Base Class for Model Providers (Groq Client & Stand-in Simulator)."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Tuple
from app.models.schemas import ModelRequest, ModelResponse


class AbstractModelClient(ABC):
    """Common interface contract for real and simulated LLM backends."""

    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute a non-streaming completion call."""
        pass

    @abstractmethod
    async def stream_complete(
        self, request: ModelRequest
    ) -> AsyncIterator[Tuple[str, Optional[ModelResponse]]]:
        """Stream token chunks and yield the final ModelResponse upon completion."""
        pass
