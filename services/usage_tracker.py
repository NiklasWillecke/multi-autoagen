from dataclasses import dataclass
from typing import Any

from autogen_core.models import CreateResult

# USD per 1M tokens (Stand: OpenAI pricing für gpt-4o-mini / gpt-4o)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
}


@dataclass
class UsageSummary:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, result: CreateResult) -> None:
        self.prompt_tokens += result.usage.prompt_tokens
        self.completion_tokens += result.usage.completion_tokens
        self.llm_calls += 1

    def estimated_cost_usd(self) -> float:
        pricing = MODEL_PRICING.get(self.model)
        if pricing is None:
            for key, value in MODEL_PRICING.items():
                if self.model.startswith(key):
                    pricing = value
                    break
        if pricing is None:
            pricing = MODEL_PRICING["gpt-4o-mini"]

        return (
            self.prompt_tokens / 1_000_000 * pricing["input"]
            + self.completion_tokens / 1_000_000 * pricing["output"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 6),
        }


class TrackingChatClient:
    """Wraps an OpenAI chat client and accumulates token usage per request."""

    def __init__(self, client: Any, usage: UsageSummary) -> None:
        self._client = client
        self._usage = usage

    async def create(self, *args: Any, **kwargs: Any) -> CreateResult:
        result = await self._client.create(*args, **kwargs)
        self._usage.record(result)
        return result

    async def close(self) -> None:
        await self._client.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
