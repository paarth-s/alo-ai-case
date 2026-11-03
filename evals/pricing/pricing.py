"""Token prices in USD per million tokens (input, output). Source: Anthropic API reference, cached 2026-10-06.
Haiku 5.5 prices apply to prompts up to 100K tokens (higher beyond), which every call here is under."""
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}


def cost(model: str, in_tokens: int, out_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))  # unknown model -> $0, never a wrong guess
    return (in_tokens * pin + out_tokens * pout) / 1_000_000
