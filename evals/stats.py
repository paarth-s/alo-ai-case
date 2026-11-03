"""Small statistics helpers shared by the eval scripts."""
import math
import random
import statistics


def bootstrap_ci(values: list[float], n: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean: resample the questions with replacement."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(n * alpha / 2)], means[int(n * (1 - alpha / 2)) - 1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(values: list[float], q: float) -> float:
    """q-th percentile (0-100) by nearest rank."""
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))]
