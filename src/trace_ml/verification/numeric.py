"""Numerically stable helpers for finite scalar metric evidence."""

from __future__ import annotations

import math
from numbers import Real


def finite_float(value) -> float | None:
    """Return a finite float representation, or None when unsupported."""

    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def stable_mean(values) -> float | None:
    """Compute a finite mean without overflowing sums or underflowing each term."""

    numbers = []
    for value in values:
        converted = finite_float(value)
        if converted is None:
            return None
        numbers.append(converted)

    if not numbers:
        return None

    scale = max(abs(value) for value in numbers)
    if scale == 0:
        return 0.0

    normalized_mean = math.fsum(value / scale for value in numbers) / len(numbers)
    result = scale * normalized_mean
    return result if math.isfinite(result) else None


def stable_population_std(values) -> float | None:
    """Compute population standard deviation with scale-normalized differences."""

    numbers = []
    for value in values:
        converted = finite_float(value)
        if converted is None:
            return None
        numbers.append(converted)

    if not numbers:
        return None
    if len(numbers) == 1:
        return 0.0

    scale = max(abs(value) for value in numbers)
    if scale == 0:
        return 0.0

    mean = stable_mean(numbers)
    if mean is None:
        return None

    normalized_mean = mean / scale
    normalized_std = math.hypot(
        *(value / scale - normalized_mean for value in numbers)
    ) / math.sqrt(len(numbers))
    result = scale * normalized_std
    return result if math.isfinite(result) else None


def stable_midpoint(first, second) -> float | None:
    """Return the mean of two finite values without midpoint overflow/underflow."""

    return stable_mean((first, second))


def within_tolerance(first, second, absolute_tolerance: float) -> bool:
    """Compare finite floats with a strict decimal tolerance plus float ULP slack."""

    left = finite_float(first)
    right = finite_float(second)
    if left is None or right is None:
        return False

    representation_slack = 2.0 * max(math.ulp(left), math.ulp(right))
    threshold = max(absolute_tolerance, representation_slack)
    return abs(left - right) < threshold
