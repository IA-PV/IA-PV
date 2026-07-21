"""Small dependency-light statistical helpers used by reports and charts."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from statistics import fmean, median, stdev


# Two-sided 95% Student-t critical values for 1..30 degrees of freedom.
# Larger samples use a conservative value from the nearest lower tabulated df.
_T_CRITICAL_95 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def confidence_interval_95(
    values: Sequence[float],
) -> tuple[float | None, float | None]:
    """Return a two-sided 95% Student-t interval for the sample mean."""

    data = tuple(float(value) for value in values)
    if len(data) < 2:
        return None, None
    mean = fmean(data)
    return confidence_interval_95_from_summary(mean, stdev(data), len(data))


def confidence_interval_95_from_summary(
    mean: float,
    sample_stddev: float,
    count: int,
) -> tuple[float | None, float | None]:
    """Return the same interval when only mean, sample SD and n are retained."""

    if count < 2:
        return None, None
    standard_error = float(sample_stddev) / sqrt(count)
    critical = _student_t_critical_95(count - 1)
    half_width = critical * standard_error
    return float(mean) - half_width, float(mean) + half_width


def descriptive_statistics(values: Sequence[float]) -> dict[str, object]:
    """Return report-safe descriptive statistics and uncertainty for a sample."""

    data = tuple(float(value) for value in values)
    if not data:
        raise ValueError("At least one value is required.")
    sample_stddev = stdev(data) if len(data) > 1 else None
    standard_error = sample_stddev / sqrt(len(data)) if sample_stddev is not None else None
    low, high = confidence_interval_95(data)
    confidence_interval = (
        {
            "level": 0.95,
            "method": "student_t",
            "low": low,
            "high": high,
        }
        if low is not None and high is not None
        else None
    )
    return {
        "count": len(data),
        "mean": fmean(data),
        "sample_stddev": sample_stddev,
        "standard_error": standard_error,
        "confidence_interval_95": confidence_interval,
        "median": median(data),
        "min": min(data),
        "max": max(data),
    }


def _student_t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive.")
    if degrees_of_freedom <= 30:
        return _T_CRITICAL_95[degrees_of_freedom]
    if degrees_of_freedom <= 40:
        return _T_CRITICAL_95[30]
    if degrees_of_freedom <= 60:
        return 2.021
    if degrees_of_freedom <= 120:
        return 2.000
    return 1.980 if degrees_of_freedom <= 240 else 1.960


__all__ = [
    "confidence_interval_95",
    "confidence_interval_95_from_summary",
    "descriptive_statistics",
]
