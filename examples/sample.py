"""Small source file used by the rendering and OCR examples."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Reading:
    sensor: str
    value: float
    unit: str = "nm"


def summarize(readings: list[Reading]) -> dict[str, float]:
    """Return the average value for each sensor."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for reading in readings:
        totals[reading.sensor] = totals.get(reading.sensor, 0.0) + reading.value
        counts[reading.sensor] = counts.get(reading.sensor, 0) + 1
    return {name: total / counts[name] for name, total in totals.items()}


if __name__ == "__main__":
    sample = [Reading("red", 632.8), Reading("red", 633.1), Reading("green", 532.0)]
    print(summarize(sample))

