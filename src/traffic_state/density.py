"""Roadway-density labelling from vehicle counts.

The density label uses integer cutoffs calibrated to the vehicles-per-image
histogram of the BMD-45 training subset:

    count == 0                    -> "unclear"  (no confident detections)
    1 <= count <= low_max         -> "low"
    low_max < count <= medium_max -> "medium"
    count > medium_max            -> "high"

``low_max`` is the low/medium boundary (Q1 of the distribution) and
``medium_max`` is the medium/high boundary (the natural trough in the
distribution, which fits the data better than Q3).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from traffic_state.config import (
    DENSITY_LOW_MAX,
    DENSITY_MEDIUM_MAX,
    THRESHOLDS_FILE,
    TRAIN_ANNOTATIONS,
)


@dataclass(frozen=True)
class DensityThresholds:
    """Density cutoffs plus the distribution stats used to derive them."""

    low_max: int = DENSITY_LOW_MAX
    medium_max: int = DENSITY_MEDIUM_MAX
    median: float = 0.0
    mean: float = 0.0
    min: int = 0
    max: int = 0
    p99: float = 0.0
    n_images: int = 0
    source: str = "calibrated"

    def to_json(self, path: Path = THRESHOLDS_FILE) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy's default)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return float(sorted_values[low] * (1 - frac) + sorted_values[high] * frac)


def compute_thresholds_from_coco(annotations_path: Path = TRAIN_ANNOTATIONS) -> DensityThresholds:
    """Derive distribution stats from a COCO file, with calibrated cutoffs.

    The low/medium and medium/high cutoffs are fixed calibrated constants; only
    the descriptive stats (median, mean, percentiles) are read from the data.
    """
    if not Path(annotations_path).exists():
        return DensityThresholds(source="calibrated (dataset not found)")

    coco = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    counts_by_image: dict[int, int] = defaultdict(int)
    for ann in coco.get("annotations", []):
        counts_by_image[ann["image_id"]] += 1

    # Images with zero annotations still count as 0 vehicles.
    counts = [counts_by_image[img["id"]] for img in coco.get("images", [])]
    if not counts:
        return DensityThresholds(source="calibrated (no annotations)")

    counts.sort()
    n = len(counts)
    return DensityThresholds(
        median=_percentile(counts, 50),
        mean=sum(counts) / n,
        min=counts[0],
        max=counts[-1],
        p99=_percentile(counts, 99),
        n_images=n,
        source=str(annotations_path),
    )


def load_thresholds() -> DensityThresholds:
    """Load cached thresholds if present, else derive (and cache) from the dataset."""
    if THRESHOLDS_FILE.exists():
        try:
            data = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
            return DensityThresholds(**data)
        except (TypeError, ValueError, json.JSONDecodeError):
            # Stale cache from an older schema; recompute below.
            pass
    thresholds = compute_thresholds_from_coco()
    try:
        thresholds.to_json()
    except OSError:
        pass  # read-only environment; thresholds still usable in memory
    return thresholds


def density_label(
    vehicle_count: int,
    thresholds: DensityThresholds | None = None,
    min_confident_detections: int = 1,
) -> str:
    """Map a vehicle count to a roadway-density label.

    Args:
        vehicle_count: number of confident detections in the image.
        thresholds: cutoffs; loaded from the dataset when omitted.
        min_confident_detections: below this count the frame is "unclear".

    Returns:
        One of "unclear", "low", "medium", "high".
    """
    if thresholds is None:
        thresholds = load_thresholds()
    if vehicle_count < min_confident_detections:
        return "unclear"
    if vehicle_count <= thresholds.low_max:
        return "low"
    if vehicle_count <= thresholds.medium_max:
        return "medium"
    return "high"
