"""RF-DETR-Nano inference wrapper for BMD-45 vehicle detection.

Loads the fine-tuned checkpoint once and exposes:
- run_on_image: detect vehicles in a single image (path / PIL / ndarray),
- run_on_folder: iterate a directory of images,
- annotate: draw boxes + labels on a copy of the image.

The returned :class:`ImageResult` is a plain dataclass (no torch / supervision
objects), so it is safe to cache with ``st.cache_data`` and to serialise.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from traffic_state.config import (
    ADAPTIVE_BASE_THRESHOLD,
    ADAPTIVE_MARGIN,
    BASE_INFERENCE_THRESHOLD,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DEFAULT_THRESHOLD,
    DISPLAY_COUNT_CAP,
    ID2LABEL,
    IMAGE_EXTENSIONS,
    OTSU_BINS,
    OTSU_FLOOR,
    OTSU_FALLBACK_TOP_FRACTION,
    OTSU_MIN_DETECTIONS,
    get_device,
)
from traffic_state.density import DensityThresholds, density_label, load_thresholds


@dataclass
class Detection:
    """A single detected vehicle."""

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


@dataclass
class ImageResult:
    """Detections + traffic-state summary for one image."""

    file_name: str
    width: int
    height: int
    threshold: float
    detections: list[Detection] = field(default_factory=list)
    thresholds: DensityThresholds | None = None
    method: str = "fixed"
    applied_threshold: float = 0.0

    @property
    def vehicle_count(self) -> int:
        return len(self.detections)

    @property
    def display_count(self) -> int:
        """Vehicle count capped for display so outliers do not distort the UI.

        The density label always uses the true (uncapped) ``vehicle_count``.
        """
        return min(self.vehicle_count, DISPLAY_COUNT_CAP)

    @property
    def class_counts(self) -> dict[str, int]:
        counts = Counter(d.class_name for d in self.detections)
        # Stable ordering by the canonical class list, only non-zero classes.
        return {name: counts[name] for name in CLASS_NAMES if counts[name] > 0}

    @property
    def density(self) -> str:
        return density_label(self.vehicle_count, self.thresholds)

    def to_row(self) -> dict:
        """Flat dict for CSV/JSON export."""
        return {
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "vehicle_count": self.vehicle_count,
            "display_count": self.display_count,
            "density": self.density,
            "method": self.method,
            "applied_threshold": round(self.applied_threshold, 4),
            "class_counts": dict(self.class_counts),
            "detections": [
                {
                    "class_name": d.class_name,
                    "class_id": d.class_id,
                    "confidence": round(d.confidence, 4),
                    "xyxy": [round(v, 2) for v in d.xyxy],
                }
                for d in self.detections
            ],
        }


@lru_cache(maxsize=1)
def load_detector(checkpoint: str | None = None):
    """Load the fine-tuned RF-DETR-Nano model once (cached process-wide)."""
    from rfdetr import RFDETRNano

    ckpt = Path(checkpoint) if checkpoint else CHECKPOINT_PATH
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt}. Train first with training_mps.py "
            "or set TRAFFIC_STATE_CHECKPOINT to a valid .pth file."
        )
    device = get_device()
    return RFDETRNano(pretrain_weights=str(ckpt), device=device)


def _load_image(image) -> tuple[Image.Image, str]:
    """Normalise input into an RGB PIL image and a display name."""
    if isinstance(image, (str, os.PathLike)):
        name = os.path.basename(str(image))
        return Image.open(image).convert("RGB"), name
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB"), "array"
    if isinstance(image, Image.Image):
        return image.convert("RGB"), getattr(image, "filename", "image") or "image"
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def _adaptive_acceptance(confidences: list[float], base_threshold: float, margin: float) -> float:
    """Return the per-image acceptance threshold for the adaptive rule.

    The bar is ``max(base_threshold, image_max_confidence - margin)``: it drops
    toward the base for images whose best detection is weak, and rises for clean
    images so low-confidence noise is filtered out.
    """
    if not confidences:
        return base_threshold
    return max(base_threshold, max(confidences) - margin)


def _top_fraction_threshold(confidences: list[float], fraction: float, floor: float) -> float:
    """Return a threshold that keeps the top ``fraction`` of boxes by confidence.

    Used as the Otsu fallback for images with too few detections to split.
    """
    if not confidences:
        return floor
    keep = max(1, math.ceil(fraction * len(confidences)))
    kth_highest = sorted(confidences, reverse=True)[keep - 1]
    return max(kth_highest, floor)


def _otsu_acceptance(
    confidences: list[float],
    floor: float = OTSU_FLOOR,
    bins: int = OTSU_BINS,
    min_detections: int = OTSU_MIN_DETECTIONS,
    fallback_fraction: float = OTSU_FALLBACK_TOP_FRACTION,
) -> float:
    """Return the per-image acceptance threshold for the Otsu method.

    Otsu's method finds the value that maximises between-class variance between
    the low-confidence "noise" cluster and the high-confidence "signal" cluster.
    When fewer than ``min_detections`` boxes are present, Otsu cannot find a
    meaningful split, so the top ``fallback_fraction`` by confidence is kept
    instead. The result is always clamped to ``floor``.
    """
    if len(confidences) < min_detections or len(set(confidences)) < 2:
        return _top_fraction_threshold(confidences, fallback_fraction, floor)

    hist, edges = np.histogram(confidences, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return floor

    total_mean = float((centers * hist).sum())
    weight_bg = 0.0
    sum_bg = 0.0
    best_variance = -1.0
    best_threshold = floor
    for i in range(bins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += centers[i] * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (total_mean - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance >= best_variance:
            best_variance = variance
            best_threshold = float(centers[i])

    return max(best_threshold, floor)


def run_on_image(
    image,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    method: str = "fixed",
    base_threshold: float = ADAPTIVE_BASE_THRESHOLD,
    margin: float = ADAPTIVE_MARGIN,
    otsu_floor: float = OTSU_FLOOR,
    detector=None,
    thresholds: DensityThresholds | None = None,
    file_name: str | None = None,
) -> ImageResult:
    """Run detection on a single image and return an :class:`ImageResult`.

    Args:
        image: image path, PIL image, or ndarray.
        threshold: fixed confidence threshold, used when ``method`` is "fixed".
        method: one of "fixed", "adaptive", or "otsu". For "adaptive" and "otsu",
            inference runs once at the low base inference threshold to gather all
            candidate boxes, then a per-image acceptance bar is applied.
        base_threshold: acceptance floor for the "adaptive" method.
        margin: margin below the image max confidence for the "adaptive" method.
        otsu_floor: safety floor for the "otsu" method.
        detector: optional preloaded detector.
        thresholds: optional density thresholds.
        file_name: optional display name override.
    """
    detector = detector or load_detector()
    thresholds = thresholds or load_thresholds()
    pil_image, name = _load_image(image)
    name = file_name or name

    # Fixed mode filters at its own threshold; the adaptive and otsu methods run
    # a single low-threshold pass to gather all candidates, then post-filter.
    predict_threshold = threshold if method == "fixed" else BASE_INFERENCE_THRESHOLD
    predictions = detector.predict(pil_image, threshold=predict_threshold)

    detections: list[Detection] = []
    for xyxy, class_id, confidence in zip(
        predictions.xyxy, predictions.class_id, predictions.confidence
    ):
        cid = int(class_id)
        detections.append(
            Detection(
                class_id=cid,
                class_name=ID2LABEL.get(cid, str(cid)),
                confidence=float(confidence),
                xyxy=tuple(round(float(v), 2) for v in xyxy),
            )
        )

    confidences = [d.confidence for d in detections]
    if method == "adaptive":
        applied_threshold = _adaptive_acceptance(confidences, base_threshold, margin)
    elif method == "otsu":
        applied_threshold = _otsu_acceptance(confidences, otsu_floor)
    else:
        applied_threshold = threshold
    detections = [d for d in detections if d.confidence >= applied_threshold]

    return ImageResult(
        file_name=name,
        width=pil_image.width,
        height=pil_image.height,
        threshold=threshold,
        detections=detections,
        thresholds=thresholds,
        method=method,
        applied_threshold=applied_threshold,
    )


def iter_image_paths(folder: str | os.PathLike) -> list[Path]:
    """Return sorted image paths in a folder (non-recursive)."""
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_on_folder(
    folder: str | os.PathLike,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    method: str = "fixed",
    base_threshold: float = ADAPTIVE_BASE_THRESHOLD,
    margin: float = ADAPTIVE_MARGIN,
    otsu_floor: float = OTSU_FLOOR,
    detector=None,
    thresholds: DensityThresholds | None = None,
) -> list[ImageResult]:
    """Run detection on every image in a folder."""
    detector = detector or load_detector()
    thresholds = thresholds or load_thresholds()
    results = []
    for path in iter_image_paths(folder):
        results.append(
            run_on_image(
                str(path),
                threshold=threshold,
                method=method,
                base_threshold=base_threshold,
                margin=margin,
                otsu_floor=otsu_floor,
                detector=detector,
                thresholds=thresholds,
            )
        )
    return results


# Distinct colours cycled per class id for drawing boxes.
_PALETTE = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#9A6324", "#800000",
]


def annotate(image, result: ImageResult) -> Image.Image:
    """Draw bounding boxes + labels onto a copy of the image."""
    pil_image, _ = _load_image(image)
    canvas = pil_image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:  # pragma: no cover - font always available in practice
        font = None

    for det in result.detections:
        x1, y1, x2, y2 = det.xyxy
        color = _PALETTE[det.class_id % len(_PALETTE)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{det.class_name} {det.confidence:.2f}"
        ty = max(0, y1 - 12)
        draw.text((x1 + 2, ty), label, fill=color, font=font)

    return canvas
