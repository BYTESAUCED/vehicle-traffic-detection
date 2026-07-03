"""Project paths, class names, and device selection.

Everything downstream (CLI + Streamlit app) imports constants from here so the
model checkpoint, class list, and density thresholds stay consistent.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root is two levels up from this file (src/traffic_state/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "bmd45_subset"
TRAIN_ANNOTATIONS = DATASET_DIR / "train" / "_annotations.coco.json"
VALID_DIR = DATASET_DIR / "valid"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
THRESHOLDS_FILE = OUTPUT_DIR / "density_thresholds.json"

# Fine-tuned checkpoint produced by training_mps.py. Override with the
# TRAFFIC_STATE_CHECKPOINT environment variable if you move it.
DEFAULT_CHECKPOINT = PROJECT_ROOT / "rfdetr-nano-bmd45-finetune" / "checkpoint_best_ema.pth"
CHECKPOINT_PATH = Path(os.environ.get("TRAFFIC_STATE_CHECKPOINT", DEFAULT_CHECKPOINT))

# BMD-45 vehicle categories, ordered by COCO category id. Fine-tuned RF-DETR
# remaps category ids to 0-based contiguous indices, so this list is indexable
# directly by the predicted class_id (0-12 respectively).
CLASS_NAMES = [
    "Hatchback",
    "Sedan",
    "SUV",
    "MUV",
    "Bus",
    "Truck",
    "Three-wheeler",
    "Two-wheeler",
    "LCV",
    "Mini-bus",
    "Tempo-traveller",
    "Bicycle",
    "Van",
]
ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}

DEFAULT_THRESHOLD = 0.25
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Per-image confidence filtering. Inference always runs once at the low base
# threshold to catch every candidate box, then boxes are kept per image.
#   "fixed":    keep boxes with confidence >= a single global threshold.
#   "adaptive": keep boxes with confidence >= max(base, image_max - margin).
#   "otsu":     split each image's confidence distribution into noise/signal
#               clusters with Otsu's method and keep the signal cluster, with a
#               safety floor so the bar never drops below the base reliability.
CONFIDENCE_METHODS = ("fixed", "adaptive", "otsu")

# Non-fixed methods run a single forward pass at this low base threshold to
# gather every candidate box; all further filtering is post-processing.
BASE_INFERENCE_THRESHOLD = 0.05

ADAPTIVE_BASE_THRESHOLD = 0.10
ADAPTIVE_MARGIN = 0.15

# Otsu method: split each image's confidence scores into noise/signal clusters.
# When an image has fewer than OTSU_MIN_DETECTIONS boxes, Otsu cannot find a
# meaningful split, so keep the top OTSU_FALLBACK_TOP_FRACTION by confidence
# instead. A hard floor is always enforced so very uncertain boxes are dropped.
OTSU_FLOOR = 0.10
OTSU_BINS = 32
OTSU_MIN_DETECTIONS = 5
OTSU_FALLBACK_TOP_FRACTION = 0.35

# Density label cutoffs, calibrated to the training histogram (all vehicles).
# Counts 1..DENSITY_LOW_MAX are low, DENSITY_LOW_MAX+1..DENSITY_MEDIUM_MAX are
# medium, and anything above is high. DENSITY_MEDIUM_MAX uses the natural trough
# at 11 rather than Q3=14, which would over-inflate the medium class.
DENSITY_LOW_MAX = 5
DENSITY_MEDIUM_MAX = 11

# Cap only the displayed vehicle count so outlier frames (the sparse tail up to
# 45+) do not distort the UI or aggregate stats. The density label is always
# computed from the true, uncapped count.
DISPLAY_COUNT_CAP = 22


def get_device() -> str:
    """Pick the best available backend: Apple MPS, then CUDA, then CPU."""
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
