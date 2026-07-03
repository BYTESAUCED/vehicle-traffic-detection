"""
Fine-tuning RF-DETR-Nano on BMD-45 (Bengaluru Mobility Dataset) — local/MPS script.

Plain-Python (non-notebook) version of training_colab.py:
  - notebook shell-magics (!pip / !uv) removed,
  - no pinned package versions,
  - Apple-Silicon MPS backend support (falls back to CUDA, then CPU).

Install dependencies once with uv (no version pins):

    uv pip install torch torchvision transformers "rfdetr[train,loggers]" \
        supervision matplotlib huggingface_hub numpy pillow datasets accelerate

reference for training setup:
https://github.com/qubvel/transformers-notebooks/blob/main/notebooks/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb

What this script does:
  1. Load a small BMD-45 subset (150 train, 30 validation images).
  2. Inspect the dataset visually with matplotlib.
  3. Fine-tune RF-DETR-Nano for a short, time-boxed adaptation run.
  4. Run inference on single images and folders, saving results to CSV/JSON.
  5. Convert detections into a low/medium/high/unclear density label.
"""

# Enable CPU fallback for any MPS-unsupported ops. Must be set before importing torch.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import shutil

import torch


# Device selection (MPS, CUDA, CPU)
def get_device():
    """Pick the best available backend: Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()
print(f"Using device: {DEVICE}")


# --- Dataset ----------------------------------------------------------------
# The dataset subset is obtained by cloning the prepared repository:
#   git clone https://huggingface.co/datasets/M20VJ/bmd45_subset
# It already contains the train/valid/test splits with images and COCO
# annotation files, so there is no download or build step here.
DATASET_DIR = "bmd45_subset"
# ---------------------------------------------------------------------------

if not os.path.isdir(os.path.join(DATASET_DIR, "train")):
    raise SystemExit(
        f"Dataset folder '{DATASET_DIR}' with a train/ split was not found.\n"
        "Clone the prepared subset first (uses Git-Xet for large files):\n"
        "  git clone https://huggingface.co/datasets/M20VJ/bmd45_subset"
    )
print(f"Using dataset: {DATASET_DIR}")


# --- Reproducibility --------------------------------------------------------
# Everything downstream is seeded from a single SEED so the run is reproducible:
# Python random, NumPy, and PyTorch (CPU + CUDA) are all seeded, and cuDNN is set
# to deterministic mode.
import random
import numpy as np

SEED = 42

def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

DATASET_DIR = "bmd45_subset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")
ANNOTATION_FILE = "_annotations.coco.json"

if not os.path.isdir(TEST_DIR):
    shutil.copytree(VALID_DIR, TEST_DIR)
    print(f"Created {TEST_DIR} as a copy of {VALID_DIR} (RF-DETR needs a test split).")
    print(f"Seed: {SEED}")
print(f"Dataset dir: {DATASET_DIR}  ->  train/ valid/ test/")


# --- Dataset inspection -----------------------------------------------------
# Before training, visually inspect the subset: sample frames with bounding boxes,
# class distribution, box-size distribution, and objects-per-image counts (which
# motivate the density thresholds used later).
from collections import defaultdict, Counter

def load_coco_split(split_dir):
    with open(os.path.join(split_dir, ANNOTATION_FILE)) as f:
        coco = json.load(f)
    id2label = {c["id"]: c["name"] for c in coco["categories"]}
    images = coco["images"]
    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)
    return coco, id2label, images, anns_by_image

train_coco, id2label, train_images, train_anns_by_image = load_coco_split(TRAIN_DIR)
val_coco, _, val_images, _ = load_coco_split(VALID_DIR)

print(f"Train subset: {len(train_images)} images")
print(f"Validation subset: {len(val_images)} images")
print(f"Categories ({len(id2label)}): {list(id2label.values())}")

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for a plain script
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

FIGURES_DIR = "outputs"
os.makedirs(FIGURES_DIR, exist_ok=True)
samples = random.sample(train_images, 9)
fig, axes = plt.subplots(3, 3, figsize=(13, 13))
for ax, im in zip(axes.flat, samples):
    img = Image.open(os.path.join(TRAIN_DIR, im["file_name"])).convert("RGB")
    boxes = [a["bbox"] for a in train_anns_by_image[im["id"]]]
    ax.imshow(img)
    for box in boxes:
        x, y, w, h = box
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1.5))
    ax.set_title(f"{len(boxes)} vehicles", fontsize=10)
    ax.axis("off")
plt.suptitle("BMD-45 Sample Frames with Vehicle Annotations", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "sample_frames.png"), dpi=150)
plt.close(fig)

class_counts = Counter()
for im in train_images:
    for a in train_anns_by_image[im["id"]]:
        class_counts[id2label[a["category_id"]]] += 1

plt.figure(figsize=(8, 5))
plt.bar(class_counts.keys(), class_counts.values(), color="steelblue")
plt.title("Vehicle Class Distribution (Train Subset)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
plt.close()

box_areas, objects_per_image = [], []
for im in train_images:
    boxes = [a["bbox"] for a in train_anns_by_image[im["id"]]]
    objects_per_image.append(len(boxes))
    for _, _, w, h in boxes:
        box_areas.append(w * h)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(box_areas, bins=20, color="coral")
axes[0].set_title("Bounding Box Area Distribution")
axes[0].set_xlabel("Area (px²)")
axes[1].hist(objects_per_image, bins=range(0, max(objects_per_image) + 2), color="mediumseagreen")
axes[1].set_title("Vehicles per Image")
axes[1].set_xlabel("Vehicle count")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "box_and_density_dist.png"), dpi=150)
plt.close(fig)


# --- Density threshold justification ---------------------------------------
# Thresholds are derived from the vehicle-count distribution of the training subset
# rather than fixed manually:
#   low:     fewer than Q1 vehicles detected
#   medium:  between Q1 and Q3 vehicles
#   high:    more than Q3 vehicles
#   unclear: zero confident detections
counts = np.array([len(train_anns_by_image[im["id"]]) for im in train_images])

print(f"Images: {len(train_images)}")
print(f"Annotations: {int(counts.sum())}")
print(f"Categories: {list(id2label.values())}")

median = np.median(counts)
q1, q3 = np.percentile(counts, [25, 75])
mean, std = counts.mean(), counts.std()
print(f"Vehicles/image -> min: {counts.min()}, max: {counts.max()}, "
      f"mean: {mean:.1f}, median: {median:.1f}, Q1: {q1:.1f}, Q3: {q3:.1f}")

def density_label(vehicle_count, q1=q1, q3=q3, min_confident_detections=1):
    if vehicle_count < min_confident_detections:
        return "unclear"
    elif vehicle_count < q1:
        return "low"
    elif vehicle_count <= q3:
        return "medium"
    else:
        return "high"

labels = [density_label(c) for c in counts]
label_counts = Counter(labels)
print(label_counts)

plt.figure(figsize=(8, 5))
plt.hist(counts, bins=range(0, counts.max() + 2), color="steelblue", edgecolor="white")
plt.axvline(q1, color="orange", linestyle="--", label=f"Q1 = {q1:.1f} (low/medium cutoff)")
plt.axvline(median, color="black", linestyle="-", label=f"Median = {median:.1f}")
plt.axvline(q3, color="red", linestyle="--", label=f"Q3 = {q3:.1f} (medium/high cutoff)")
plt.title("Vehicles per Image — Density Threshold Derivation")
plt.xlabel("Vehicle count")
plt.ylabel("Number of images")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "density_threshold.png"), dpi=150)
plt.close()

split_summary = {"Train": len(train_images), "Validation": len(val_images)}
plt.figure(figsize=(5, 4))
plt.bar(split_summary.keys(), split_summary.values(), color=["#4C72B0", "#DD8452"])
plt.title("Dataset Split Summary")
plt.ylabel("Image count")
plt.savefig(os.path.join(FIGURES_DIR, "split_summary.png"), dpi=150)
plt.close()


# --- Pre-training annotation preprocessing ----------------------------------
# Following the RT-DETRv2 reference notebook, "runaway" boxes that stretch beyond
# the image edge (and zero/negative-area boxes) will raise errors during training.
# The reference handles this with clip=True and min_area/min_width/min_height in
# Albumentations BboxParams. RF-DETR loads the COCO files directly from disk, so we
# apply the equivalent sanitization to the on-disk COCO files before training:
#   - clip every box to the image bounds [0, width] x [0, height],
#   - drop boxes whose clipped width/height is below MIN_BOX_SIDE or area below MIN_BOX_AREA,
#   - recompute the area field so it matches the clipped box.
MIN_BOX_AREA = 1.0   # drop boxes smaller than this many px^2 after clipping
MIN_BOX_SIDE = 1.0   # drop boxes whose clipped width/height is below this many px

def sanitize_coco_annotations(split_dir, min_area=MIN_BOX_AREA, min_side=MIN_BOX_SIDE):
    """Clip runaway boxes to image bounds and drop degenerate boxes in place."""
    ann_path = os.path.join(split_dir, ANNOTATION_FILE)
    with open(ann_path) as f:
        coco = json.load(f)

    image_wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}

    kept, clipped, dropped = [], 0, 0
    for a in coco["annotations"]:
        img_w, img_h = image_wh[a["image_id"]]
        x, y, w, h = a["bbox"]

        # convert to corners, clip to image bounds, convert back to [x, y, w, h]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
        new_w, new_h = x2 - x1, y2 - y1

        if new_w != w or new_h != h or x1 != x or y1 != y:
            clipped += 1

        # drop degenerate / runaway boxes that fell outside the image
        if new_w < min_side or new_h < min_side or (new_w * new_h) < min_area:
            dropped += 1
            continue

        a["bbox"] = [x1, y1, new_w, new_h]
        a["area"] = new_w * new_h
        kept.append(a)

    coco["annotations"] = kept
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    print(f"[sanitize] {split_dir}: kept {len(kept)}, clipped {clipped}, dropped {dropped}")

for _split_dir in (TRAIN_DIR, VALID_DIR, TEST_DIR):
    if os.path.isdir(_split_dir):
        sanitize_coco_annotations(_split_dir)


# --- 5. Fine-Tuning RF-DETR-Nano -------------------------------------------
# Training is short and time-boxed. All hyperparameters are defined as named
# variables in one config block so the exact run configuration is traceable.
OUTPUT_DIR = "rfdetr-nano-bmd45-finetune"

EPOCHS = 15                 # short adaptation run
BATCH_SIZE = 4              # per-step batch (small so it fits limited memory)
GRAD_ACCUM_STEPS = 4        # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 5e-5

print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, "
      f"batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} (effective {BATCH_SIZE * GRAD_ACCUM_STEPS})")

from rfdetr import RFDETRNano

seed_everything(SEED)

model = RFDETRNano(device=DEVICE)
model.train(
    dataset_dir=DATASET_DIR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    lr=LEARNING_RATE,
    output_dir=OUTPUT_DIR,
    device=DEVICE,
    # macOS uses the "spawn" start method for DataLoader workers. Since this is a
    # flat script (not guarded by `if __name__ == '__main__'`), spawned workers
    # re-import the module and crash ("bootstrapping phase" RuntimeError). Running
    # with 0 workers keeps data loading in the main process and avoids the spawn.
    num_workers=0,
)


# Evaluation: density-label agreement
# RF-DETR logs COCO-style mAP/mAR on the validation split every epoch during
# training. Here we add a task-level check: run the fine-tuned model over the
# validation images and compare predicted vs ground-truth density labels.
def find_best_checkpoint(output_dir):
    for name in ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth",
                 "checkpoint_best_total.pth", "checkpoint.pth"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            return path
    pths = [f for f in os.listdir(output_dir) if f.endswith(".pth")]
    if pths:
        return os.path.join(output_dir, sorted(pths)[0])
    raise FileNotFoundError(f"No .pth checkpoint found in {output_dir}")

BEST_CHECKPOINT = find_best_checkpoint(OUTPUT_DIR)
print(f"Loading fine-tuned weights: {BEST_CHECKPOINT}")

DETECTION_THRESHOLD = 0.4
detector = RFDETRNano(pretrain_weights=BEST_CHECKPOINT, device=DEVICE)

# ground-truth vehicle count -> density label, per validation image
gt_counts = {im["file_name"]: 0 for im in val_images}
_val_coco, _, _val_images, val_anns_by_image = load_coco_split(VALID_DIR)
for im in _val_images:
    gt_counts[im["file_name"]] = len(val_anns_by_image[im["id"]])

correct, total = 0, 0
for im in _val_images:
    image = Image.open(os.path.join(VALID_DIR, im["file_name"])).convert("RGB")
    detections = detector.predict(image, threshold=DETECTION_THRESHOLD)
    pred_label = density_label(len(detections))
    gt_label = density_label(gt_counts[im["file_name"]])
    correct += int(pred_label == gt_label)
    total += 1

print(f"Held-out density-label agreement: {correct}/{total} = {correct / total:.2%}")


# Inference pipeline
# Run the fine-tuned RF-DETR-Nano on a single image and on the validation folder,
# draw detections, and convert detected vehicle counts into a density label.
# Per-image results are saved to CSV and JSON.
import csv
from PIL import ImageDraw

print(f"Running inference on: {DEVICE}")

def run_inference(image_path, threshold=DETECTION_THRESHOLD):
    image = Image.open(image_path).convert("RGB")
    detections = detector.predict(image, threshold=threshold)
    return image, detections

def draw_detections(image, detections, id2label=id2label):
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    for xyxy, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
        x, y, x2, y2 = [round(float(v), 2) for v in xyxy]
        name = id2label.get(int(class_id), str(int(class_id)))
        draw.rectangle((x, y, x2, y2), outline="red", width=2)
        draw.text((x, y), f"{name} [{confidence:.2f}]", fill="blue")
    return image_with_boxes

# --- single image ----------------------------------------------------------
EXAMPLE_IMAGE_PATH = os.path.join(VALID_DIR, _val_images[0]["file_name"])
image, detections = run_inference(EXAMPLE_IMAGE_PATH)

for xyxy, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
    box = [round(float(v), 2) for v in xyxy]
    print(f"Detected {id2label.get(int(class_id), int(class_id))} "
          f"with confidence {round(float(confidence), 3)} at {box}")

os.makedirs("sample_detection_images", exist_ok=True)
annotated = draw_detections(image, detections)
annotated.save("sample_detection_images/example_annotated.jpg")

vehicle_count = len(detections)
print(f"\nExample image: {vehicle_count} vehicles -> density: {density_label(vehicle_count)}")

# --- folder inference -> CSV / JSON ----------------------------------------
results = []
for im in _val_images:
    path = os.path.join(VALID_DIR, im["file_name"])
    _, dets = run_inference(path)
    count = len(dets)
    results.append({
        "file_name": im["file_name"],
        "vehicle_count": count,
        "density": density_label(count),
    })

with open(os.path.join(FIGURES_DIR, "inference_results.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file_name", "vehicle_count", "density"])
    writer.writeheader()
    writer.writerows(results)

with open(os.path.join(FIGURES_DIR, "inference_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} inference rows to {FIGURES_DIR}/inference_results.csv / .json")
print("Density distribution:", Counter(r["density"] for r in results))
