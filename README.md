# BMD-45 Traffic-State Detector

Vehicle detection on traffic-camera images with a simple roadway-density label.
An RF-DETR-Nano model is fine-tuned on a small subset of the BMD-45 (Bengaluru
Mobility Dataset), then used to detect vehicles, count them by class, and assign
a density label of low, medium, high, or unclear. The project ships a reusable
library, a command-line pipeline, and a Streamlit app.

This is a reproducible pipeline demonstration. Model accuracy is not the goal;
a clean end-to-end flow (data, training, inference, density label, UI) is.


## Project layout

```
.
├── pyproject.toml                # uv-managed project and dependencies
├── README.md
├── training_mps.py               # fine-tune RF-DETR-Nano (MPS / CUDA / CPU)
├── src/traffic_state/            # reusable library
│   ├── config.py                 # paths, class names, device, defaults
│   ├── density.py                # density thresholds and labelling
│   ├── detector.py               # model loading, inference, annotation
│   └── pipeline.py               # CLI: image/folder to CSV/JSON
├── app/streamlit_app.py          # Streamlit UI
├── .streamlit/config.toml        # light theme
├── rfdetr-nano-bmd45-finetune/   # training outputs and checkpoints
└── outputs/                      # predictions, plots, sample results
```


## Setup

Clone the repo:
```
git clone https://github.com/BYTESAUCED/vehicle-traffic-detection.git
```

Install uv (see https://docs.astral.sh/uv/getting-started/installation/), then
create the environment from the project root:

```
cd vehicle-traffic-detection
uv venv

[windows]
 .venv\Scripts\activate

[macos]
source .venv/bin/activate
```

PyTorch is platform and hardware dependent (CPU, CUDA version, or Apple MPS), so
install it first. Check the official selector at
https://pytorch.org/get-started/locally/ and install the correct `torch` and
`torchvision` build for your operating system and GPU. For example, on a machine
with CUDA you install the matching CUDA wheels; on Apple Silicon the default
build already includes MPS support. If the wrong build is installed, the model
may fall back to CPU or fail to load.


```
#Example [windows cuda version 12.8]
 uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Example [macos mps+cpu backend] (see the selector for the command that matches your hardware)
uv pip install torch torchvision
```

Then install the project and its remaining dependencies:

```
uv pip install -e .
uv pip install "rfdetr[train,loggers]"
```

## Get the model weights and dataset

The fine-tuned model weights and the dataset (optional) subset are hosted on the Hugging
Face Hub, use Git-Xet and git-lfs for large files. Install git-xet first (see
https://hf.co/docs/hub/git-xet) and (see https://git-lfs.com/), then clone both repositories into the project
root:

```
[macOS (Homebrew)]
brew install git-xet git-lfs
git xet install
git lfs install

[windows]
winget install git-xet GitHub.GitLFS
```

Windows (Manual Installer, winget Doesn't work)

1. Download and run the setup from [git-lfs.com](https://git-lfs.com).
2. Run the initialization command:
```cmd
git lfs install
```


 Verify Installation
```bash
git lfs version
```

```
# Model weights required!
git clone https://huggingface.co/M20VJ/rfdetr-nano-bmd45-finetune

# Dataset subset 
git clone https://huggingface.co/datasets/M20VJ/bmd45_subset
```

After cloning, `rfdetr-nano-bmd45-finetune/checkpoint_best_ema.pth` is used for
inference by default, and `bmd45_subset/` provides the full train/valid/test
splits, including images and COCO annotation files. No extra download or build
step is needed.

## How to start the Streamlit app

```
uv run streamlit run app/streamlit_app.py
```

Open the local URL that is printed (default http://localhost:8501). The app
lets you:

- upload one or more images, or pick sample frames from the validation set;
- choose a confidence method (fixed, adaptive, or Otsu) and its parameters;
- set a batch chunk size (images are processed in chunks with a progress bar);
- view one image at a time using Previous and Next buttons;
- click a row in the summary table to open that image;
- for each image, view the original, the image with detections, the per-class
  counts, and the density label;
- see bar charts for the density distribution and total vehicles per class;
- rename the output CSV; predictions are auto-saved to
  `outputs/predictions/` and a notification confirms the save.

Before uploading a large batch of images, set the batch chunk size in the
sidebar. Images are processed in chunks of that size with a progress bar, which
keeps memory use bounded and gives incremental feedback. Use a smaller chunk
size (for example 4) on low-memory machines, and a larger one (for example 16 or
32) when you have more memory and want fewer progress updates. Per-image results
are cached, so navigating between images or changing the view does not re-run
the model.

## Dataset subset used

- Source: the iisc-aim/BMD-45 dataset on the Hugging Face Hub. It contains
  traffic-camera frames from Bengaluru with COCO-format annotations.
- Subset: 150 training images and 30 validation images, taken from the first
  image shard (images_000). A test split is created as a copy of the validation
  split because RF-DETR expects a train/valid/test structure.
- Classes (13): Hatchback, Sedan, SUV, MUV, Bus, Truck, Three-wheeler,
  Two-wheeler, LCV, Mini-bus, Tempo-traveller, Bicycle, Van.


## How frames were selected or extracted

The subset was built deterministically from the BMD-45 COCO split files:

1. Read the COCO split JSON files (train and validation).
2. Keep only images whose file name is in the images_000 shard.
3. Sort those images by numeric file name so the selection is reproducible.
4. Take the first N per split (150 train, 30 validation).
5. Keep only the raw image files for those selections, with matching COCO
   annotation files so the images on disk stay in sync.
6. Copy the validation split to a test split.

The result is published as `M20VJ/bmd45_subset` on the Hugging Face Hub, so you
just clone it (see the setup section). It already contains the images, the COCO
annotation files, and the train/valid/test structure:

```
bmd45_subset/
    train/   images + _annotations.coco.json
    valid/   images + _annotations.coco.json
    test/    images + _annotations.coco.json
```


## Model setup and training/adaptation performed

- Model: RF-DETR-Nano (a DINOv2 backbone with a DETR-style detection
  transformer), fine-tuned from Roboflow pretrained weights. The detection head
  is re-initialised to the 13 BMD-45 classes, detected automatically from the
  dataset.
- Preprocessing: before training, the COCO annotations are sanitised. Boxes are
  clipped to image bounds and degenerate or zero-area boxes are dropped. This
  mirrors the clip and min-area handling in the RT-DETRv2 reference. RF-DETR
  performs resize, normalisation, and augmentation internally.
- Configuration: 40 epochs, batch size 4 with gradient accumulation 4
  (effective batch 16), learning rate 5e-5, seed 42. DataLoader workers are set
  to 0 to avoid the macOS spawn start-method crash.

Run training with:

```
uv run python training_mps.py
```

Checkpoints and metrics are written to `rfdetr-nano-bmd45-finetune/`. The file
`checkpoint_best_ema.pth` is used for inference by default. Override it with the
`TRAFFIC_STATE_CHECKPOINT` environment variable.


## How to run inference

Run on a single image or a folder, and save predictions as CSV or JSON:

```
# Single image to JSON
uv run traffic-state --input bmd45_subset/valid/122.png --format json

# Folder to CSV, and also save annotated images
uv run traffic-state --input bmd45_subset/valid --format csv --save-annotated
```

Confidence options (see the next section for how these differ):

```
uv run traffic-state --input <path> \
    --method {fixed,adaptive,otsu} \
    --threshold 0.25 \
    --base-threshold 0.10 \
    --margin 0.15 \
    --otsu-floor 0.10 \
    --output-dir outputs \
    --format {csv,json} \
    --save-annotated \
    --checkpoint <path/to/checkpoint.pth>
```

Prediction files are written to `outputs/predictions_<name>.<format>`.

The library can also be called directly:

```
from traffic_state.detector import run_on_image

result = run_on_image("bmd45_subset/valid/122.png", method="otsu")
print(result.vehicle_count, result.density, result.class_counts)
```


## How the density label is computed

The label is derived from the count of detected vehicles (all vehicle classes)
using integer cutoffs calibrated to the training histogram:

```
count == 0          -> unclear   (no confident detections)
1 <= count <= 5     -> low
6 <= count <= 11    -> medium
count >= 12         -> high
```

The low/medium boundary at 5 is Q1 of the distribution. The medium/high boundary
at 11 is the natural trough in the distribution, which fits the data better than
Q3 = 14 (Q3 would over-inflate the medium class). Descriptive stats (median 9,
mean 10.5, maximum 45) are read from
`bmd45_subset/train/_annotations.coco.json` and cached to
`outputs/density_thresholds.json`; delete that file to recompute.

Display cap: individual frames can contain a large outlier count (the tail
reaches 45+). The displayed count is capped (default 22) so outliers do not
distort the UI or aggregate stats. The label is always computed from the true,
uncapped count.

The count itself depends on the confidence method used to filter detections.
The adaptive and otsu methods run a single forward pass at a low base threshold
(0.05) to gather every candidate box; all further steps are post-processing.

- fixed: keep boxes whose confidence is at least a single global threshold.
  This is the default method in the app.
- adaptive: per image keep boxes whose confidence is at least
  max(base, image_max_confidence - margin). The bar drops for images whose best
  detection is weak and rises for clean images.
- otsu: histogram the per-image confidence scores and use Otsu's method to find
  the value that maximises between-class variance between the noise cluster and
  the signal cluster. If an image has fewer than 5 detections, Otsu cannot find
  a meaningful split, so the top 35 percent of boxes by confidence are kept
  instead. Either way a hard floor of 0.10 is enforced. This adapts to each
  image's own distribution with no margin and no max dependency.


## Known limitations

- Trained on a small subset (150 train, 30 validation) for a short, time-boxed
  run. This is a pipeline demonstration, not an accuracy-optimised model.
- Density labels count detected vehicles, so heavy occlusion, night frames, and
  small or distant vehicles can undercount and shift the label.
- The density cutoffs (5 and 11) are calibrated to the training subset, so they
  are camera and scene dependent.
- Counts depend on the confidence method and its parameters. The adaptive
  max-margin rule is aggressive and can prune valid detections on clean frames;
  Otsu is more data-driven; fixed is simplest but uses one number everywhere.
- No temporal information is used, so results are per-frame only.
