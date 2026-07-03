"""Streamlit app: vehicle detection + roadway-density labelling on BMD-45.

Run with:
    uv run streamlit run app/streamlit_app.py

Features:
- Upload one or more images, or pick sample frames from the validation set.
- Switchable model checkpoint (sidebar).
- Chunked batch inference with a progress bar (memory-friendly for many images).
- One-image-at-a-time viewer with Previous / Next buttons (no long scroll).
- Predictions auto-saved to a CSV named by confidence level, with a notification.
- Bar graphs: density distribution + total vehicles per class.
"""

from __future__ import annotations

import io
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import streamlit as st
from PIL import Image

# The traffic_state package lives under src/. Make it importable when running
# `streamlit run app/streamlit_app.py` without an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_state.config import (  # noqa: E402  (import after sys.path setup)
    ADAPTIVE_BASE_THRESHOLD,
    ADAPTIVE_MARGIN,
    CHECKPOINT_PATH,
    CONFIDENCE_METHODS,
    DEFAULT_THRESHOLD,
    DISPLAY_COUNT_CAP,
    OTSU_FLOOR,
    OUTPUT_DIR,
    VALID_DIR,
    get_device,
)
from traffic_state.density import load_thresholds  # noqa: E402
from traffic_state.detector import (  # noqa: E402
    ImageResult,
    annotate,
    iter_image_paths,
    load_detector,
    run_on_image,
)

DENSITY_ORDER = ["low", "medium", "high", "unclear"]
DEFAULT_CHUNK_SIZE = 8
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

LIMITATIONS = """
- Trained on a **small subset** (150 train / 30 val) of BMD-45 for a short,
  time-boxed run; this is a pipeline demo, not an accuracy-optimised model.
- Density labels count **detected** vehicles; heavy occlusion, night frames, or
  small/distant vehicles can undercount and skew the label.
- Thresholds (Q1/Q3) are derived from the training subset's vehicle-per-image
  distribution, so they are camera/scene dependent.
- Counts depend on the confidence threshold: raise it to reduce false
  positives, lower it to catch faint detections.
"""


class Controls(NamedTuple):
    """User-selected settings from the sidebar."""

    ckpt_name: str
    ckpt_path: Path
    ckpt_mtime: float
    method: str
    threshold: float
    base_threshold: float
    margin: float
    otsu_floor: float
    chunk_size: int
    csv_name: str


@st.cache_resource(show_spinner="Loading model...")
def get_detector(ckpt_path: str, _ckpt_mtime: float):
    """Load a detector for a specific checkpoint (cached per path + mtime)."""
    return load_detector(ckpt_path)


@st.cache_resource
def get_thresholds():
    """Load the density thresholds once per session."""
    return load_thresholds()


@st.cache_data(show_spinner=False)
def infer(
    image_bytes: bytes,
    file_name: str,
    method: str,
    threshold: float,
    base_threshold: float,
    margin: float,
    otsu_floor: float,
    ckpt_path: str,
    _ckpt_mtime: float,
) -> ImageResult:
    """Run cached inference keyed on image bytes + settings + checkpoint."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return run_on_image(
        image,
        threshold=threshold,
        method=method,
        base_threshold=base_threshold,
        margin=margin,
        otsu_floor=otsu_floor,
        detector=get_detector(ckpt_path, _ckpt_mtime),
        thresholds=get_thresholds(),
        file_name=file_name,
    )


def run_batch_chunked(
    images: list[tuple[str, bytes]], controls: Controls
) -> list[ImageResult]:
    """Run inference over the whole batch in chunks, with a progress bar.

    Per-image results are cached, so re-runs (navigation, chart redraws) are
    instant and only new/changed images actually hit the model.
    """
    results: list[ImageResult] = []
    total = len(images)
    ckpt_path = str(controls.ckpt_path)
    progress = st.progress(0.0, text=f"Running inference on {total} image(s)...")
    for start in range(0, total, controls.chunk_size):
        chunk = images[start:start + controls.chunk_size]
        for name, data in chunk:
            results.append(
                infer(
                    data,
                    name,
                    controls.method,
                    controls.threshold,
                    controls.base_threshold,
                    controls.margin,
                    controls.otsu_floor,
                    ckpt_path,
                    controls.ckpt_mtime,
                )
            )
        done = min(start + len(chunk), total)
        progress.progress(done / total, text=f"Processed {done}/{total} image(s)")
    progress.empty()
    return results


def discover_checkpoints() -> dict[str, Path]:
    """Find available .pth checkpoints next to the default one."""
    ckpts: dict[str, Path] = {}
    ckpt_dir = CHECKPOINT_PATH.parent
    if ckpt_dir.exists():
        for path in sorted(ckpt_dir.glob("*.pth")):
            ckpts[path.name] = path
    if CHECKPOINT_PATH.exists():
        ckpts.setdefault(CHECKPOINT_PATH.name, CHECKPOINT_PATH)
    return ckpts


def build_summary(results: list[ImageResult]) -> pd.DataFrame:
    """Build the per-image summary table, including the confidence used per image."""
    return pd.DataFrame(
        {
            "file_name": [r.file_name for r in results],
            "vehicle_count": [r.display_count for r in results],
            "density": [r.density for r in results],
            "vehicles_detected": [
                ", ".join(f"{name}: {count}" for name, count in r.class_counts.items()) or "-"
                for r in results
            ],
            "method": [r.method for r in results],
            "confidence_threshold": [round(r.applied_threshold, 2) for r in results],
        }
    )


def default_csv_name(method: str, threshold: float, base_threshold: float,
                     margin: float, otsu_floor: float) -> str:
    """Suggest a CSV filename that encodes the confidence method and its params."""
    if method == "adaptive":
        return (
            f"predictions_adaptive_b{int(round(base_threshold * 100)):02d}"
            f"_m{int(round(margin * 100)):02d}.csv"
        )
    if method == "otsu":
        return f"predictions_otsu_floor{int(round(otsu_floor * 100)):02d}.csv"
    return f"predictions_conf{int(round(threshold * 100)):02d}.csv"


def sanitize_csv_name(name: str) -> str:
    """Reduce a user-entered name to a safe .csv filename."""
    stem = Path(name.strip()).name or "predictions.csv"
    if not stem.lower().endswith(".csv"):
        stem += ".csv"
    return stem


def save_predictions_csv(summary_df: pd.DataFrame, csv_name: str) -> Path:
    """Write predictions to the predictions folder under the given file name."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / sanitize_csv_name(csv_name)
    summary_df.to_csv(out_path, index=False)
    return out_path


def sidebar_controls(thresholds, checkpoints: dict[str, Path]) -> Controls:
    """Render the sidebar and return the selected settings.

    Calls ``st.stop()`` when no model checkpoint is available.
    """
    with st.sidebar:
        st.header("Settings")

        st.subheader("Model")
        if not checkpoints:
            st.error(
                f"No `.pth` checkpoints found in `{CHECKPOINT_PATH.parent}`.\n\n"
                "Train first (`uv run python training_mps.py`) or set "
                "`TRAFFIC_STATE_CHECKPOINT`."
            )
            st.stop()

        ckpt_names = list(checkpoints)
        default_idx = (
            ckpt_names.index(CHECKPOINT_PATH.name)
            if CHECKPOINT_PATH.name in ckpt_names
            else 0
        )
        ckpt_name = st.selectbox("Checkpoint", ckpt_names, index=default_idx)
        ckpt_path = checkpoints[ckpt_name]

        st.subheader("Confidence method")
        method = st.radio(
            "Method",
            CONFIDENCE_METHODS,
            index=CONFIDENCE_METHODS.index("fixed"),
            format_func=lambda m: {
                "fixed": "Fixed slider",
                "adaptive": "Adaptive (max - margin)",
                "otsu": "Otsu clustering",
            }[m],
            help=(
                "Fixed: one global threshold. Adaptive: per image keep boxes with "
                "confidence >= max(base, image_max - margin). Otsu: split each "
                "image's confidence distribution into noise/signal clusters."
            ),
        )
        threshold = DEFAULT_THRESHOLD
        base_threshold = ADAPTIVE_BASE_THRESHOLD
        margin = ADAPTIVE_MARGIN
        otsu_floor = OTSU_FLOOR
        if method == "fixed":
            threshold = st.slider("Detection confidence", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05)
        elif method == "adaptive":
            base_threshold = st.slider("Base threshold", 0.05, 0.50, ADAPTIVE_BASE_THRESHOLD, 0.05)
            margin = st.slider("Margin below image max", 0.05, 0.40, ADAPTIVE_MARGIN, 0.05)
        else:
            base_threshold = st.slider("Base threshold", 0.05, 0.50, ADAPTIVE_BASE_THRESHOLD, 0.05)
            otsu_floor = st.slider("Otsu safety floor", 0.05, 0.50, OTSU_FLOOR, 0.05)

        chunk_size = st.number_input(
            "Batch chunk size",
            min_value=1,
            max_value=64,
            value=DEFAULT_CHUNK_SIZE,
            step=1,
            help="Images are processed in chunks of this size, with a progress bar.",
        )

        st.markdown("---")
        st.subheader("Save")
        csv_name = st.text_input(
            "CSV file name",
            value=default_csv_name(method, threshold, base_threshold, margin, otsu_floor),
            help="Predictions are auto-saved to outputs/predictions/ under this name.",
        )

        st.markdown("---")
        st.subheader("Density thresholds")
        st.markdown(
            f"- **unclear**: 0 vehicles\n"
            f"- **low**: 1-{thresholds.low_max} vehicles\n"
            f"- **medium**: {thresholds.low_max + 1}-{thresholds.medium_max} vehicles\n"
            f"- **high**: > {thresholds.medium_max} vehicles\n\n"
            f"_(Calibrated from the training subset. Displayed count capped at "
            f"{DISPLAY_COUNT_CAP}; the label always uses the true count.)_"
        )
        st.markdown("---")
        st.caption(f"Device: `{get_device()}`")

    return Controls(
        ckpt_name=ckpt_name,
        ckpt_path=ckpt_path,
        ckpt_mtime=ckpt_path.stat().st_mtime,
        method=str(method),
        threshold=float(threshold),
        base_threshold=float(base_threshold),
        margin=float(margin),
        otsu_floor=float(otsu_floor),
        chunk_size=int(chunk_size),
        csv_name=str(csv_name),
    )


def select_images() -> list[tuple[str, bytes]]:
    """Let the user upload images or pick validation frames; return (name, bytes)."""
    source = st.radio(
        "Image source", ["Upload", "Sample (validation set)"], horizontal=True
    )
    if source == "Upload":
        uploads = st.file_uploader(
            "Upload one or more traffic-camera images",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
        return [(u.name, u.getvalue()) for u in (uploads or [])]

    sample_paths = iter_image_paths(VALID_DIR) if VALID_DIR.exists() else []
    if not sample_paths:
        st.warning(f"No sample images found in `{VALID_DIR}`.")
        return []
    names = [p.name for p in sample_paths]
    chosen = st.multiselect("Pick validation frames", names, default=names[:3])
    path_by_name = {p.name: p for p in sample_paths}
    return [(name, path_by_name[name].read_bytes()) for name in chosen]


def show_result(image_bytes: bytes, result: ImageResult) -> None:
    """Render one image's original + annotated view and its per-class counts."""
    count_text = f"{result.display_count} vehicles"
    if result.vehicle_count > result.display_count:
        count_text = f"{result.display_count}+ vehicles (capped)"
    st.subheader(f"{result.file_name} - {result.density.upper()} ({count_text})")

    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.caption("Original")
        st.image(image_bytes, use_container_width=True)
    with col_annot:
        st.caption("Detections")
        annotated = annotate(Image.open(io.BytesIO(image_bytes)).convert("RGB"), result)
        st.image(annotated, use_container_width=True)

    if result.class_counts:
        counts_df = pd.DataFrame(
            sorted(result.class_counts.items(), key=lambda kv: kv[1], reverse=True),
            columns=["Vehicle class", "Count"],
        )
        st.dataframe(counts_df, hide_index=True, use_container_width=True)
    else:
        st.info(
            "No vehicles detected above the current confidence threshold; "
            "density = unclear."
        )


@st.fragment
def image_navigator(
    images: list[tuple[str, bytes]], results: list[ImageResult]
) -> None:
    """Show one image at a time with Previous / Next buttons.

    Isolated in a fragment so paging between images only reruns this section
    (not the whole app and not inference).
    """
    n = len(results)
    st.session_state.setdefault("img_idx", 0)
    if st.session_state.img_idx >= n:
        st.session_state.img_idx = 0
    idx = st.session_state.img_idx

    nav_prev, nav_info, nav_next = st.columns([1, 2, 1])
    with nav_prev:
        if st.button("Previous", disabled=idx <= 0, use_container_width=True, key="nav_prev"):
            st.session_state.img_idx = max(0, idx - 1)
            st.rerun(scope="fragment")
    with nav_next:
        if st.button("Next", disabled=idx >= n - 1, use_container_width=True, key="nav_next"):
            st.session_state.img_idx = min(n - 1, idx + 1)
            st.rerun(scope="fragment")
    with nav_info:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>"
            f"Image <b>{idx + 1}</b> of <b>{n}</b></div>",
            unsafe_allow_html=True,
        )

    _, data = images[idx]
    show_result(data, results[idx])


def render_bar_graphs(results: list[ImageResult]) -> None:
    """Draw density distribution and total vehicles-per-class bar charts."""
    st.markdown("### Charts")
    col_density, col_class = st.columns(2)

    with col_density:
        st.caption("Density distribution (images per label)")
        dist = pd.Series(Counter(r.density for r in results)).reindex(
            DENSITY_ORDER, fill_value=0
        )
        st.bar_chart(dist, color="#4363d8")

    with col_class:
        st.caption("Total detected vehicles per class")
        class_totals: Counter = Counter()
        for result in results:
            class_totals.update(result.class_counts)
        if class_totals:
            class_df = pd.DataFrame(
                sorted(class_totals.items(), key=lambda kv: kv[1], reverse=True),
                columns=["class", "count"],
            ).set_index("class")
            st.bar_chart(class_df, color="#3cb44b")
        else:
            st.info("No vehicles detected across the current batch.")


def render_summary(results: list[ImageResult], controls: Controls) -> None:
    """Show the summary table, auto-save the CSV, and notify the user."""
    st.markdown("### Summary")
    st.caption("Click a row to open that image below.")
    summary_df = build_summary(results)
    event = st.dataframe(
        summary_df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="summary_table",
    )

    # Clicking a summary row selects that image in the per-image viewer.
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        st.session_state.img_idx = int(selected_rows[0])

    # Auto-save once per unique (checkpoint, confidence settings, file name, image
    # set) combination so the CSV always reflects the current settings without
    # re-saving on every rerun. A transient toast confirms each new save.
    save_sig = (
        controls.ckpt_name,
        controls.method,
        controls.threshold,
        controls.base_threshold,
        controls.margin,
        controls.otsu_floor,
        controls.csv_name,
        tuple(r.file_name for r in results),
    )
    if st.session_state.get("saved_sig") != save_sig:
        saved_path = save_predictions_csv(summary_df, controls.csv_name)
        st.session_state.saved_sig = save_sig
        st.session_state.saved_path = str(saved_path)
        st.toast(f"Saved predictions to {saved_path}")

    st.success(f"Predictions auto-saved to `{st.session_state.saved_path}`")
    st.download_button(
        "Download predictions (CSV)",
        summary_df.to_csv(index=False).encode(),
        file_name=Path(st.session_state.saved_path).name,
        mime="text/csv",
    )


def main() -> None:
    """Run the Streamlit traffic-state application."""
    st.set_page_config(page_title="BMD-45 Traffic State", layout="wide")
    st.title("Traffic-State Detector")
    st.write(
        "Vehicle detection on traffic-camera images (RF-DETR-Nano fine-tuned on "
        "BMD-45), with a simple **low / medium / high / unclear** density label."
    )

    thresholds = get_thresholds()
    controls = sidebar_controls(thresholds, discover_checkpoints())

    images = select_images()
    if not images:
        st.info("Upload images or select sample frames to run detection.")
        st.markdown("### Notes & limitations")
        st.markdown(LIMITATIONS)
        return

    # Reset the viewer to the first image whenever the input set changes.
    batch_sig = (
        controls.ckpt_name,
        controls.method,
        controls.threshold,
        controls.base_threshold,
        controls.margin,
        controls.otsu_floor,
        tuple(n for n, _ in images),
    )
    if st.session_state.get("batch_sig") != batch_sig:
        st.session_state.batch_sig = batch_sig
        st.session_state.img_idx = 0

    results = run_batch_chunked(images, controls)

    render_summary(results, controls)

    st.markdown("### Per-image results")
    with st.container(border=True):
        image_navigator(images, results)

    render_bar_graphs(results)

    with st.expander("Notes & limitations"):
        st.markdown(LIMITATIONS)


if __name__ == "__main__":
    main()
