"""Command-line inference pipeline.

Run detection on a single image or a folder of images, print a traffic-state
summary, optionally save annotated images, and write per-image predictions to
CSV or JSON.

Examples
--------
    # Single image -> JSON
    traffic-state --input bmd45_subset/valid/some.png --format json

    # Folder -> CSV, also save annotated images
    traffic-state --input bmd45_subset/valid --format csv --save-annotated
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from traffic_state.config import (
    ADAPTIVE_BASE_THRESHOLD,
    ADAPTIVE_MARGIN,
    CONFIDENCE_METHODS,
    DEFAULT_THRESHOLD,
    OTSU_FLOOR,
    OUTPUT_DIR,
)
from traffic_state.density import load_thresholds
from traffic_state.detector import (
    ImageResult,
    annotate,
    load_detector,
    run_on_folder,
    run_on_image,
)


def _write_csv(results: list[ImageResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "width",
                "height",
                "vehicle_count",
                "display_count",
                "density",
                "method",
                "applied_threshold",
                "class_counts",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "file_name": r.file_name,
                    "width": r.width,
                    "height": r.height,
                    "vehicle_count": r.vehicle_count,
                    "display_count": r.display_count,
                    "density": r.density,
                    "method": r.method,
                    "applied_threshold": round(r.applied_threshold, 4),
                    # class_counts serialised as JSON so it fits one CSV cell.
                    "class_counts": json.dumps(r.class_counts),
                }
            )


def _write_json(results: list[ImageResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_row() for r in results], indent=2))


def _print_summary(results: list[ImageResult]) -> None:
    from collections import Counter

    print(f"\nProcessed {len(results)} image(s):")
    for r in results:
        classes = ", ".join(f"{k}:{v}" for k, v in r.class_counts.items()) or "-"
        print(
            f"  {r.file_name:<40} count={r.vehicle_count:>3}  "
            f"density={r.density:<7}  conf>={r.applied_threshold:.2f}  [{classes}]"
        )
    dist = Counter(r.density for r in results)
    print(f"\nDensity distribution: {dict(dist)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BMD-45 vehicle detection + traffic-state labelling.")
    parser.add_argument("--input", "-i", required=True, help="Path to an image or a folder of images.")
    parser.add_argument("--output-dir", "-o", default=str(OUTPUT_DIR), help="Directory for prediction files.")
    parser.add_argument("--format", "-f", choices=["csv", "json"], default="json", help="Output format.")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="Detection confidence threshold (fixed method).")
    parser.add_argument("--method", "-m", choices=CONFIDENCE_METHODS, default="fixed", help="Per-image confidence method.")
    parser.add_argument("--base-threshold", type=float, default=ADAPTIVE_BASE_THRESHOLD, help="Low base threshold for adaptive/otsu passes.")
    parser.add_argument("--margin", type=float, default=ADAPTIVE_MARGIN, help="Adaptive margin below each image's max confidence.")
    parser.add_argument("--otsu-floor", type=float, default=OTSU_FLOOR, help="Safety floor for the otsu method.")
    parser.add_argument("--save-annotated", action="store_true", help="Also save annotated images.")
    parser.add_argument("--checkpoint", default=None, help="Override the model checkpoint path.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    thresholds = load_thresholds()
    detector = load_detector(args.checkpoint)

    print(
        f"Density cutoffs -> low<={thresholds.low_max}, "
        f"medium<={thresholds.medium_max}, high>{thresholds.medium_max}"
    )
    print(f"Confidence method -> {args.method}")

    detect_kwargs = dict(
        threshold=args.threshold,
        method=args.method,
        base_threshold=args.base_threshold,
        margin=args.margin,
        otsu_floor=args.otsu_floor,
        detector=detector,
        thresholds=thresholds,
    )
    if input_path.is_dir():
        results = run_on_folder(input_path, **detect_kwargs)
        stem = input_path.name or "folder"
    elif input_path.is_file():
        results = [run_on_image(str(input_path), **detect_kwargs)]
        stem = input_path.stem
    else:
        parser.error(f"Input path does not exist: {input_path}")
        return 2

    if not results:
        print("No images found to process.")
        return 1

    out_file = output_dir / f"predictions_{stem}.{args.format}"
    if args.format == "csv":
        _write_csv(results, out_file)
    else:
        _write_json(results, out_file)

    if args.save_annotated:
        annotated_dir = output_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            src = input_path if input_path.is_file() else input_path / r.file_name
            annotate(str(src), r).save(annotated_dir / f"annotated_{r.file_name}")
        print(f"Saved annotated images to {annotated_dir}")

    _print_summary(results)
    print(f"\nSaved predictions to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
