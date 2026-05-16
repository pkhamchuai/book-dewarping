"""
Book edge detection comparison.
Runs 5 detection methods on every image in input_dir and saves a
2×3 subplot PNG per image to output_dir.

Layout per image:
    [Original]   [M1: Extreme Pts]  [M2: GrabCut]
    [M3: SAM]    [M4: Depth/MiDaS]  [M5: HSV+Hough]

Usage:
    python main.py [--input data/SitaPB] [--output output]
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np

from methods import (
    ExtremePointsDetector,
    GrabCutDetector,
    SAMDetector,
    DepthDetector,
    HSVHoughDetector,
)

RESIZE_HEIGHT = 800
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def load_image(path: str, height: int = RESIZE_HEIGHT) -> np.ndarray:
    """Read, resize, and return an RGB uint8 image."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read {path}")
    ratio = height / bgr.shape[0]
    w = int(bgr.shape[1] * ratio)
    bgr = cv2.resize(bgr, (w, height))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def collect_images(directory: str) -> list[str]:
    paths = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(IMG_EXTENSIONS):
            paths.append(os.path.join(directory, name))
    return paths


def process_image(image_path: str, detectors: list, output_dir: str):
    print(f"  Processing {os.path.basename(image_path)} ...", flush=True)

    try:
        image = load_image(image_path)
    except Exception as e:
        print(f"    SKIP — {e}")
        return

    # 2 rows × 3 cols per image
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    fig.suptitle(os.path.basename(image_path), fontsize=14, fontweight="bold", y=1.01)

    # (0,0) — original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original", fontsize=11)
    axes[0, 0].axis("off")

    # Methods in the remaining 5 slots
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for detector, pos in zip(detectors, positions):
        method_name = detector.name.replace("\n", " ")
        print(f"    → {method_name}", flush=True)
        result = detector.safe_detect(image.copy())
        axes[pos].imshow(result)
        axes[pos].set_title(detector.name, fontsize=10)
        axes[pos].axis("off")

    plt.tight_layout()

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{stem}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Book edge detection — 5 methods")
    parser.add_argument("--input",  default="data/SitaPB", help="Input image directory")
    parser.add_argument("--output", default="output",      help="Output directory for PNGs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image_files = collect_images(args.input)
    if not image_files:
        sys.exit(f"No images found in {args.input}")

    print(f"Found {len(image_files)} image(s) in '{args.input}'")
    print("Loading detectors …")

    detectors = [
        ExtremePointsDetector(),
        GrabCutDetector(),
        SAMDetector(),
        DepthDetector(),
        HSVHoughDetector(),
    ]

    for path in image_files:
        process_image(path, detectors, args.output)

    print(f"\nDone. Results saved to '{args.output}/'")


if __name__ == "__main__":
    main()
