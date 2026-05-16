"""
Book edge detection — HSV + Hough Lines method.
Saves side-by-side (original | result) PNGs to output_dir.

Usage:
    python main.py [--input data/SitaPB] [--output output]
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods import HSVHoughDetector

RESIZE_HEIGHT = 800
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def load_image(path: str, height: int = RESIZE_HEIGHT) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read {path}")
    ratio = height / bgr.shape[0]
    bgr = cv2.resize(bgr, (int(bgr.shape[1] * ratio), height))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def collect_images(directory: str) -> list[str]:
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(IMG_EXTENSIONS)
    )


def process_image(path: str, detector: HSVHoughDetector, output_dir: str):
    print(f"  {os.path.basename(path)}", flush=True)
    try:
        image = load_image(path)
    except Exception as e:
        print(f"    SKIP — {e}")
        return

    result = detector.safe_detect(image.copy())

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(os.path.basename(path), fontsize=12, fontweight="bold")

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(result)
    axes[1].set_title(detector.name)
    axes[1].axis("off")

    plt.tight_layout()
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(output_dir, f"{stem}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/SitaPB")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = collect_images(args.input)
    if not files:
        sys.exit(f"No images found in {args.input}")

    print(f"Found {len(files)} image(s) in '{args.input}'")
    detector = HSVHoughDetector()

    for path in files:
        process_image(path, detector, args.output)

    print(f"\nDone — results in '{args.output}/'")


if __name__ == "__main__":
    main()
