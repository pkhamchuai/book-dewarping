from .method1_extreme_points import ExtremePointsDetector
from .method2_grabcut import GrabCutDetector
from .method3_sam import SAMDetector
from .method4_depth import DepthDetector
from .method5_hsv_hough import HSVHoughDetector

__all__ = [
    "ExtremePointsDetector",
    "GrabCutDetector",
    "SAMDetector",
    "DepthDetector",
    "HSVHoughDetector",
]
