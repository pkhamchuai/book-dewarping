import cv2
import numpy as np
from .base import BookEdgeDetector


class ExtremePointsDetector(BookEdgeDetector):
    """Grayscale Canny → largest contour → sum/diff extreme points."""
    name = "M1: Extreme Points\n(Canny + Contour)"

    def detect(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        largest = max(contours, key=cv2.contourArea)
        pts = largest.reshape(-1, 2).astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        corners = np.array([
            pts[np.argmin(s)],    # top-left
            pts[np.argmin(diff)], # top-right
            pts[np.argmax(s)],    # bottom-right
            pts[np.argmax(diff)], # bottom-left
        ])

        result = image.copy()
        cv2.drawContours(result, [largest], -1, self.line_color, 2)
        return self.draw_polygon(result, corners)
