import cv2
import math
import numpy as np
from .base import BookEdgeDetector


class HSVHoughDetector(BookEdgeDetector):
    """
    Multi-channel edge detection (gray + HSV saturation + LAB luminance)
    fed into probabilistic Hough lines.  Lines are clustered into
    near-horizontal and near-vertical groups; the outermost pair from each
    group is intersected to recover the four page corners.
    """
    name = "M5: HSV + Hough Lines"

    def detect(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Edge-preserving smooth
        smooth = cv2.bilateralFilter(bgr, 9, 75, 75)

        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        hsv_s = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)[:, :, 1]
        lab_l = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)[:, :, 0]

        e_gray = cv2.Canny(gray,    30, 100)
        e_s    = cv2.Canny(hsv_s,   20,  80)
        e_l    = cv2.Canny(lab_l,   30, 100)
        edges  = cv2.bitwise_or(cv2.bitwise_or(e_gray, e_s), e_l)

        min_len = max(w, h) // 5
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=60, minLineLength=min_len, maxLineGap=25
        )

        result = image.copy()
        if lines is None:
            cv2.putText(result, "No lines found", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 60, 60), 2)
            return result

        # Draw all detected lines (faint)
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            cv2.line(result, (x1, y1), (x2, y2), (100, 200, 100), 1)

        # Cluster lines into near-horizontal / near-vertical
        horiz, vert = [], []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            dx, dy = x2 - x1, y2 - y1
            angle = abs(math.degrees(math.atan2(dy, dx)))
            if angle < 30 or angle > 150:
                horiz.append((x1, y1, x2, y2))
            elif 60 < angle < 120:
                vert.append((x1, y1, x2, y2))

        if len(horiz) < 2 or len(vert) < 2:
            cv2.putText(result, "Not enough boundary lines",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 60, 60), 2)
            return result

        def y_at_mid(ln):
            x1, y1, x2, y2 = ln
            if x2 == x1:
                return (y1 + y2) / 2
            return y1 + (y2 - y1) * (w / 2 - x1) / (x2 - x1)

        def x_at_mid(ln):
            x1, y1, x2, y2 = ln
            if y2 == y1:
                return (x1 + x2) / 2
            return x1 + (x2 - x1) * (h / 2 - y1) / (y2 - y1)

        top_ln    = min(horiz, key=y_at_mid)
        bottom_ln = max(horiz, key=y_at_mid)
        left_ln   = min(vert,  key=x_at_mid)
        right_ln  = max(vert,  key=x_at_mid)

        # Highlight the 4 boundary lines
        for ln, col in [
            (top_ln,    (255, 255,   0)),
            (bottom_ln, (255, 165,   0)),
            (left_ln,   (  0, 200, 255)),
            (right_ln,  (200,   0, 255)),
        ]:
            x1, y1, x2, y2 = ln
            # Extend line across the full image
            if abs(x2 - x1) > 1:
                slope = (y2 - y1) / (x2 - x1)
                ya = int(y1 + slope * (0 - x1))
                yb = int(y1 + slope * (w - x1))
                cv2.line(result, (0, ya), (w, yb), col, 2)
            else:
                cv2.line(result, (x1, 0), (x2, h), col, 2)

        # Intersect the 4 boundary lines
        corners = []
        for h_ln in (top_ln, bottom_ln):
            for v_ln in (left_ln, right_ln):
                pt = _intersect(h_ln, v_ln)
                if pt is not None:
                    corners.append(pt)

        if len(corners) == 4:
            corners = np.array(corners, dtype=np.float32)
            # Sort: TL, TR, BR, BL
            s    = corners.sum(axis=1)
            diff = np.diff(corners, axis=1).ravel()
            ordered = np.array([
                corners[np.argmin(s)],
                corners[np.argmin(diff)],
                corners[np.argmax(s)],
                corners[np.argmax(diff)],
            ])
            return self.draw_polygon(result, ordered)

        return result


def _intersect(line1, line2):
    """Return the intersection point of two lines (each as x1,y1,x2,y2), or None."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return np.array([x, y], dtype=np.float32)
