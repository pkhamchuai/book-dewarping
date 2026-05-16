import cv2
import math
import numpy as np
from .base import BookEdgeDetector


class HSVHoughDetector(BookEdgeDetector):
    """
    Book page boundary detector.

    Primary method — edge-color scanning:
        Scan inward from each image border at many positions.  At each
        position, compare pixels against the local edge reference (the first
        few pixels from the border, which should be background / table /
        finger).  Collect the transition points, remove outliers, then fit a
        straight line through them with cv2.fitLine.  This finds the real
        page edge even when strong diagonal content lines exist inside the
        photo, because it anchors on the actual color change at the physical
        page boundary.

    Fallback — Hough lines:
        For any edge where the color scan fails (too few transition points),
        fall back to probabilistic Hough with position-zone and angle-
        tolerance constraints.
    """
    name = "HSV + Hough Lines"

    # ── color-scan parameters ──────────────────────────────────────────────
    SCAN_SAMPLES = 50   # positions sampled along each edge
    COLOR_THRESH = 28   # L2 RGB distance that counts as "different from edge"
    SCAN_SKIP    = 4    # pixels to skip from the very border (avoid aliasing)
    MIN_SCAN_PTS = 8    # minimum inlier points to trust the scanned line

    # ── Hough fallback parameters ──────────────────────────────────────────
    BORDER_FRAC   = 0.30
    MIN_SPAN_FRAC = 0.20
    ANGLE_TOL     = 15   # degrees from H or V


    def detect(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.copy()

        # ── 1. Primary: color-scan each edge ──────────────────────────────
        boundary = {}
        for edge in ('top', 'bottom', 'left', 'right'):
            pts = _scan_boundary(image, edge,
                                 self.SCAN_SAMPLES,
                                 self.COLOR_THRESH,
                                 self.SCAN_SKIP,
                                 self.MIN_SCAN_PTS)
            if pts is not None:
                ln = _fit_line(pts, h, w, edge)
                if ln is not None:
                    boundary[edge] = ln

        # ── 2. Fallback: Hough for any edge color-scan missed ─────────────
        missing = [e for e in ('top', 'bottom', 'left', 'right')
                   if e not in boundary]
        if missing:
            bgr    = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            smooth = cv2.bilateralFilter(bgr, 9, 75, 75)
            gray   = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
            hsv_s  = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)[:, :, 1]
            lab_l  = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)[:, :, 0]
            edges  = cv2.bitwise_or(
                cv2.bitwise_or(cv2.Canny(gray, 30, 100),
                               cv2.Canny(hsv_s, 20, 80)),
                cv2.Canny(lab_l, 30, 100))
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=60,
                minLineLength=max(w, h) // 5, maxLineGap=25)

            if lines is not None:
                horiz, vert = _classify_lines(lines, self.ANGLE_TOL)
                for edge in missing:
                    pool = horiz if edge in ('top', 'bottom') else vert
                    ln = _pick_hough(pool, edge, h, w,
                                     self.BORDER_FRAC, self.MIN_SPAN_FRAC)
                    if ln is not None:
                        boundary[edge] = ln

        if len(boundary) < 4:
            cv2.putText(result, "Detection failed", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 60, 60), 2)
            return result

        # ── 3. Intersect the 4 boundary lines → corners ───────────────────
        corners = []
        for h_ln in (boundary['top'], boundary['bottom']):
            for v_ln in (boundary['left'], boundary['right']):
                pt = _intersect(h_ln, v_ln)
                if pt is not None:
                    corners.append(pt)

        if len(corners) == 4:
            corners = np.array(corners, dtype=np.float32)
            s    = corners.sum(axis=1)
            diff = np.diff(corners, axis=1).ravel()
            # TL, TR, BR, BL
            ordered = np.array([corners[np.argmin(s)],
                                 corners[np.argmin(diff)],
                                 corners[np.argmax(s)],
                                 corners[np.argmax(diff)]])
            result = self.draw_polygon(result, ordered)
            result = _label_spine_edge(result, ordered)
            return result

        return result


# ── helpers ────────────────────────────────────────────────────────────────

def _scan_boundary(image, edge, n_samples, thresh, skip, min_pts):
    """
    Scan inward from 'edge' at n_samples evenly-spaced positions.
    At each position the reference color is the mean of the first `skip`
    pixels from the border.  We walk inward until the pixel diverges by
    more than `thresh` in L2 RGB distance.  Outliers (>2σ from median) are
    removed.  Returns Nx2 int32 array or None.
    """
    h, w = image.shape[:2]
    pts = []

    if edge == 'top':
        xs = np.linspace(int(w * 0.05), int(w * 0.95), n_samples, dtype=int)
        for x in xs:
            ref = image[:skip, x].astype(np.float32).mean(axis=0)
            for y in range(skip, h // 2):
                if np.linalg.norm(image[y, x].astype(np.float32) - ref) > thresh:
                    pts.append((x, y))
                    break

    elif edge == 'bottom':
        xs = np.linspace(int(w * 0.05), int(w * 0.95), n_samples, dtype=int)
        for x in xs:
            ref = image[h - skip:, x].astype(np.float32).mean(axis=0)
            for y in range(h - 1 - skip, h // 2, -1):
                if np.linalg.norm(image[y, x].astype(np.float32) - ref) > thresh:
                    pts.append((x, y))
                    break

    elif edge == 'left':
        ys = np.linspace(int(h * 0.05), int(h * 0.95), n_samples, dtype=int)
        for y in ys:
            ref = image[y, :skip].astype(np.float32).mean(axis=0)
            for x in range(skip, w // 2):
                if np.linalg.norm(image[y, x].astype(np.float32) - ref) > thresh:
                    pts.append((x, y))
                    break

    elif edge == 'right':
        ys = np.linspace(int(h * 0.05), int(h * 0.95), n_samples, dtype=int)
        for y in ys:
            ref = image[y, w - skip:].astype(np.float32).mean(axis=0)
            for x in range(w - 1 - skip, w // 2, -1):
                if np.linalg.norm(image[y, x].astype(np.float32) - ref) > thresh:
                    pts.append((x, y))
                    break

    if len(pts) < min_pts:
        return None

    pts = np.array(pts, dtype=np.int32)

    # Directional median filter: the correct transitions are always the ones
    # closest to the image border.  Use the median as a natural split point:
    # keep the border-proximate half and discard the content-side half.
    # This handles wide spine strips and internal photo colour changes that
    # produce deep outliers, without needing a hand-tuned IQR coefficient.
    vals = pts[:, 1] if edge in ('top', 'bottom') else pts[:, 0]
    med = float(np.median(vals))

    if edge in ('top', 'left'):
        mask = vals <= med      # keep LOW values (near border)
    else:
        mask = vals >= med      # keep HIGH values (near border)

    pts = pts[mask]
    if len(pts) < min_pts:
        return None

    # Sanity check: if the remaining cluster is still too deep into the image
    # the scan latched onto internal content — reject and let Hough handle it.
    med2 = float(np.median(vals[mask]))
    if edge == 'top'    and med2 > h * 0.20: return None
    if edge == 'bottom' and med2 < h * 0.80: return None
    if edge == 'left'   and med2 > w * 0.20: return None
    if edge == 'right'  and med2 < w * 0.80: return None

    return pts


def _fit_line(pts, h, w, edge):
    """
    Fit a line through pts (Nx2) using cv2.fitLine (L2) and extend it to
    cover the full image width (top/bottom) or height (left/right).
    Returns (x1, y1, x2, y2) or None.
    """
    pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
    vx, vy, cx, cy = cv2.fitLine(pts_cv, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, cx, cy = float(vx), float(vy), float(cx), float(cy)

    if edge in ('top', 'bottom'):
        if abs(vx) < 1e-6:
            y = int(cy)
            return (0, y, w, y)
        t1 = (0 - cx) / vx
        t2 = (w - cx) / vx
        return (0, int(cy + t1 * vy), w, int(cy + t2 * vy))
    else:
        if abs(vy) < 1e-6:
            x = int(cx)
            return (x, 0, x, h)
        t1 = (0 - cy) / vy
        t2 = (h - cy) / vy
        return (int(cx + t1 * vx), 0, int(cx + t2 * vx), h)


def _classify_lines(lines, angle_tol):
    horiz, vert = [], []
    at = angle_tol
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < at or angle > (180 - at):
            horiz.append((x1, y1, x2, y2))
        elif (90 - at) < angle < (90 + at):
            vert.append((x1, y1, x2, y2))
    return horiz, vert


def _pick_hough(candidates, edge, h, w, border_frac, min_span_frac):
    if not candidates:
        return None

    def y_mid(ln):
        x1, y1, x2, y2 = ln
        return y1 + (y2-y1)*(w/2-x1)/(x2-x1) if x2 != x1 else (y1+y2)/2

    def x_mid(ln):
        x1, y1, x2, y2 = ln
        return x1 + (x2-x1)*(h/2-y1)/(y2-y1) if y2 != y1 else (x1+x2)/2

    if edge in ('top', 'bottom'):
        key_fn   = y_mid
        span_fn  = lambda ln: abs(ln[2] - ln[0])
        min_span = w * min_span_frac
        zone     = (lambda ln: y_mid(ln) < h * border_frac) if edge == 'top' \
                   else (lambda ln: y_mid(ln) > h * (1 - border_frac))
        prefer_min = edge == 'top'
    else:
        key_fn   = x_mid
        span_fn  = lambda ln: abs(ln[3] - ln[1])
        min_span = h * min_span_frac
        zone     = (lambda ln: x_mid(ln) < w * border_frac) if edge == 'left' \
                   else (lambda ln: x_mid(ln) > w * (1 - border_frac))
        prefer_min = edge == 'left'

    zoned_long = [c for c in candidates if zone(c) and span_fn(c) >= min_span]
    pool = zoned_long or [c for c in candidates if zone(c)] or candidates
    return min(pool, key=key_fn) if prefer_min else max(pool, key=key_fn)


def _intersect(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    return np.array([x1 + t*(x2-x1), y1 + t*(y2-y1)], dtype=np.float32)


def _label_spine_edge(image: np.ndarray, ordered: np.ndarray) -> np.ndarray:
    """
    The spine side of an open book curves toward the camera, making it
    appear taller in the image than the flat outer edge.  Compare the pixel
    lengths of TL→BL (left side) and TR→BR (right side) and label each.
    """
    tl, tr, br, bl = ordered
    left_len  = float(np.linalg.norm(tl - bl))
    right_len = float(np.linalg.norm(tr - br))

    left_mid  = ((tl + bl) / 2).astype(int)
    right_mid = ((tr + br) / 2).astype(int)

    labels = ([(left_mid, "Spine"), (right_mid, "Edge")]
              if left_len >= right_len
              else [(right_mid, "Spine"), (left_mid, "Edge")])

    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.75, 2
    for mid, text in labels:
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        x = int(mid[0] - tw / 2)
        y = int(mid[1] + th / 2)
        pad = 5
        cv2.rectangle(result,
                      (x - pad, y - th - pad),
                      (x + tw + pad, y + baseline + pad),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(result, text, (x, y), font, scale, (255, 255, 255), thick)
    return result
