import cv2
import numpy as np
from .base import BookEdgeDetector


class GrabCutDetector(BookEdgeDetector):
    """GrabCut color segmentation: border → background, center → foreground."""
    name = "M2: GrabCut\n(Color Segmentation)"

    def detect(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        # Definite background: outer border strip
        b = max(5, min(h, w) // 25)
        mask[:b, :] = cv2.GC_BGD
        mask[-b:, :] = cv2.GC_BGD
        mask[:, :b] = cv2.GC_BGD
        mask[:, -b:] = cv2.GC_BGD

        # Probable foreground: inner half of the image
        cy, cx = h // 2, w // 2
        mask[cy - h // 4 : cy + h // 4, cx - w // 4 : cx + w // 4] = cv2.GC_PR_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        k = np.ones((20, 20), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)

        quad = self.mask_to_quad(fg)
        if quad is None:
            return image
        return self.draw_polygon(image, quad)
