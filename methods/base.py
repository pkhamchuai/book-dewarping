import cv2
import numpy as np


class BookEdgeDetector:
    name = "Base Detector"
    line_color = (0, 220, 0)      # green outline
    corner_color = (255, 50, 50)  # red corners

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: H×W×3 RGB uint8 array
        Returns:
            H×W×3 RGB uint8 array with detected edges/corners drawn
        """
        raise NotImplementedError

    def safe_detect(self, image: np.ndarray) -> np.ndarray:
        try:
            return self.detect(image)
        except Exception as e:
            result = image.copy()
            msg = str(e)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, chunk in enumerate([msg[j:j+45] for j in range(0, len(msg), 45)][:4]):
                cv2.putText(result, chunk, (20, 50 + i * 35), font, 0.7, (255, 60, 60), 2)
            return result

    def draw_polygon(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        result = image.copy()
        pts = corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(result, [pts], isClosed=True, color=self.line_color, thickness=3)
        for pt in corners:
            cv2.circle(result, tuple(pt.astype(int)), 12, self.corner_color, -1)
            cv2.circle(result, tuple(pt.astype(int)), 12, (255, 255, 255), 2)
        return result

    def mask_to_quad(self, mask: np.ndarray) -> np.ndarray | None:
        """Largest contour → minimum-area rectangle → 4 corner points."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 1000:
            return None
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)
