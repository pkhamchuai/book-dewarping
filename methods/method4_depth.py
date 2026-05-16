import cv2
import numpy as np
from .base import BookEdgeDetector

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class DepthDetector(BookEdgeDetector):
    """
    MiDaS monocular depth estimation (MiDaS_small via torch.hub).
    The book page sits in the near depth plane; thresholding separates it
    from the more distant background surface.
    """
    name = "M4: Depth Estimation\n(MiDaS)"
    _model = None
    _transform = None

    def _load(self):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Missing deps. Run:\n  pip install torch")
        if DepthDetector._model is None:
            DepthDetector._model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            DepthDetector._transform = transforms.small_transform
            DepthDetector._model.eval()

    def detect(self, image: np.ndarray) -> np.ndarray:
        self._load()
        import torch

        # MiDaS expects RGB as numpy H×W×3
        inp = DepthDetector._transform(image)
        with torch.no_grad():
            depth = DepthDetector._model(inp)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().numpy()

        # MiDaS returns inverse depth: larger value = closer to camera
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)

        # Keep near foreground (upper 60 % of depth values = closer objects)
        threshold = np.percentile(depth_norm, 40)
        fg = (depth_norm >= threshold).astype(np.uint8) * 255

        k = np.ones((25, 25), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)

        quad = self.mask_to_quad(fg)
        if quad is None:
            return image

        # Overlay depth map as a tinted background for visual clarity
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(image, 0.55, depth_color, 0.45, 0)

        return self.draw_polygon(blended, quad)
