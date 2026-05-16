import cv2
import numpy as np
from .base import BookEdgeDetector

try:
    import torch
    from transformers import Sam2Model, Sam2Processor
    from PIL import Image as PILImage
    _SAM2_AVAILABLE = True
except ImportError:
    _SAM2_AVAILABLE = False


class SAMDetector(BookEdgeDetector):
    """
    SAM2 (facebook/sam2-hiera-base-plus via HuggingFace transformers).
    Single center-point prompt isolates the book page from the background.
    """
    name = "M3: SAM2\n(Segment Anything 2)"
    _model = None
    _processor = None

    def _load(self):
        if not _SAM2_AVAILABLE:
            raise RuntimeError("Missing deps. Run: pip install torch transformers")
        if SAMDetector._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            SAMDetector._processor = Sam2Processor.from_pretrained(
                "facebook/sam2-hiera-base-plus"
            )
            SAMDetector._model = Sam2Model.from_pretrained(
                "facebook/sam2-hiera-base-plus"
            ).to(device)
            SAMDetector._model.eval()

    def detect(self, image: np.ndarray) -> np.ndarray:
        self._load()
        import torch

        device = next(SAMDetector._model.parameters()).device
        h, w = image.shape[:2]
        pil_img = PILImage.fromarray(image)

        # SAM2 requires 4-level nesting: [image, object, point, coords]
        # and 3-level labels:            [image, object, point]
        input_points = [[[[w // 2, h // 2]]]]
        input_labels = [[[1]]]  # 1 = foreground

        inputs = SAMDetector._processor(
            pil_img,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = SAMDetector._model(**inputs)

        # pred_masks: (batch=1, point_batch=1, num_masks, H_enc, W_enc)
        # iou_scores: (batch=1, point_batch=1, num_masks)
        scores = outputs.iou_scores[0, 0]  # (num_masks,)

        # post_process_masks expects a list of tensors (one per image)
        # each tensor shape: (point_batch, num_masks, H_enc, W_enc)
        masks_list = [outputs.pred_masks[0].cpu()]
        original_sizes = inputs["original_sizes"].cpu().tolist()

        processed = SAMDetector._processor.post_process_masks(
            masks_list, original_sizes
        )
        # processed[0]: bool tensor (point_batch=1, num_masks, H, W)
        best_idx = scores.argmax().item()
        best_mask = processed[0][0, best_idx].numpy().astype(np.uint8) * 255

        k = np.ones((15, 15), np.uint8)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, k)

        quad = self.mask_to_quad(best_mask)
        if quad is None:
            return image
        return self.draw_polygon(image, quad)
