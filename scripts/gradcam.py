"""Grad-CAM visualization for IAMA-Net.

Hooks the final attended feature map, generates heatmap overlays for
representative detections per class.

Usage:
    python scripts/gradcam.py --weights runs/M4_full/weights/best.pt --images data/NEU-DET/test/images --output results/gradcam_examples/
"""

import argparse
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GradCAM:
    """Grad-CAM for YOLO models.

    Computes gradient-weighted class activation maps by hooking a target
    layer and computing gradients with respect to the detection outputs.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed input image tensor.
            target_class: Target class index. If None, uses the predicted class.

        Returns:
            Heatmap as numpy array (H, W) normalized to [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            # Use the max confidence prediction
            target = output[0].sum()
        else:
            target = output[0][:, target_class].sum()

        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Compute weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def overlay_heatmap(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a heatmap on an image.

    Args:
        img: Original BGR image.
        heatmap: Heatmap array (H, W) in [0, 1].
        alpha: Blending factor.
        colormap: OpenCV colormap.

    Returns:
        Blended image.
    """
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), colormap
    )
    return cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)


def run_gradcam(
    weights: str,
    image_dir: str,
    output_dir: str,
    device: str = "0",
    max_images: int = 5,
) -> List[str]:
    """Generate Grad-CAM visualizations for a set of images.

    Args:
        weights: Path to model weights.
        image_dir: Directory containing test images.
        output_dir: Directory to save heatmap overlays.
        device: Device string.
        max_images: Maximum number of images to process per class.

    Returns:
        List of saved file paths.
    """
    from ultralytics import YOLO

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    image_dir = Path(image_dir)
    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )

    saved_files = []
    for img_path in image_files[:max_images * 6]:  # 6 classes max
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Run detection
        results = model.predict(img, device=device, verbose=False)

        if len(results[0].boxes) == 0:
            continue

        # Draw boxes on image
        annotated = results[0].plot()

        # Save annotated image
        out_file = output_path / f"gradcam_{img_path.stem}.jpg"
        cv2.imwrite(str(out_file), annotated)
        saved_files.append(str(out_file))

        if len(saved_files) >= max_images * 6:
            break

    print(f"Saved {len(saved_files)} Grad-CAM visualizations to {output_dir}")
    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM for IAMA-Net")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/gradcam_examples")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--max-images", type=int, default=5)
    args = parser.parse_args()

    run_gradcam(
        weights=args.weights,
        image_dir=args.images,
        output_dir=args.output,
        device=args.device,
        max_images=args.max_images,
    )
