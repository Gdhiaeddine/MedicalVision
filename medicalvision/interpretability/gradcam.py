"""
Grad-CAM: Gradient-weighted Class Activation Mapping for Chest X-ray models.
Visualizes spatial regions influencing the network's prediction for any specified pathology.
"""

from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM implementation using forward and backward hooks on PyTorch models.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "target_conv_layer"):
                self.target_layer = model.target_conv_layer
            else:
                raise ValueError("Target layer must be specified if model does not have 'target_conv_layer'.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generate a 2D normalized Grad-CAM heatmap for the given target class index.
        Args:
            input_tensor: (1, 3, H, W) normalized image tensor.
            target_class_idx: Index of the disease class (0..13).
            target_size: Optional (width, height) to resize heatmap to match original image.
        Returns:
            2D numpy array of shape (H, W) in range [0.0, 1.0].
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        # Target class score
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check target_layer registration.")

        # Global average pooling of gradients: (batch, channels, H, W) -> (batch, channels)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU to keep only features that have a POSITIVE influence on class score
        cam = F.relu(cam)

        # Interpolate to input tensor size or requested target size
        if target_size is not None:
            cam = F.interpolate(cam, size=(target_size[1], target_size[0]), mode="bilinear", align_corners=False)
        else:
            cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> Image.Image:
    """
    Superimpose a normalized 2D heatmap [0, 1] onto a PIL Image.
    Returns:
        RGB PIL Image showing the blended visualization.
    """
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    orig_w, orig_h = original_image.size

    # Ensure heatmap matches image dimensions
    if heatmap.shape != (orig_h, orig_w):
        cam_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        cam_img = cam_img.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
        heatmap = np.array(cam_img) / 255.0

    # Apply Matplotlib colormap (compatible with modern matplotlib >= 3.8 and legacy versions)
    if hasattr(matplotlib, "colormaps"):
        colormap = matplotlib.colormaps[colormap_name]
    else:
        colormap = cm.get_cmap(colormap_name)
    colored_heatmap = colormap(heatmap)[:, :, :3]  # drop alpha, shape (H, W, 3)
    colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(colored_heatmap_uint8)

    # Blend original image with heatmap
    blended = Image.blend(original_image, heatmap_pil, alpha=alpha)
    return blended
