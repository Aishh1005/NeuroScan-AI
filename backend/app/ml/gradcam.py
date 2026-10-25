"""
Real Grad-CAM (Selvaraju et al., 2017) implementation using forward/backward
hooks on the target convolutional layer. No heatmap is faked: the CAM is
computed from actual gradients of the predicted class score with respect to
the chosen layer's activations.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=False)

        gradients = self.gradients[0]       # (C, H, W)
        activations = self.activations[0]   # (C, H, W)

        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam  # normalized 0..1, shape (H_feat, W_feat)


def overlay_heatmap_on_image(cam: np.ndarray, original_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """original_rgb: HxWx3 uint8 RGB image. Returns HxWx3 uint8 RGB overlay."""
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap.astype(np.float32) * alpha + original_rgb.astype(np.float32) * (1 - alpha))
    return np.clip(overlay, 0, 255).astype(np.uint8)
