import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from kernels import create_kernel, kernel_sobel


class SobelFilter(nn.Module):
    def __init__(self, radius=2, scale=2):
        super().__init__()
        self.radius = radius
        self.scale = scale

        # Create sobel kernels and normalise them
        self.kernel_gx, self.kernel_gy = kernel_sobel(radius=radius, scale=scale)
        self.norm_term = (self.kernel_gx.shape[0] * self.kernel_gx.shape[1]) - 1
        self.kernel_gx = self.kernel_gx / self.norm_term
        self.kernel_gy = self.kernel_gy / self.norm_term

        # Shape the kernels for conv2d
        self.padding = (self.kernel_gx.shape[0] - 1) // 2
        self.kernel_gx = torch.Tensor(self.kernel_gx).unsqueeze(0).unsqueeze(0)
        self.kernel_gy = torch.Tensor(self.kernel_gy).unsqueeze(0).unsqueeze(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Same padding
        input = F.pad(input, pad=(self.padding, self.padding, self.padding, self.padding), mode="replicate")

        gx = F.conv2d(
            input,
            weight=self.kernel_gx,
            bias=None,
            padding=0,
            groups=1,
        )
        gy = F.conv2d(
            input,
            weight=self.kernel_gy,
            bias=None,
            padding=0,
            groups=1,
        )

        # Gradient magnitude
        magnitude = torch.sqrt(torch.pow(gx, 2) + torch.pow(gy, 2))
        # gradient_direction = torch.atan2(gy, gx)

        return magnitude


class SoftSpatialCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    This loss allows the targets for the cross entropy loss to be multi-label.
    The labels are smoothed by a spatial gaussian kernel before being normalized.

    NOTE: Only works on channel-first
    """
    def __init__(
        self,
        reduction: str = "mean",
        method: Optional[str] = "max",
        classes: list[int] = [0, 1, 2, 3],
        strength: float = 1.01,
        kernel_radius: float = 1.0,
        kernel_circular: bool = True,
        kernel_sigma: float = 2.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(reduction=reduction)
        assert method in ["half", "max", None], "method must be one of 'half', 'max', or None"
        assert isinstance(classes, list) and len(classes) > 1, "classes must be a list of at least two ints"
        assert classes == sorted(classes), "classes must be sorted in ascending order"

        self._eps = torch.Tensor([torch.finfo(torch.float32).eps]).to(device)
        self.classes_count = len(classes)
        self.classes = torch.Tensor(classes).view(1, -1, 1, 1).to(device)
        self.strength = strength
        self.method = method

        self.kernel_np = create_kernel(
            radius=kernel_radius,
            circular=kernel_circular,
            distance_weighted=True,
            hole=False if method is None else True,
            method=3,
            sigma=kernel_sigma,
            normalised=False
        )
        if method == "half":
            self.kernel_np[self.kernel_np.shape[0] // 2, self.kernel_np.shape[1] // 2] = self.kernel_np.sum() * self.strength

        self.padding = (self.kernel_np.shape[0] - 1) // 2
        self.kernel = torch.Tensor(self.kernel_np).unsqueeze(0).repeat(self.classes_count, 1, 1, 1).to(device)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted_targets = (target == self.classes).float()
        adjusted_targets = F.pad(adjusted_targets, pad=(self.padding, self.padding, self.padding, self.padding), mode="replicate")
        convolved = F.conv2d(
            adjusted_targets,
            weight=self.kernel,
            bias=None,
            padding=self.padding,
            groups=self.classes_count,
        )

        if self.method == "max":
            maxpool = F.max_pool2d(convolved, kernel_size=self.kernel.shape[-1], stride=1, padding=self.padding)
            surroundings = (1 - adjusted_targets) * convolved
            center = ((maxpool * adjusted_targets) * self.strength)
            convolved = center + surroundings

        convolved = convolved[:, :, self.padding:-self.padding, self.padding:-self.padding]
        normalised = convolved / (self._eps + convolved.sum(dim=(1), keepdim=True))

        return super().forward(input, normalised)
