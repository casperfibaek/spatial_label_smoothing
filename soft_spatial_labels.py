import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from kernels import create_kernel, kernel_sobel


class OneHotEncoder2D(nn.Module):
    """
    One hot encodes a 2D tensor.
    """
    def __init__(self, classes: list[int], device: Optional[str] = None) -> None:
        super().__init__()
        assert isinstance(classes, list) and len(classes) > 1, "classes must be a list of at least two ints"
        assert classes == sorted(classes), "classes must be sorted in ascending order"

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.classes_count = len(classes)
        self.classes = torch.Tensor(classes).view(1, -1, 1, 1).to(self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input == self.classes).float()


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


class SoftSpatialCrossEntropyLoss(nn.Module):
    """
    This loss allows the targets for the cross entropy loss to be multi-label.
    The labels are smoothed by a spatial gaussian kernel before being normalized.

    NOTE: Only works on channel-first
    """
    def __init__(
        self,
        method: Optional[str] = "max",
        loss_method: str = "cross_entropy",
        classes: list[int] = [0, 1, 2, 3],
        apply_softmax: bool = True,
        kernel_radius: float = 1.0,
        kernel_circular: bool = True,
        kernel_sigma: float = 2.0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert method in ["half", "max", None], "method must be one of 'half', 'max', or None"
        assert loss_method in ["cross_entropy", "dice", "logcosh_dice", "error", "focal_error", "focal_error_squared"], "loss_method must be one of 'cross_entropy', 'dice', or 'logcosh_dice'"
        assert isinstance(classes, list) and len(classes) > 1, "classes must be a list of at least two ints"
        assert classes == sorted(classes), "classes must be sorted in ascending order"

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.method = method
        self.loss_method = loss_method
        self.apply_softmax = apply_softmax
        self.classes_count = len(classes)
        self.classes = torch.Tensor(classes).view(1, -1, 1, 1).to(self.device)

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
        self.eps = torch.Tensor([1e-07]).to(self.device)
        self.kernel_size = self.kernel.shape[2:].numel()
        self.strength = (1 + self.kernel_size) / self.kernel_size

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Expects input to be of shape [B, C, H, W] and target to be of shape [B, 1, H, W]. """
        adjusted_targets = (target == self.classes).float()
        adjusted_targets_pad = F.pad(adjusted_targets, pad=(self.padding, self.padding, self.padding, self.padding), mode="replicate")
        convolved = F.conv2d(
            adjusted_targets_pad,
            weight=self.kernel,
            bias=None,
            padding=self.padding,
            groups=self.classes_count,
        )

        if self.method == "max":
            maxpool = F.max_pool2d(convolved, kernel_size=self.kernel.shape[-1], stride=1, padding=self.padding)
            surroundings = (1 - adjusted_targets_pad) * convolved
            center = ((maxpool * adjusted_targets_pad) * self.strength)
            convolved = center + surroundings

        convolved = convolved[:, :, self.padding:-self.padding, self.padding:-self.padding]

        if self.apply_softmax:
            target_smooth = torch.softmax(convolved, dim=1)
        else:
            target_smooth = convolved / (self.eps + convolved.sum(dim=(1), keepdim=True))

        # Compute the cross entropy
        if self.loss_method == "cross_entropy":
            loss = -torch.sum(target_smooth * torch.log(output + self.eps)) / torch.numel(output)
        elif self.loss_method == "dice":
            intersection = torch.sum(output * target_smooth, dim=(0, 2, 3))
            cardinality = torch.sum(output + target_smooth, dim=(0, 2, 3))

            loss = 1 - (2. * intersection / (cardinality + self.eps)).mean()
        elif self.loss_method == "logcosh_dice":
            intersection = torch.sum(output * target_smooth, dim=(0, 2, 3))
            cardinality = torch.sum(output + target_smooth, dim=(0, 2, 3))

            loss = torch.log(torch.cosh(1 - torch.mean((intersection + self.eps) / (cardinality + self.eps))))
        elif self.loss_method == "error":
            loss = torch.mean(torch.sum(torch.abs(output - target_smooth), dim=1) / 2.0)
        elif self.loss_method == "focal_error":
            target_focal = adjusted_targets * target_smooth
            output_focal = adjusted_targets * output
            output_focal_adjusted = torch.where(output_focal > target_focal, target_focal, output_focal)

            loss = torch.mean(torch.abs(target_focal - output_focal_adjusted)) * self.classes_count
        elif self.loss_method == "focal_error_squared":
            target_focal = adjusted_targets * target_smooth
            output_focal = adjusted_targets * output
            output_focal_adjusted = torch.where(output_focal > target_focal, target_focal, output_focal)

            loss = torch.mean(torch.pow(target_focal - output_focal_adjusted, 2.0)) * self.classes_count
        else:
            raise ValueError("loss_method must be one of 'cross_entropy' or 'dice'")

        return loss

