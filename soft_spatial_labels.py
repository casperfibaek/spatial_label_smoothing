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


def _calculate_loss(
    output,
    target_hot,
    target_soft,
    loss_method,
    eps=1e-07,
) -> torch.Tensor:
    """
    Calculates the loss using the given method. The methods here are compatible with the
    SoftSegmentationLoss and SoftSpatialSegmentation classes.
    """
    if loss_method == "cross_entropy":
        loss = -torch.sum(target_soft * torch.log(output + eps)) / torch.numel(output)

    elif loss_method == "dice":
        intersection = torch.sum(output * target_soft, dim=(0, 2, 3))
        cardinality = torch.sum(output + target_soft, dim=(0, 2, 3))
        loss = 1 - (2. * intersection / (cardinality + eps)).mean()

    elif loss_method == "logcosh_dice":
        intersection = torch.sum(output * target_soft, dim=(0, 2, 3))
        cardinality = torch.sum(output + target_soft, dim=(0, 2, 3))
        loss = torch.log(torch.cosh(1 - torch.mean((intersection + eps) / (cardinality + eps))))

    elif loss_method == "error":
        loss = torch.mean(torch.sum(torch.abs(output - target_soft), dim=1) / 2.0)

    elif loss_method == "focal_error":
        target_focal = target_hot * target_soft
        output_focal = target_hot * output
        output_focal_adjusted = torch.where(output_focal > target_focal, target_focal, output_focal)
        loss = torch.mean(torch.abs(target_focal - output_focal_adjusted)) * target_hot.shape[1]

    elif loss_method == "focal_error_squared":
        target_focal = target_hot * target_soft
        output_focal = target_hot * output
        output_focal_adjusted = torch.where(output_focal > target_focal, target_focal, output_focal)
        loss = torch.mean(torch.pow(target_focal - output_focal_adjusted, 2.0)) * target_hot.shape[1]

    elif loss_method == "kl_divergence":
        loss = F.kl_div(F.log_softmax(output, dim=1), target_soft, reduction="batchmean")

    elif loss_method == "nll":
        loss = F.nll_loss(F.log_softmax(output, dim=1), target_soft)

    elif loss_method == "nll_poisson":
        if torch.is_tensor(eps):
            eps = eps.item()
        loss = F.poisson_nll_loss(output, target_soft, log_input=True, full=False, eps=eps, reduction="mean")

    else:
        raise ValueError("loss_method must be one of 'cross_entropy' or 'dice'")
    
    return loss


class SoftSegmentationLoss(nn.Module):
    """
    This loss allows the targets for the cross entropy loss to be multi-label.
    The labels are smoothed before being normalized using a global strategy.

    Input is expected to be of shape [B, C, H, W] and target is expected to be class integers of the shape [B, 1, H, W].

    Parameters
    ----------
    smoothing : float, optional
        The smoothing factor to use. Default is 0.1.

    loss_method : str, optional
        loss_method must be one of `'cross_entropy'`, `'dice'`, `'logcosh_dice'`, `'error'`, `'focal_error'`, `'focal_error_squared'`, `'kl_divergence'`, `'nll'`, or `'nll_poisson'`.
        Default is 'cross_entropy'.

    device : str, optional
        The device to use for the computations. Default is 'cuda' if available, else 'cpu'.
    
    epsilon : float, optional
        A small value to add to the denominator to avoid division by zero. Default is 1e-07.
    """
    def __init__(self,
        smoothing:float = 0.1,
        loss_method: str = "cross_entropy",
        classes: Optional[list[int]] = None,
        device: Optional[str] = None,
        epsilon: float = 1e-07,
    ) -> None:
        super().__init__()
        assert isinstance(classes, list) and len(classes) > 1, "classes must be a list of at least two ints"
        assert classes == sorted(classes), "classes must be sorted in ascending order"
        assert isinstance(smoothing, (float, int)), "smoothing must be a float or an int"
        assert smoothing >= 0.0 and smoothing <= 1.0, "smoothing must be between 0.0 and 1.0"
        assert loss_method in ["cross_entropy", "dice", "logcosh_dice", "error", "focal_error", "focal_error_squared", "kl_divergence", "nll", "nll_poisson"], \
            "loss_method must be one of 'cross_entropy', 'dice', 'logcosh_dice', 'error', 'focal_error', 'focal_error_squared', 'kl_divergence', 'nll', or 'nll_poisson'."

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.smoothing = smoothing
        self.loss_method = loss_method
        self.eps = torch.Tensor([epsilon]).to(self.device)

        self.classes_count = len(classes)
        self.classes = torch.Tensor(classes).view(1, -1, 1, 1).to(self.device)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Expects input to be of shape [B, C, H, W] and target to be of shape [B, 1, H, W]. """
        # One-hot encode the target and cast to float
        target_hot = (target == self.classes).float()

        # Smooth the target
        target_soft = ((1 - self.smoothing) * target_hot) + (self.smoothing / target_hot.shape[1])

        # Calculate the loss using the smoothed targets.
        loss = _calculate_loss(output, target_hot, target_soft, self.loss_method, self.eps)

        return loss


class SoftSpatialSegmentationLoss(nn.Module):
    """
    This loss allows the targets for the cross entropy loss to be multi-label.
    The labels are smoothed by a spatial gaussian kernel before being normalized.

    Input is expected to be of shape [B, C, H, W] and target is expected to be class integers of the shape [B, 1, H, W].

    NOTE: Only works on channel-first tensors.

    Parameters
    ----------
    method : str, optional
        The method to use for smoothing the labels. One of 'half', 'max', or None.
        By setting a method, you ensure that the center pixel will never 'flip' to another class.
        Default is 'max'.
        - `'half'` will set the center of the kernel to at least half the sum of the surrounding weighted classes.
        - `'max'` will set the center of the kernel to be weighted as the maximum of the surrounding weighted classes. Multiplied by `(1 + self.kernel_size) / self.kernel_size`. 
        - `None` will not treat the center pixel differently. Does not ensure that the center pixel will not 'flip' to another class.

    loss_method : str, optional
        loss_method must be one of `'cross_entropy'`, `'dice'`, `'logcosh_dice'`, `'error'`, `'focal_error'`, `'focal_error_squared'`, `'kl_divergence'`, `'nll'`, or `'nll_poisson'`.
        Default is 'cross_entropy'.
    
    classes : list[int]
        A sorted (ascending) list of the classes to use for the loss.
        Should correspond to the classes in the target. I.e. for ESA WorldCover it should be
        `[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]`.

    kernel_radius : float, optional
        The radius of the gaussian kernel. Default is 1.0. Can be fractional.

    kernel_circular : bool, optional
        Whether to use a circular kernel or not. Default is True.

    kernel_sigma : float, optional
        The sigma of the gaussian kernel. Default is 2.0. Can be fractional.

    epsilon : float, optional
        A small value to add to the denominator to avoid division by zero. Default is 1e-07.

    device : str, optional
        The device to use for the computations. Default is 'cuda' if available, else 'cpu'.   

    Returns
    -------
    loss : torch.Tensor
        The computed loss.
    """
    def __init__(
        self,
        method: Optional[str] = "max",
        loss_method: str = "cross_entropy",
        classes: Optional[list[int]] = None,
        kernel_radius: float = 1.0,
        kernel_circular: bool = True,
        kernel_sigma: float = 2.0,
        epsilon: float = 1e-07,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert method in ["half", "max", None], "method must be one of 'half', 'max', or None"
        assert loss_method in ["cross_entropy", "dice", "logcosh_dice", "error", "focal_error", "focal_error_squared", "kl_divergence", "nll", "nll_poisson"], \
            "loss_method must be one of 'cross_entropy', 'dice', 'logcosh_dice', 'error', 'focal_error', 'focal_error_squared', 'kl_divergence', 'nll', or 'nll_poisson'."
        assert isinstance(classes, list) and len(classes) > 1, "classes must be a list of at least two ints"
        assert classes == sorted(classes), "classes must be sorted in ascending order"

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.method = method
        self.loss_method = loss_method
        self.apply_softmax = False # Kept for testing purposes
        self.eps = torch.Tensor([epsilon]).to(self.device)

        # Precalculate the classes and reshape them for broadcasting
        self.classes_count = len(classes)
        self.classes = torch.Tensor(classes).view(1, -1, 1, 1).to(self.device)

        # Calculate the kernel using numpy. It is slower, but it is only done once.
        self.kernel_np = create_kernel(
            radius=kernel_radius,
            circular=kernel_circular,
            distance_weighted=True,
            hole=False if method is None else True,
            method=3,
            sigma=kernel_sigma,
            normalised=False
        )

        # To ensure that the classes are not tied in probability, we multiply the center of the kernel by a factor
        # E.g. if the kernel is 5x5, then the center pixel will be multiplied by 25 / (25 - 1) = 1.04
        self.kernel_size = self.kernel_np.size
        self.kernel_width = self.kernel_np.shape[1]
        self.strength = self.kernel_size / (self.kernel_size - 1.0)

        # Set the center of the kernel to be at least half the sum of the surrounding weighted classes
        if method == "half":
            self.kernel_np[self.kernel_np.shape[0] // 2, self.kernel_np.shape[1] // 2] = self.kernel_np.sum() * self.strength

        # Padding for conv2d
        self.padding = (self.kernel_np.shape[0] - 1) // 2

        # Turn the 2D kernel into a 4D kernel for conv2d
        self.kernel = torch.Tensor(self.kernel_np).unsqueeze(0).repeat(self.classes_count, 1, 1, 1).to(device)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Expects input to be of shape [B, C, H, W] and target to be of shape [B, 1, H, W]. """
        # One-hot encode the target and cast to float
        target_hot = (target == self.classes).float()

        # Pad the targets using the same padding as the kernel and 'same' padding
        target_hot_pad = F.pad(target_hot, pad=(self.padding, self.padding, self.padding, self.padding), mode="replicate")
        
        # Convolve the padded targets with the custom kernel
        convolved = F.conv2d(
            target_hot_pad,
            weight=self.kernel,
            bias=None,
            padding=self.padding,
            groups=self.classes_count,
        )

        # If the method is 'max', then we need to find the maximum weighted class in the convolved tensor
        if self.method == "max":
            maxpool = F.max_pool2d(convolved, kernel_size=self.kernel_width, stride=1, padding=self.padding)
            surroundings = (1 - target_hot_pad) * convolved
            center = ((maxpool * target_hot_pad) * self.strength)
            convolved = center + surroundings

        # Remove the padding
        convolved = convolved[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # The output is not normalised, so we can either apply softmax or normalise it using the sum.
        # Since it is common to apply softmax to the output of a segmentation model, we also apply it here.
        if self.apply_softmax:
            target_soft = torch.softmax(convolved, dim=1)
        else:
            target_soft = convolved / (self.eps + convolved.sum(dim=(1), keepdim=True))

        # Calculate the loss using the smoothed targets.
        loss = _calculate_loss(output, target_hot, target_soft, self.loss_method, self.eps)

        return loss
