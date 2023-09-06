import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class SoftSpatialSegmentationLoss(nn.Module):
    """
    This loss allows the targets for the cross entropy loss to be multi-label.
    The labels are smoothed by a spatial gaussian kernel before being normalized.

    Input is expected to be of shape [B, C, H, W] and target is expected to be class integers of the shape [B, 1, H, W].

    Parameters
    ----------
    method : str, optional
        The method to use for smoothing the labels. One of 'half', 'max', or None.
        By setting a method, you ensure that the center pixel will never 'flip' to another class.
        Default is 'half'.
        - `'half'` will set the pixel to at least half the sum of the weighted classes for the pixel.
        - `'kernel_half'` will set the center of the kernel to be weighted as half the sum of the surrounding weighted classes. Multiplied by `(1 + self.kernel_size) / self.kernel_size`.
        - `'max'` will set the center of the kernel to be weighted as the maximum of the surrounding weighted classes. Multiplied by `(1 + self.kernel_size) / self.kernel_size`. 
        - `None` will not treat the center pixel differently. Does not ensure that the center pixel will not 'flip' to another class.

    classes : list[int]
        A sorted (ascending) list of the classes to use for the loss.
        Should correspond to the classes in the target. I.e. for ESA WorldCover it should be
        `[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]`.

    device : str, optional
        The device to use for the computations. Default is 'cuda' if available, else 'cpu'.

    channel_last : bool, optional
        Whether the input is channel last or not. Default is False.
    """
    def __init__(
        self,
        loss_func: Callable,
        method: Optional[str] = "half",
        classes: Optional[list[int]] = None,
        device: Optional[str] = None,
        channel_last: bool = False,
    ) -> None:
        super().__init__()
        assert method in ["half", "kernel_half", "max", None], \
            "method must be one of 'half', 'kernel_half', 'max', or None"
        assert isinstance(classes, list) and len(classes) > 1, "classes must be a list of at least two ints"
        assert classes == sorted(classes), "classes must be sorted in ascending order"

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.loss_func = loss_func
        self.method = method
        self.channel_last = channel_last

        # Precalculate the classes and reshape them for broadcasting
        self.classes_count = len(classes)
        self.classes = torch.Tensor(classes).view(1, -1, 1, 1).to(self.device)

        if self.method in [None, "max"]:
            self.kernel_np = torch.Tensor([
                [0.4049764, 0.7942472, 0.4049764],
                [0.7942472, 1.       , 0.7942472],
                [0.4049764, 0.7942472, 0.4049764]], dtype="float32")
        else:
            self.kernel_np = torch.Tensor([
                [0.4049764, 0.7942472, 0.4049764],
                [0.7942472, 0.       , 0.7942472],
                [0.4049764, 0.7942472, 0.4049764]], dtype="float32")

        # To ensure that the classes are not tied in probability, we multiply the center of the kernel by a factor
        # E.g. if the kernel is 5x5, then the center pixel will be multiplied by 25 / (25 - 1) = 1.04
        self.kernel_size = self.kernel_np.size
        self.kernel_width = self.kernel_np.shape[1]
        self.strength = self.kernel_size / (self.kernel_size - 1.0)

        # Set the center of the kernel to be at least half the sum of the surrounding weighted classes
        if method == "kernel_half":
            self.kernel_np[self.kernel_np.shape[0] // 2, self.kernel_np.shape[1] // 2] = self.kernel_np.sum() * self.strength

        # Padding for conv2d
        self.padding = (self.kernel_np.shape[0] - 1) // 2

        # Turn the 2D kernel into a 4D kernel for conv2d
        self.kernel = torch.Tensor(self.kernel_np).unsqueeze(0).repeat(self.classes_count, 1, 1, 1).to(device)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Expects input to be of shape [B, C, H, W] or [B, H, W, C] if channel_last is True. """
        if self.channel_last:
            output = output.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

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

        # Handle the flip protection methods
        if self.method == "max":
            valmax, argmax = torch.max(convolved, dim=1, keepdim=True)
            _, argmax_hot = torch.max(target_hot_pad, dim=1, keepdim=True)

            weight = torch.where(argmax == argmax_hot, convolved, valmax * self.strength)
            convolved = torch.where(target_hot_pad == 1, weight, convolved)

        elif self.method == "half":
            weight = self.kernel_np.sum() * self.strength
            convolved = torch.where(target_hot_pad == 1, weight, convolved)

        # No handling necessary for these two methods
        elif self.method == "kernel_half" or self.method is None:
            pass

        # Remove the padding
        convolved = convolved[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # The output is not normalised, so we can either apply softmax or normalise it using the sum.
        target_soft = convolved / torch.maximum(convolved.sum(dim=(1), keepdim=True), self.eps)

        if self.channel_last:
            output = output.permute(0, 2, 3, 1)
            target_soft = target_soft.permute(0, 2, 3, 1)

        # Calculate the loss using the smoothed targets.
        return self.loss_func(output, target_soft)
