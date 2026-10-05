"""ECA and Spatial Attention modules for IAMA-Net.

These modules are applied on each of the multi-scale feature maps (F3, F4, F5)
produced by the YOLOv11s backbone. They are additional to the built-in C2PSA
block which only operates at the deepest scale.

ECA: Efficient Channel Attention via 1D convolution.
SpatialAttn: CBAM-style spatial attention via channel pooling + 7x7 conv.
"""

import torch
import torch.nn as nn
import math


class ECA(nn.Module):
    """Efficient Channel Attention module.

    Global-average-pools the feature map to a channel descriptor, passes it
    through a 1D convolution with adaptive kernel size, applies sigmoid, then
    performs channel-wise multiplication.

    Args:
        channels: Number of input channels.
        gamma: Kernel size mapping parameter (default: 2).
        b: Kernel size mapping bias (default: 1).
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size: k = |log2(C)/gamma + b/gamma|_odd
        k = int(abs(math.log2(channels) / gamma + b / gamma))
        k = k if k % 2 else k + 1  # Ensure odd
        k = max(k, 3)  # Minimum kernel size of 3

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=True)
        # Identity initialization: sigmoid(3.0) ~ 0.95 to preserve pretrained feature magnitude
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 3.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Channel-attended tensor of same shape.
        """
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.sigmoid(self.conv(y))  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


class SpatialAttn(nn.Module):
    """CBAM-style Spatial Attention module.

    Channel-wise avg-pool and max-pool the feature map, concatenate,
    pass through a 7x7 conv, sigmoid, then element-wise multiply.

    Args:
        kernel_size: Convolution kernel size (default: 7).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=True
        )
        # Identity initialization: sigmoid(3.0) ~ 0.95 to preserve pretrained feature magnitude
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 3.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Spatially-attended tensor of same shape.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class ECA_SpatialAttn(nn.Module):
    """Combined ECA + Spatial Attention block.

    Applies ECA (channel attention) followed by SpatialAttn (spatial attention)
    sequentially. This is the attention block inserted after each P3/P4/P5
    feature map in IAMA-Net.

    Args:
        channels: Number of input channels (passed to ECA).
        spatial_kernel: Kernel size for spatial attention conv (default: 7).
    """

    def __init__(self, channels: int, spatial_kernel: int = 7):
        super().__init__()
        self.eca = ECA(channels)
        self.spatial = SpatialAttn(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ECA then SpatialAttn."""
        x = self.eca(x)
        x = self.spatial(x)
        return x
