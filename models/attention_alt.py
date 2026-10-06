"""Alternative attention modules for Phase 3.2 (controls vs ECA+SpatialAttn).

SE, CBAM and Coordinate Attention blocks, inserted at the SAME three points
(P3/P4/P5, IAMA-yaml layers 5/8/13) with the SAME identity-initialisation
principle as ECA_SpatialAttn: final sigmoid gate(s) get zero weights and a
positive bias so the block starts as a near-identity (gate ~0.98 per stage,
cascaded product ~0.964 where two gates apply).

IMPORTANT: unlike ECA (whose Conv1d(1,1,k) is channel-agnostic), these blocks
allocate channel-dependent weights, so their yaml args must carry the TRUE
post-width-scaling channel counts (P3=256, P4=256, P5=512 for YOLO11s),
measured with a forward probe (see results_v3/audit/param_counts.csv).
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018), 1x1-conv implementation."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # identity init: gate = sigmoid(0 + 4.0) = 0.982
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 4.0)

    def forward(self, x):
        z = x.mean(dim=(2, 3), keepdim=True)
        s = torch.relu(self.fc1(z))
        g = torch.sigmoid(self.fc2(s))
        return x * g


class CBAMBlock(nn.Module):
    """CBAM (Woo et al., 2018): shared-MLP channel attention + 7x7 spatial."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp0 = nn.Conv2d(channels, hidden, 1)
        self.mlp1 = nn.Conv2d(hidden, channels, 1)
        nn.init.kaiming_uniform_(self.mlp0.weight)
        nn.init.zeros_(self.mlp0.bias)
        # identity init: channel gate = sigmoid(2.0 + 2.0) = 0.982
        nn.init.zeros_(self.mlp1.weight)
        nn.init.constant_(self.mlp1.bias, 2.0)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                      padding=spatial_kernel // 2, bias=True)
        nn.init.zeros_(self.spatial_conv.weight)
        nn.init.constant_(self.spatial_conv.bias, 4.0)

    def forward(self, x):
        avg = x.mean(dim=(2, 3), keepdim=True)
        mx = x.amax(dim=(2, 3), keepdim=True)
        ca = torch.sigmoid(self.mlp1(torch.relu(self.mlp0(avg)))
                           + self.mlp1(torch.relu(self.mlp0(mx))))
        x1 = x * ca
        s_avg = x1.mean(dim=1, keepdim=True)
        s_max = x1.amax(dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([s_avg, s_max], dim=1)))
        return x1 * sa


class CABlock(nn.Module):
    """Coordinate Attention (Hou et al., 2021), simplified standard form."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)
        for c in (self.conv_h, self.conv_w):
            # identity init: each gate = sigmoid(4.0) = 0.982; product 0.964
            nn.init.zeros_(c.weight)
            nn.init.constant_(c.bias, 4.0)

    def forward(self, x):
        B, C, H, W = x.shape
        # coordinate descriptors
        x_h = x.mean(dim=3, keepdim=True)                    # (B,C,H,1)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)    # (B,C,W,1)
        y = torch.cat([x_h, x_w], dim=2)                     # (B,C,H+W,1)
        y = self.act(self.bn(self.reduce(y)))
        y_h, y_w = y.split([H, W], dim=2)
        g_h = torch.sigmoid(self.conv_h(y_h))                # (B,C,H,1)
        g_w = torch.sigmoid(self.conv_w(y_w.transpose(2, 3)))  # (B,C,1,W)
        return x * g_h * g_w
