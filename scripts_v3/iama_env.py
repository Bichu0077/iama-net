"""Shared environment setup for v3 scripts.

Ensures the repo root is importable and registers the custom IAMA attention
modules with ultralytics so that configs/yolo11s_iama.yaml can be parsed and
checkpoints containing ECA_SpatialAttn can be unpickled.

AUDIT FINDING (2026-10-05): the original training venv patched ultralytics
site-packages (class pickled as `ultralytics.nn.modules.block.ECA_SpatialAttn`)
and that patched class is LOST. The checkpoints store submodules
`eca` (an ECA instance with `eca.conv` Conv1d) and `spatial_conv` (a raw
nn.Conv2d(2, 1, 7, 7)). This does NOT match models/attention.py, whose
ECA_SpatialAttn uses a `spatial` SpatialAttn wrapper. Parameter counts are
identical (105 per block); the only difference is code organisation.

ECA_SpatialAttnCompat below reproduces the checkpoint structure exactly:
ECA followed by CBAM-style spatial attention implemented with a raw 7x7
Conv2d over concat(channel-mean, channel-max). Correctness of the assumed
forward semantics is verified empirically: re-evaluating the ANCHOR and M3
checkpoints must reproduce the originally reported 0.7608 / 0.7484 mAP@0.5
(see results_v3/audit/reconciliation_neu.csv).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.attention import ECA  # noqa: E402


class ECA_SpatialAttnCompat(nn.Module):
    """Checkpoint-compatible ECA + spatial attention block.

    Submodule names match the pickled state dicts: `eca`, `spatial_conv`.
    Constructor signature matches yaml parsing: ECA_SpatialAttn(channels).
    """

    def __init__(self, channels: int, spatial_kernel: int = 7):
        super().__init__()
        self.eca = ECA(channels)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                      padding=spatial_kernel // 2, bias=True)
        nn.init.zeros_(self.spatial_conv.weight)
        nn.init.constant_(self.spatial_conv.bias, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


# Expose under the original pickled name
ECA_SpatialAttn = ECA_SpatialAttnCompat


def register_iama_modules():
    """Inject compat attention classes into ultralytics namespaces.

    The checkpoints pickle the class as
    ``ultralytics.nn.modules.block.ECA_SpatialAttn``; parse_model resolves
    yaml module names from ``ultralytics.nn.tasks`` globals. We inject into
    all three namespaces at runtime (no site-packages modification).
    Phase 3.2 alternative attention blocks (SE/CBAM/CA) are injected too.
    """
    import ultralytics.nn.tasks as T
    import ultralytics.nn.modules as M
    import ultralytics.nn.modules.block as B
    from models.attention_alt import SEBlock, CBAMBlock, CABlock

    for mod in (T, M, B):
        for cls in (ECA, ECA_SpatialAttnCompat, SEBlock, CBAMBlock, CABlock):
            name = "ECA_SpatialAttn" if cls is ECA_SpatialAttnCompat else cls.__name__
            setattr(mod, name, cls)
    return T


# Dataset roots / model registry used across v3 scripts
NEU_RAW = PROJECT_ROOT / "data" / "NEU-DET"
NEU_PP = PROJECT_ROOT / "data" / "NEU-DET_preprocessed"
NEU_GRAY = PROJECT_ROOT / "data" / "NEU-DET_gray_clahe"
NEU_NOBIL = PROJECT_ROOT / "data" / "NEU-DET_lab_nobil"
GC10_RAW = PROJECT_ROOT / "data" / "GC10-DET"
GC10_PP = PROJECT_ROOT / "data" / "GC10-DET_preprocessed_v3"

AUDIT_DIR = PROJECT_ROOT / "results_v3" / "audit"
PRED_DIR = AUDIT_DIR / "predictions"

# Original checkpoints (read-only)
CHECKPOINTS = {
    "M1": PROJECT_ROOT / "runs" / "M1_baseline_seed42_20260924_195541" / "weights" / "best.pt",
    "M2": PROJECT_ROOT / "runs" / "M2_aligned_seed42" / "weights" / "best.pt",
    "M3": PROJECT_ROOT / "runs" / "M3_aligned_seed42" / "weights" / "best.pt",
    "M4": PROJECT_ROOT / "runs" / "M4_aligned_seed42" / "weights" / "best.pt",
    "ANCHOR": PROJECT_ROOT / "runs" / "m3_aligned_init_seed42.pt",
}

NEU_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
GC10_NAMES = ["crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole",
              "rolled_pit", "silk_spot", "waist_folding", "water_spot", "welding_line"]
