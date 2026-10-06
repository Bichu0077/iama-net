"""Phase 1.2: recount parameters of M1 (baseline) and M3/M4 (IAMA) models.

Outputs results_v3/audit/param_counts.csv and prints:
  - total / trainable params of every checkpoint and of the yaml-built models
  - attention-module param breakdown (ECA conv1d, spatial conv2d) per insertion
  - actual ECA kernel sizes and actual input channels at each insertion point
    (measured with a forward probe, not from the yaml text)

Note discovered during audit: original checkpoints pickle attention blocks with
submodule name 'spatial_conv' (site-packages-patched ultralytics class), while
models/attention.py names it 'spatial'. Both are handled here.
"""

import csv
import sys
from pathlib import Path

from iama_env import register_iama_modules, CHECKPOINTS, AUDIT_DIR, PROJECT_ROOT

register_iama_modules()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def get_spatial(layer):
    for attr in ("spatial", "spatial_conv"):
        if hasattr(layer, attr):
            return attr, getattr(layer, attr)
    raise AttributeError(f"no spatial submodule on {type(layer).__name__}: "
                         f"{[n for n, _ in layer.named_children()]}")


def attention_breakdown(model):
    rows = []
    for i, layer in enumerate(model.model):
        if type(layer).__name__ not in ("ECA_SpatialAttn", "ECA_SpatialAttnCompat"):
            continue
        eca = layer.eca
        sp_name, sp = get_spatial(layer)
        sp_conv = sp if isinstance(sp, nn.Conv2d) else sp.conv
        k = int(eca.conv.weight.shape[-1])
        eca_params = sum(p.numel() for p in eca.parameters())
        sp_params = sum(p.numel() for p in sp.parameters())
        rows.append({
            "layer_index": i,
            "spatial_attr_name": sp_name,
            "eca_kernel_size": k,
            "eca_params": int(eca_params),
            "spatial_kernel": int(sp_conv.weight.shape[-1]),
            "spatial_params": int(sp_params),
            "block_params": int(eca_params + sp_params),
            "eca_bias_value": float(eca.conv.bias.detach().flatten()[0]),
            "spatial_bias_value": float(sp_conv.bias.detach().flatten()[0]),
            "eca_weight_absmax": float(eca.conv.weight.detach().abs().max()),
            "spatial_weight_absmax": float(sp_conv.weight.detach().abs().max()),
        })
    return rows


def probe_input_channels(model):
    """Forward-probe actual input channel counts at each attention layer."""
    import numpy as np
    chans = {}
    hooks = []
    for i, layer in enumerate(model.model.model):
        if type(layer).__name__ in ("ECA_SpatialAttn", "ECA_SpatialAttnCompat"):
            def make_hook(idx):
                def hook(mod, inp, out):
                    chans[idx] = int(inp[0].shape[1])
                return hook
            hooks.append(layer.register_forward_hook(make_hook(i)))
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=640, device="cpu", verbose=False)
    for h in hooks:
        h.remove()
    return chans


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_rows = []

    for tag, yaml_rel in [("yaml_baseline", "configs/yolo11s_baseline.yaml"),
                          ("yaml_iama", "configs/yolo11s_iama.yaml")]:
        m = YOLO(str(PROJECT_ROOT / yaml_rel))
        dm = m.model
        total = sum(p.numel() for p in dm.parameters())
        trainable = sum(p.numel() for p in dm.parameters() if p.requires_grad)
        attn = attention_breakdown(dm)
        chans = probe_input_channels(m) if attn else {}
        for r in attn:
            r["actual_input_channels"] = chans.get(r["layer_index"])
        attn_total = sum(r["block_params"] for r in attn)
        m.fuse()
        fused_total = sum(p.numel() for p in m.model.parameters())
        out_rows.append({
            "model": tag, "total_params": total, "trainable_params": trainable,
            "fused_params": fused_total,
            "attention_params": attn_total,
            "attention_layers": str([r["layer_index"] for r in attn]),
            "eca_kernel_sizes": str([r["eca_kernel_size"] for r in attn]),
        })
        print(f"{tag}: total={total:,} trainable={trainable:,} fused={fused_total:,} attention={attn_total}")
        for r in attn:
            print("   ", r)

    for name, path in CHECKPOINTS.items():
        m = YOLO(str(path))
        dm = m.model.float()
        total = sum(p.numel() for p in dm.parameters())
        attn = attention_breakdown(dm)
        attn_total = sum(r["block_params"] for r in attn)
        # fused count (BN folded into conv) — this is what ultralytics prints in
        # "model summary (fused)" and what the old report quoted (9,415,122)
        m.fuse()
        fused_total = sum(p.numel() for p in m.model.parameters())
        out_rows.append({
            "model": f"ckpt_{name}", "total_params": total, "trainable_params": total,
            "fused_params": fused_total,
            "attention_params": attn_total,
            "attention_layers": str([r["layer_index"] for r in attn]),
            "eca_kernel_sizes": str([r["eca_kernel_size"] for r in attn]),
        })
        print(f"ckpt {name}: total={total:,} fused={fused_total:,} attention={attn_total}")
        for r in attn:
            print("   ", r)

    with open(AUDIT_DIR / "param_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved {AUDIT_DIR / 'param_counts.csv'}")


if __name__ == "__main__":
    sys.exit(main())

