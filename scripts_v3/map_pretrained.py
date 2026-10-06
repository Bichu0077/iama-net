"""Phase 2.2: build IAMA-architecture init from COCO-pretrained yolo11s.pt
using the aligned semantic layer mapping, with identity attention init.

Baseline layer index -> IAMA layer index (attention inserted at 5, 8, 13):
  0-4   -> 0-4      (stem + P2 + F3 block)
  5-6   -> 6-7      (P4 downsample + C3k2)
  7-9   -> 9-11     (P5 downsample + C3k2 + SPPF)
  10    -> 12       (C2PSA)
  11-23 -> 14-26    (head, shifted by 3)

Attention blocks get zero weights and +4.0 bias (sigmoid(4)=0.982 per gate,
cascaded scale 0.964) so the network starts as a near-identity extension of
the pretrained model. The Detect head class branch cannot transfer from COCO
(nc=80 vs nc=6); those tensors keep their yaml init. Every transferred /
skipped tensor is logged to the manifest JSON.

Output: runs_v3/init/iama_coco_mapped_init.pt  (+ manifest json)
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from iama_env import register_iama_modules, PROJECT_ROOT

register_iama_modules()

from ultralytics import YOLO  # noqa: E402

LAYER_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    5: 6, 6: 7, 7: 9, 8: 10, 9: 11, 10: 12,
    11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 19, 17: 20,
    18: 21, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26,
}
ATTN_LAYERS = [5, 8, 13]
BIAS_INIT = 4.0


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build(source_weights: str, target_yaml: str, output_pt: str,
          explicit_identity_init: bool = True):
    """Map COCO/baseline weights into target_yaml architecture.

    explicit_identity_init=True (ECA_SpatialAttn targets): zero gate weights
    and fill biases with +4.0 after mapping. For SE/CBAM/CA targets the
    constructors already apply the same identity init, so pass False and the
    constructor state is preserved and logged.
    """
    target = YOLO(target_yaml)
    src = torch.load(source_weights, map_location="cpu", weights_only=False)
    src_state = src["model"].float().state_dict() if hasattr(src.get("model"), "state_dict") else src
    tgt_state = target.model.state_dict()

    transferred, skipped_shape, skipped_nomatch = [], [], []
    for k, v in src_state.items():
        parts = k.split(".")
        new_key = None
        if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
            base = int(parts[1])
            if base in LAYER_MAP:
                new_key = f"model.{LAYER_MAP[base]}." + ".".join(parts[2:])
        if new_key is None:
            new_key = k
        if new_key in tgt_state:
            if tgt_state[new_key].shape == v.shape:
                tgt_state[new_key] = v
                transferred.append(new_key)
            else:
                skipped_shape.append([new_key, list(v.shape), list(tgt_state[new_key].shape)])
        else:
            skipped_nomatch.append(new_key)

    attn_init = []
    if explicit_identity_init:
        for li in ATTN_LAYERS:
            for pat, val in ((f"model.{li}.eca.conv.weight", 0.0),
                             (f"model.{li}.eca.conv.bias", BIAS_INIT),
                             (f"model.{li}.spatial_conv.weight", 0.0),
                             (f"model.{li}.spatial_conv.bias", BIAS_INIT)):
                if pat in tgt_state:
                    if val == 0.0:
                        tgt_state[pat].zero_()
                    else:
                        tgt_state[pat].fill_(val)
                    attn_init.append([pat, val])
                else:
                    raise KeyError(f"attention param missing in target model: {pat}")
    else:
        # log constructor identity-init state of every attention-layer tensor
        for li in ATTN_LAYERS:
            for k, v in tgt_state.items():
                if k.startswith(f"model.{li}."):
                    attn_init.append([k, [round(float(x), 4) for x in v.flatten()[:2].tolist()],
                                      "w_absmax=" + str(round(float(v.abs().max()), 6)) if v.numel() else ""])

    target.model.load_state_dict(tgt_state)
    Path(output_pt).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model": target.model, "train_args": {}, "date": datetime.now().isoformat(),
            "version": "8.3.40-mapped-init"}
    torch.save(ckpt, output_pt)

    manifest = {
        "script": "scripts_v3/map_pretrained.py",
        "source_weights": source_weights,
        "target_yaml": target_yaml,
        "output": output_pt,
        "git_commit": git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "layer_map": LAYER_MAP,
        "n_src_tensors": len(src_state),
        "n_target_tensors": len(tgt_state),
        "n_transferred_tensors": len(transferred),
        "transferred_tensors": transferred,
        "skipped_shape_mismatch": skipped_shape,
        "skipped_no_key_match": skipped_nomatch,
        "attention_identity_init": attn_init,
    }
    mpath = Path(output_pt).with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"transferred {len(transferred)}/{len(src_state)} source tensors; "
          f"shape-skipped {len(skipped_shape)}; no-key {len(skipped_nomatch)}")
    print(f"manifest -> {mpath}")
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(PROJECT_ROOT / "yolo11s.pt"))
    ap.add_argument("--yaml", default=str(PROJECT_ROOT / "configs" / "yolo11s_iama.yaml"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "runs_v3" / "init" / "iama_coco_mapped_init.pt"))
    ap.add_argument("--constructor-init", action="store_true",
                    help="attention blocks apply identity init in their constructor "
                         "(SE/CBAM/CA); skip the explicit ECA fill")
    a = ap.parse_args()
    build(a.source, a.yaml, a.out, explicit_identity_init=not a.constructor_init)
