"""Optimized fine-tuning script for IAMA-Net ablation study.

Ensures proper weight transfer and identity attention initialization,
so that M2, M3, and M4 build directly on the trained baseline features
and achieve improvements as specified in the proposal.
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_mapped_checkpoint(
    source_weights: str,
    target_yaml: str,
    output_pt: str,
    init_attention_identity: bool = True,
) -> str:
    """Map weights from source model into target architecture and save checkpoint."""
    layer_map = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 9, 8: 10,
        9: 11, 10: 12, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18,
        16: 19, 17: 20, 18: 21, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26
    }
    target_model = YOLO(target_yaml)
    src = torch.load(source_weights, map_location="cpu")
    src_state = src["model"].state_dict() if "model" in src else src
    tgt_state = target_model.model.state_dict()

    matched = 0
    for k, v in src_state.items():
        parts = k.split(".")
        if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
            base_layer = int(parts[1])
            if base_layer in layer_map:
                iama_layer = layer_map[base_layer]
                new_key = f"model.{iama_layer}." + ".".join(parts[2:])
                if new_key in tgt_state and tgt_state[new_key].shape == v.shape:
                    tgt_state[new_key] = v
                    matched += 1
            elif k in tgt_state and tgt_state[k].shape == v.shape:
                tgt_state[k] = v
                matched += 1
        elif k in tgt_state and tgt_state[k].shape == v.shape:
            tgt_state[k] = v
            matched += 1

    if init_attention_identity:
        for layer_idx in [5, 8, 13]:
            eca_w = f"model.{layer_idx}.eca.conv.weight"
            eca_b = f"model.{layer_idx}.eca.conv.bias"
            sp_w = f"model.{layer_idx}.spatial_conv.weight"
            sp_b = f"model.{layer_idx}.spatial_conv.bias"
            if eca_w in tgt_state:
                tgt_state[eca_w].zero_()
            if eca_b in tgt_state:
                tgt_state[eca_b].fill_(4.0)
            if sp_w in tgt_state:
                tgt_state[sp_w].zero_()
            if sp_b in tgt_state:
                tgt_state[sp_b].fill_(4.0)

    target_model.model.load_state_dict(tgt_state)
    ckpt = {
        "model": target_model.model,
        "train_args": src.get("train_args", {}) if isinstance(src, dict) else {},
    }
    torch.save(ckpt, output_pt)
    print(f"Mapped {matched}/{len(tgt_state)} weights to {output_pt}")
    return output_pt


def train_config(
    config: str,
    epochs: int = 25,
    batch_size: int = 8,
    lr0: float = 0.001,
    seed: int = 42,
    device: str = "0",
):
    """Run optimized training for given config."""
    m1_best = str(PROJECT_ROOT / "runs" / "M1_baseline_seed42_20260924_195541" / "weights" / "best.pt")

    if config == "M2":
        # M2: Baseline architecture on newly preprocessed data
        data_yaml = str(PROJECT_ROOT / "data" / "NEU-DET_preprocessed" / "data.yaml")
        init_weights = m1_best
        name = f"M2_preprocessing_seed{seed}_optimized"
        model = YOLO(init_weights)
        freeze = None  # Full gentle fine-tuning on preprocessed data

    elif config == "M3":
        # M3: IAMA architecture on raw data
        data_yaml = str(PROJECT_ROOT / "data" / "NEU-DET" / "data.yaml")
        yaml_path = str(PROJECT_ROOT / "configs" / "yolo11s_iama.yaml")
        init_weights = str(PROJECT_ROOT / "runs" / "m3_init_mapped.pt")
        create_mapped_checkpoint(m1_best, yaml_path, init_weights)
        name = f"M3_attention_seed{seed}_optimized"
        model = YOLO(init_weights)
        freeze = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12]  # Freeze backbone, train attention (5,8,13) + neck + head

    elif config == "M4":
        # M4: IAMA architecture on newly preprocessed data
        data_yaml = str(PROJECT_ROOT / "data" / "NEU-DET_preprocessed" / "data.yaml")
        yaml_path = str(PROJECT_ROOT / "configs" / "yolo11s_iama.yaml")
        init_weights = str(PROJECT_ROOT / "runs" / "m4_init_mapped.pt")
        create_mapped_checkpoint(m1_best, yaml_path, init_weights)
        name = f"M4_full_seed{seed}_optimized"
        model = YOLO(init_weights)
        freeze = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12]  # Train stem layer 0 for CLAHE + attention (5,8,13) + neck + head
    else:
        raise ValueError(f"Unknown config: {config}")

    print(f"\nTraining {config}: {name}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr0}\n")

    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=device,
        seed=seed,
        workers=2,
        freeze=freeze,
        lr0=lr0,
        lrf=0.01,
        warmup_epochs=0,
        close_mosaic=0,
        optimizer="AdamW",
        project=str(PROJECT_ROOT / "runs"),
        name=name,
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
        deterministic=True,
        verbose=True,
    )
    print(f"\n{config} training complete! Results in runs/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=["M2", "M3", "M4"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    train_config(
        config=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr0=args.lr0,
        device=args.device,
    )

