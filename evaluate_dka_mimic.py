"""Evaluate a trained intervention JEPA on a rebuilt MIMIC DKA cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dka_world_model import load_checkpoint
from train_intervention_jepa import choose_device, evaluate_mimic_proxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="dka_intervention_jepa.pt")
    parser.add_argument("--mimic", default="dka_transitions_6h.parquet")
    parser.add_argument("--output", default="dka_mimic_evaluation.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    report = evaluate_mimic_proxy(model, args.mimic, torch.device(device))
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["checkpoint_model"] = payload.get("metadata", {}).get("model")
    report["cohort"] = str(Path(args.mimic).resolve())
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()
