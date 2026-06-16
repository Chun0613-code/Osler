"""Feature-aligned transfer from PhysioNet 2019 ICU JEPA into DKA JEPA.

The transfer is deliberately narrow.  PhysioNet 2019 has no DKA intervention
actions, so this script may initialize overlapping state-encoder features only.
It does not validate, promote, or replace the action-conditioned DKA checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dka_body import DKABody
from dka_world_model import (
    H_DIM,
    STATE_KEYS,
    WorldModel,
    randomized_dka,
    s2vec,
    save_checkpoint,
)
from physionet2019_pretrain import (
    DYNAMIC_KEYS as PHYSIONET_DYNAMIC_KEYS,
    STATIC_KEYS as PHYSIONET_STATIC_KEYS,
    PhysioNetICUJepa,
)


FEATURE_MAP = {
    "G": "Glucose",
    "pH": "pH",
    "HCO3": "HCO3",
    "Ke": "Potassium",
    "MAP": "MAP",
    "creatinine": "Creatinine",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_physionet_model(path: Path) -> tuple[PhysioNetICUJepa, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    dynamic_keys = payload.get("dynamic_keys", list(PHYSIONET_DYNAMIC_KEYS))
    static_keys = payload.get("static_keys", list(PHYSIONET_STATIC_KEYS))
    if list(dynamic_keys) != list(PHYSIONET_DYNAMIC_KEYS):
        raise ValueError("PhysioNet checkpoint dynamic_keys do not match this code")
    if list(static_keys) != list(PHYSIONET_STATIC_KEYS):
        raise ValueError("PhysioNet checkpoint static_keys do not match this code")
    model = PhysioNetICUJepa(len(PHYSIONET_DYNAMIC_KEYS), len(PHYSIONET_STATIC_KEYS))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def copy_if_compatible(source: torch.nn.Module, target: torch.nn.Module) -> list[str]:
    copied = []
    for name, source_parameter in source.state_dict().items():
        target_state = target.state_dict()
        if name in target_state and target_state[name].shape == source_parameter.shape:
            target_state[name].copy_(source_parameter)
            copied.append(name)
    return copied


def transfer_encoder(
    physionet: PhysioNetICUJepa,
    dka: WorldModel,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    mapped_features = {}
    unmapped_dka_features = []
    phys_index = {name: index for index, name in enumerate(PHYSIONET_DYNAMIC_KEYS)}
    dka_index = {name: index for index, name in enumerate(STATE_KEYS)}

    with torch.no_grad():
        # Hidden layers have the same MLP width.  Copy them first, then seed only
        # first-layer columns whose physiology has a direct name match.
        copied_encoder_layers = copy_if_compatible(physionet.E[2:], dka.E[2:])
        dka.E[0].bias.copy_(physionet.E[0].bias)
        for dka_key in STATE_KEYS:
            source_key = FEATURE_MAP.get(dka_key)
            if source_key is None:
                unmapped_dka_features.append(dka_key)
                continue
            dka_col = dka_index[dka_key]
            phys_col = phys_index[source_key]
            dka.E[0].weight[:, dka_col].copy_(physionet.E[0].weight[:, phys_col])
            mapped_features[dka_key] = source_key
        dka.Ebar.load_state_dict(dka.E.state_dict())

    return {
        "mapped_features": mapped_features,
        "unmapped_dka_features": unmapped_dka_features,
        "copied_encoder_layers": copied_encoder_layers,
        "copied_first_layer_bias": True,
        "copied_target_encoder_from_online_encoder": True,
    }


@torch.no_grad()
def latent_audit(model: WorldModel, samples: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    body = DKABody(rng=rng)
    states = np.stack([s2vec(randomized_dka(body, rng)) for _ in range(samples)])
    histories = np.zeros((samples, H_DIM), dtype=np.float32)
    latent = model.encode_state(
        torch.as_tensor(states, dtype=torch.float32),
        torch.as_tensor(histories, dtype=torch.float32),
    ).numpy()
    std = latent.std(axis=0)
    covariance = np.cov(latent, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    effective_rank = float(np.exp(entropy))
    return {
        "samples": samples,
        "mean_latent_std": round(float(std.mean()), 6),
        "min_latent_std": round(float(std.min()), 6),
        "effective_rank": round(effective_rank, 3),
        "active_dimensions_std_gt_0_01": int((std > 0.01).sum()),
        "collapsed": bool(std.mean() < 0.05 or effective_rank < 3.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physionet-checkpoint", default="physionet2019_icu_jepa.pt")
    parser.add_argument("--output", default="dka_physionet_encoder_init.pt")
    parser.add_argument("--report", default="dka_physionet_encoder_init_report.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--audit-samples", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    physionet_path = Path(args.physionet_checkpoint)
    output_path = Path(args.output)
    report_path = Path(args.report)
    physionet, payload = load_physionet_model(physionet_path)
    torch.manual_seed(args.seed)
    dka = WorldModel()
    transfer = transfer_encoder(physionet, dka, args.seed)
    audit = latent_audit(dka, args.audit_samples, args.seed)
    report = {
        "experiment": "physionet2019_feature_aligned_dka_encoder_init",
        "source_checkpoint": {
            "path": str(physionet_path),
            "sha256": sha256(physionet_path),
            "metadata": payload.get("metadata", {}),
        },
        "output_checkpoint": str(output_path),
        "seed": args.seed,
        "transfer": transfer,
        "latent_audit_on_randomized_dka_states": audit,
        "promotion_boundary": {
            "may_start_controlled_dka_training_experiment": True,
            "may_replace_dka_symbolic_jepa_v5": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "reason": (
                "PhysioNet 2019 has no explicit DKA treatment action channels; "
                "this checkpoint only seeds overlapping state-encoder weights."
            ),
        },
    }
    save_checkpoint(dka, output_path, metadata=report)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "report": str(report_path),
        "mapped_features": transfer["mapped_features"],
        "collapsed": audit["collapsed"],
    }, indent=2))


if __name__ == "__main__":
    main()
