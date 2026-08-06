"""Graph-propagated whole-body coupling features.

The explicit latent layer builds interpretable organ-system axes.  This module
adds one bounded step of message passing across those axes:

    hemodynamic <-> respiratory <-> acid-base/electrolyte <-> renal
                       |             |
                        metabolic/endocrine

The graph is fixed and physiology-inspired.  It is not learned from the audit
set, does not encode treatment effects, and does not grant causal authority.
The only question the audit may answer is whether these graph-propagated
features add held-out factual signal beyond the already strong baseline:
ordinary features + all belief families + explicit whole-body latent axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from whole_body_coupling_latent import whole_body_coupling_latent_features


AXES: tuple[str, ...] = (
    "hemodynamic",
    "respiratory",
    "metabolic",
    "electrolyte",
    "renal",
)

LATENT_COLUMNS: dict[str, str] = {
    "hemodynamic": "wbc_latent_hemodynamic_axis",
    "respiratory": "wbc_latent_respiratory_axis",
    "metabolic": "wbc_latent_metabolic_axis",
    "electrolyte": "wbc_latent_electrolyte_axis",
    "renal": "wbc_latent_renal_axis",
}

INBOUND_GRAPH: dict[str, tuple[tuple[str, float], ...]] = {
    "hemodynamic": (
        ("respiratory", 0.30),
        ("metabolic", 0.25),
        ("renal", 0.20),
    ),
    "respiratory": (
        ("hemodynamic", 0.30),
        ("metabolic", 0.20),
        ("electrolyte", 0.20),
    ),
    "metabolic": (
        ("hemodynamic", 0.25),
        ("respiratory", 0.15),
        ("electrolyte", 0.25),
        ("renal", 0.20),
    ),
    "electrolyte": (
        ("renal", 0.35),
        ("metabolic", 0.25),
        ("respiratory", 0.20),
    ),
    "renal": (
        ("hemodynamic", 0.35),
        ("metabolic", 0.20),
        ("electrolyte", 0.20),
    ),
}


def _finite(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    output[~np.isfinite(output)] = 0.0
    return output


def _axis_values(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = [column for column in LATENT_COLUMNS.values() if column not in frame]
    if missing:
        latent = whole_body_coupling_latent_features(frame)
        frame = pd.concat([frame.reset_index(drop=True), latent.reset_index(drop=True)], axis=1)
    return {
        axis: _finite(pd.to_numeric(frame[LATENT_COLUMNS[axis]], errors="coerce").to_numpy(dtype=np.float64))
        for axis in AXES
    }


def whole_body_graph_coupling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one-hop and two-hop whole-body graph coupling features."""

    axes = _axis_values(frame)
    stacked = np.vstack([axes[axis] for axis in AXES])
    global_axis = np.nanmean(stacked, axis=0)
    spread = np.nanstd(stacked, axis=0)

    messages: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    tensions: dict[str, np.ndarray] = {}
    outputs: dict[str, np.ndarray] = {}

    for target, inbound in INBOUND_GRAPH.items():
        total_weight = sum(weight for _, weight in inbound)
        message = sum(axes[source] * weight for source, weight in inbound) / max(total_weight, 1e-6)
        residual = axes[target] - message
        tension = np.abs(residual)
        messages[target] = message
        residuals[target] = residual
        tensions[target] = tension
        outputs[f"wbg_msg_{target}"] = message
        outputs[f"wbg_residual_{target}"] = residual
        outputs[f"wbg_tension_{target}"] = tension
        outputs[f"wbg_coupled_{target}"] = axes[target] * message

    total_tension = np.nanmean(np.vstack([tensions[axis] for axis in AXES]), axis=0)
    message_load = np.nanmean(np.vstack([messages[axis] for axis in AXES]), axis=0)
    outputs.update({
        "wbg_graph_message_load": message_load,
        "wbg_graph_total_tension": total_tension,
        "wbg_graph_axis_spread": spread,
        "wbg_global_x_message_load": global_axis * message_load,
        "wbg_global_x_total_tension": global_axis * total_tension,
        "wbg_spread_x_tension": spread * total_tension,
        "wbg_hemo_resp_metabolic_triad": axes["hemodynamic"] * axes["respiratory"] * axes["metabolic"],
        "wbg_hemo_metabolic_renal_triad": axes["hemodynamic"] * axes["metabolic"] * axes["renal"],
        "wbg_resp_electrolyte_renal_triad": axes["respiratory"] * axes["electrolyte"] * axes["renal"],
        "wbg_metabolic_electrolyte_renal_triad": axes["metabolic"] * axes["electrolyte"] * axes["renal"],
        "wbg_hemo_to_renal_message_chain": messages["renal"] * axes["hemodynamic"],
        "wbg_resp_to_electrolyte_message_chain": messages["electrolyte"] * axes["respiratory"],
        "wbg_metabolic_to_electrolyte_message_chain": messages["electrolyte"] * axes["metabolic"],
        "wbg_renal_to_metabolic_message_chain": messages["metabolic"] * axes["renal"],
        "wbg_hemo_renal_tension_product": tensions["hemodynamic"] * tensions["renal"],
        "wbg_resp_electrolyte_tension_product": tensions["respiratory"] * tensions["electrolyte"],
        "wbg_metabolic_renal_tension_product": tensions["metabolic"] * tensions["renal"],
    })

    return pd.DataFrame(outputs, index=frame.index, dtype=np.float64).replace([np.inf, -np.inf], np.nan)
