"""
dka_fidelity_replay.py
======================
The exam. Take a real DKA patient's route-aware insulin, fluid, electrolyte, and
buffer actions over time
and initial state, replay them through DKABody, and compare the emergent
glucose/pH/HCO3/K+ curves to the patient's actually-observed labs.

  - small error  -> the hand-written primitives captured the real mechanism;
                    the agent's learned policy is trustworthy.
  - large error  -> a pool is wrong/missing, or constants need calibration.
                    The PATTERN of error tells you which.

Trajectory schema (one dict per stay; produced by dka_fidelity_extract.py):
  {
    "stay_id": int,
    "init":    {"glucose":.., "HCO3":.., "K":.., "anion_gap":..},   # observed at onset
    "actions": [{"t":hours, "insulin_iv":U/hr, "insulin_rapid_sc":U/hr,
                 "fluids":mL/hr, "kcl":mEq/hr, ...}, ...],
    "labs":    [{"t":hours, "var":"glucose"|"pH"|"HCO3"|"K", "value":..}, ...]
  }

This file also self-tests on a synthetic patient (round-trip) so a non-zero error on
real data means sim-vs-reality gap, not a replay bug.

Run:  python dka_fidelity_replay.py trajectories.jsonl
      python dka_fidelity_replay.py            # runs the synthetic self-test
"""

import sys
import json
import numpy as np
from dka_body import (
    DKABody, K_INS_ABS, K_INS_CLEAR, K_SC_BASAL_ABS,
    K_SC_INTERMEDIATE_ABS, K_SC_RAPID_ABS, MAP_MIN, MAP_NORM,
    estimate_potassium_store,
)
from dka_action_contract import ACTION_INDEX, ACTION_KEYS, expand_action

DT = 0.5
VAR2ATTR = {
    "glucose": "G", "pH": "pH", "HCO3": "HCO3", "K": "Ke",
    "Na": "Na", "osmolality": "osmolality", "creatinine": "creatinine",
    "urine_output": "urine_output", "BHB": "BHB", "anion_gap": "anion_gap",
    "K_store": "K_store", "osmotic_injury": "osmotic_injury",
}


def init_body(init, history_actions=None):
    b = DKABody()
    b.reset()
    if "glucose" in init and init["glucose"] is not None:
        b.G = float(init["glucose"])
    if "HCO3" in init and init["HCO3"] is not None:
        b.HCO3 = float(init["HCO3"])
    if "K" in init and init["K"] is not None:
        b.Ke = float(init["K"])
    if init.get("Na") is not None:
        b.Na = float(init["Na"])
    if init.get("creatinine") is not None:
        b.Cr = float(init["creatinine"])
    if init.get("urine_output") is not None:
        b.urine_output_ml_hr = float(init["urine_output"])
    if init.get("MAP") is not None:
        filling = np.clip(
            (float(init["MAP"]) - MAP_MIN) / (MAP_NORM - MAP_MIN), 0.1, 1.2
        )
        b.V = filling * b.volume_setpoint / b.profile.vascular_tone
    prior_kcl = 0.0
    if history_actions:
        prior_kcl = sum(
            expand_action(action)[ACTION_INDEX["kcl"]] * DT
            for action in history_actions
        )
    b.Ki = estimate_potassium_store(
        b.Ke, b.pH, b.Cr, b.urine_output_ml_hr, prior_kcl
    )
    current_osm = 2.0 * b.Na + b.G / 18.0
    b.osmotic_injury = max(0.0, (current_osm - 320.0) / 20.0) ** 2
    # Ket and V are not directly observed: estimate Ket from anion gap, V default
    if init.get("anion_gap") is not None:
        b.Ket = max(0.0, float(init["anion_gap"]) - 12.0)
    elif init.get("BHB") is not None:
        b.Ket = max(0.0, float(init["BHB"]) / 0.75)
    # The concentration states above are observed at the anchor and must not be
    # replayed from an unknown pre-anchor state. Prior insulin is different: it
    # determines the hidden insulin signal still active at the anchor.
    if history_actions:
        insulin_signal = b.profile.endogenous_insulin
        rapid_depot = intermediate_depot = basal_depot = 0.0
        for action in sorted(history_actions, key=lambda item: item.get("t", 0.0)):
            values = expand_action(action)
            h = DT / 30.0
            for _ in range(30):
                rapid_absorbed = K_SC_RAPID_ABS * rapid_depot
                intermediate_absorbed = K_SC_INTERMEDIATE_ABS * intermediate_depot
                basal_absorbed = K_SC_BASAL_ABS * basal_depot
                rapid_depot += (
                    values[ACTION_INDEX["insulin_rapid_sc"]] - rapid_absorbed
                ) * h
                intermediate_depot += (
                    values[ACTION_INDEX["insulin_intermediate_sc"]]
                    - intermediate_absorbed
                ) * h
                basal_depot += (
                    values[ACTION_INDEX["insulin_basal_sc"]] - basal_absorbed
                ) * h
                insulin_signal += (
                    (
                        values[ACTION_INDEX["insulin_iv"]]
                        + rapid_absorbed + intermediate_absorbed + basal_absorbed
                    ) * K_INS_ABS
                    - K_INS_CLEAR * (insulin_signal - b.profile.endogenous_insulin)
                ) * h
                insulin_signal = max(0.0, insulin_signal)
        b.I = insulin_signal
        b.insulin_rapid_depot = max(0.0, rapid_depot)
        b.insulin_intermediate_depot = max(0.0, intermediate_depot)
        b.insulin_basal_depot = max(0.0, basal_depot)
    return b


def replay(traj):
    """Return list of (var, t, real, sim) at each lab time, plus the sim curve."""
    b = init_body(traj["init"], traj.get("history_actions"))
    # action lookup on the DT grid
    act_by_t = {round(a["t"] / DT): a for a in traj["actions"]}
    last_t = max([a["t"] for a in traj["actions"]] + [l["t"] for l in traj["labs"]] + [0])
    n = int(np.ceil(last_t / DT)) + 1

    sim_curve = {}          # grid_index -> observation dict
    sim_curve[0] = b.observe()
    for k in range(n):
        a = act_by_t.get(k, {name: 0.0 for name in ACTION_KEYS})
        obs, r, dead, info = b.step(a, dt=DT)
        sim_curve[k + 1] = obs
        if dead:
            break

    paired = []
    max_k = max(sim_curve.keys())
    for lab in traj["labs"]:
        k = min(round(lab["t"] / DT), max_k)
        sim_val = sim_curve[k][VAR2ATTR[lab["var"]]]
        paired.append((lab["var"], lab["t"], lab["value"], sim_val))
    return paired, sim_curve, b.alive, b.death_cause


def score(all_paired):
    by_var = {}
    for var, t, real, sim in all_paired:
        by_var.setdefault(var, []).append(abs(real - sim))
    return {v: (len(e), float(np.mean(e))) for v, e in by_var.items()}


def run_file(path):
    trajs = [json.loads(line) for line in open(path) if line.strip()]
    print(f"Loaded {len(trajs)} DKA trajectories\n")
    all_paired, causes = [], {}
    glu_cov, glu_nocov = [], []   # glucose |err| split by insulin coverage
    for tr in trajs:
        paired, curve, alive, cause = replay(tr)
        all_paired += paired
        if not alive:
            causes[cause] = causes.get(cause, 0) + 1
        ins_cov = tr.get("_cover", {}).get("insulin", 1) > 0
        for var, t, real, sim in paired:
            if var == "glucose":
                (glu_cov if ins_cov else glu_nocov).append(abs(real - sim))

    print("per-variable MAE (real vs simulated), across all lab timepoints:")
    print(f"{'var':<10}{'n':>6}{'MAE':>10}")
    for v, (n, mae) in score(all_paired).items():
        print(f"{v:<10}{n:>6}{mae:>10.2f}")

    print("\nglucose MAE split by whether insulin was captured for that stay:")
    if glu_cov:
        print(f"  insulin-captured stays  n={len(glu_cov):<3} MAE={np.mean(glu_cov):7.1f}")
    if glu_nocov:
        print(f"  insulin-MISSING stays   n={len(glu_nocov):<3} MAE={np.mean(glu_nocov):7.1f}")
    print("  (if MISSING >> captured, the glucose error is a coverage artifact, not sim)")

    print(f"\nsim died on {sum(causes.values())}/{len(trajs)} stays. causes:")
    for c, n in sorted(causes.items(), key=lambda x: -x[1]):
        tag = "  <- likely KCl-coverage artifact" if "kalemia" in c else \
              "  <- real sim/glucose issue" if ("osmolar" in c or "acidosis" in c) else ""
        print(f"  {c:<28} {n}{tag}")
    print("\nRead the error PATTERN: which variable, grow-over-time (rate wrong) vs "
          "start-offset (init/unobserved wrong).")


# ---------------------------------------------------------------------------
def self_test():
    """Round-trip: generate a synthetic patient from the sim, sparsely sample
    'labs', then replay and confirm near-zero error (machinery is correct)."""
    print("SELF-TEST (synthetic round-trip)\n")
    truth = DKABody(); truth.reset()
    truth.G, truth.HCO3, truth.Ke, truth.Ket = 520.0, 9.0, 5.4, 13.0
    init = {"glucose": 520.0, "HCO3": 9.0, "K": 5.4, "anion_gap": 12.0 + 13.0}
    actions, labs = [], []
    for k in range(48):
        t = k * DT
        a = {
            "t": t, "insulin": 6, "fluids": 500, "kcl": 20,
            "bicarbonate": 0, "dextrose": 0,
        }
        actions.append(a)
        obs, *_ = truth.step([
            a["insulin"], a["fluids"], a["kcl"],
            a["bicarbonate"], a["dextrose"],
        ], dt=DT)
        if k % 4 == 0:  # sample labs every 2h
            for var in ("glucose", "pH", "HCO3", "K"):
                labs.append({"t": t + DT, "var": var,
                             "value": obs[VAR2ATTR[var]]})
        if not truth.alive:
            break
    traj = {"stay_id": -1, "init": init, "actions": actions, "labs": labs}
    paired, curve, alive, cause = replay(traj)
    sc = score(paired)
    print("round-trip MAE (should be ~0):")
    for v, (n, mae) in sc.items():
        print(f"  {v:<8} n={n:<3} MAE={mae:.4f}")
    ok = all(mae < 1e-6 for _, (_, mae) in sc.items())
    print("\nMACHINERY OK" if ok else "\nWARNING: round-trip not exact -- replay bug")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        self_test()
