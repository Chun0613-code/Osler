"""
dka_osler.py  (L3 symbolic engine)
==================================
Three jobs:
  (1) SAFETY SHIELD: hard clinical constraints applied between the controller and the
      body, acting on the (believed) state. e.g. low K+ + insulin -> force KCl / hold
      insulin. In deployment it reads the L1 belief-filter's inferred K+ (validated at
      MAE 0.18 even when K+ is unmeasured); here we demo the rule logic on true state.
  (2) MECHANISM VERIFICATION: the sim's known couplings are ground-truth causal edges.
      We check that the LEARNED L1 world model respects them (perturb an action in the
      latent model, decode, check the effect SIGN matches the known mechanism).
  (3) REASONING TRACE: every shield intervention emits a cause->effect explanation.

Run:  python dka_osler.py
"""

import numpy as np
import torch
from dka_body import DKABody
from dka_action_contract import ACTION_INDEX, ACTION_KEYS, expand_action, legacy_action
from dka_world_model import (WorldModel, collect, train, S_MEAN, S_STD, STATE_KEYS,
                             A_DIM, A_SCALE, s2vec, DT, HORIZON, randomized_dka)

np.random.seed(0); torch.manual_seed(0)


# ======================= (1) SAFETY SHIELD ==================================
def shield(s, action):
    """s: state dict (believed). action: [insulin, fluids, kcl]. -> (safe, trace)."""
    safe, trace = shield_full(s, action)
    return safe[:3], trace


def shield_full(s, action):
    """Apply DKA safety rules to all five intervention channels."""
    values = list(action) + [0.0] * (5 - len(action))
    ins, fl, kcl, bicarb, dextrose = values[:5]
    Ke, G, pH, MAP = s["Ke"], s["G"], s["pH"], s["MAP"]
    hco3 = s.get("HCO3", 24.0)
    anion_gap = s.get("anion_gap", 12.0)
    bhb = s.get("BHB", max(0.0, anion_gap - 12.0) * 0.75)
    creatinine = s.get("creatinine", 1.0)
    urine_output = s.get("urine_output", 100.0)
    potassium_store = s.get("K_store")
    osmotic_injury = s.get("osmotic_injury", 0.0)
    trace = []

    if Ke > 5.5 and kcl > 0:                       # hyperkalemia guard
        trace.append(f"K+={Ke:.1f} high → withhold KCl"); kcl = 0.0
    if (creatinine > 2.5 or urine_output < 30) and Ke >= 4.5 and kcl > 0:
        trace.append("renal dysfunction/oliguria + non-low K+ → withhold KCl")
        kcl = 0.0
    if Ke < 3.0 and ins > 0:                        # critical hypokalemia: hold insulin
        trace.append(f"K+={Ke:.1f} critically low → hold insulin until K+ recovers")
        ins = 0.0
        if kcl < 20:
            trace.append("→ give KCl 20 mEq/hr"); kcl = 20.0
    elif Ke < 3.3 and ins > 0 and kcl == 0:         # low K+ + insulin: mandatory KCl
        trace.append(f"K+={Ke:.1f} low + insulin running → mandatory KCl 10 mEq/hr")
        kcl = 10.0
    elif (
        potassium_store is not None and potassium_store < 80
        and ins > 0 and Ke < 5.0 and kcl == 0
        and creatinine <= 2.5 and urine_output >= 30
    ):
        replacement = 20.0 if potassium_store < 60 else 10.0
        trace.append(
            f"estimated total-body K+ store={potassium_store:.0f} depleted "
            f"despite serum K+={Ke:.1f} → add KCl {replacement:.0f} mEq/hr"
        )
        kcl = replacement
    if G < 80 and ins > 0:                           # hypoglycemia guard
        trace.append(f"glucose={G:.0f} low → hold insulin"); ins = 0.0
        dextrose = max(dextrose, 10.0)
    elif G < 250 and ins > 0 and (hco3 < 18 or anion_gap > 12 or bhb >= 1.0):
        if dextrose < 5.0:
            trace.append(
                f"glucose={G:.0f} but ketoacidosis unresolved → add dextrose and continue insulin"
            )
            dextrose = 5.0
    if MAP < 55 and fl == 0:                         # hypotension guard
        trace.append(f"MAP={MAP:.0f} low → give fluids"); fl = 500.0
    elif osmotic_injury >= 8.0 and fl == 0:
        trace.append(
            f"cumulative hyperosmolar injury={osmotic_injury:.1f} high → give isotonic fluids"
        )
        fl = 250.0
    if pH < 6.95 and ins == 0:                       # untreated severe acidosis
        trace.append(f"pH={pH:.2f} severe acidosis untreated → start insulin"); ins = 4.0

    return [ins, fl, kcl, bicarb, dextrose], trace


def shield_route_aware(s, action):
    """Apply legacy safety rules while preserving the proposed insulin route."""
    proposed = expand_action(action)
    legacy = legacy_action(proposed)
    safe_legacy, trace = shield_full(s, legacy)
    safe = proposed.copy()
    proposed_insulin = float(proposed[:4].sum())
    safe_insulin = float(safe_legacy[0])
    if proposed_insulin > 0:
        safe[:4] *= safe_insulin / proposed_insulin
    elif safe_insulin > 0:
        safe[ACTION_INDEX["insulin_iv"]] = safe_insulin
    safe[ACTION_INDEX["fluids"]] = safe_legacy[1]
    safe[ACTION_INDEX["kcl"]] = safe_legacy[2]
    safe[ACTION_INDEX["bicarbonate"]] = safe_legacy[3]
    safe[ACTION_INDEX["dextrose"]] = safe_legacy[4]
    return safe.tolist(), trace


def run_policy(policy, shield_on, n=300):
    b = DKABody(); surv = 0
    for _ in range(n):
        o = randomized_dka(b)
        for _ in range(HORIZON):
            a = policy(o)
            if shield_on:
                a, _ = shield(o, a)
            o, r, dead, _ = b.step(a + [0.0], dt=DT)
            if dead:
                break
        surv += b.alive
    return surv / n


# ======================= (2) MECHANISM VERIFICATION =========================
# ground-truth causal edges in the sim: (action component, state var, expected sign)
EDGES = [
    ("insulin_iv", "G", -1), ("insulin_iv", "Ke", -1),
    ("insulin_rapid_sc", "G", -1),
    ("fluids", "MAP", +1), ("fluids", "V", +1),
    ("kcl", "Ke", +1), ("kcl", "K_store", +1),
    ("bicarbonate", "HCO3", +1),
    ("dextrose", "G", +1),
]
ACT_COMP = ACTION_INDEX
PERT = {
    "insulin_iv": 6.0, "insulin_rapid_sc": 16.0,
    "fluids": 500.0, "kcl": 20.0,
    "bicarbonate": 50.0, "dextrose": 10.0,
}


def verify_mechanisms(wm, n_states=512):
    device = next(wm.parameters()).device
    b = DKABody(); states = []
    for _ in range(n_states):
        o = randomized_dka(b)
        for _ in range(np.random.randint(0, 10)):   # spread across the trajectory
            o, *_ = b.step([np.random.uniform(0, 6), np.random.uniform(0, 500),
                            np.random.uniform(0, 20), 0.0], dt=DT)
            if not b.alive:
                break
        states.append(s2vec(o))
    Z = wm.E(torch.tensor(np.array(states, dtype=np.float32), device=device)).detach()

    print("\nmechanism verification (does L1 respect the sim's causal directions?):")
    print(f"  {'edge':<22}{'expect':>7}{'mean Δ':>10}  match")
    for comp, var, sign in EDGES:
        base = np.zeros((1, A_DIM), np.float32)
        pert = np.zeros((1, A_DIM), np.float32)
        pert[0, ACT_COMP[comp]] = PERT[comp] / A_SCALE[ACT_COMP[comp]]
        ab = torch.tensor(np.repeat(base, len(Z), 0), device=device)
        ap = torch.tensor(np.repeat(pert, len(Z), 0), device=device)
        zb = wm.predict_latent(Z, ab)
        zp = wm.predict_latent(Z, ap)
        vi = STATE_KEYS.index(var)
        db = (wm.D(zb).detach().cpu().numpy()[:, vi]) * S_STD[vi] + S_MEAN[vi]
        dp = (wm.D(zp).detach().cpu().numpy()[:, vi]) * S_STD[vi] + S_MEAN[vi]
        delta = float(np.mean(dp - db))
        ok = "Y" if np.sign(delta) == sign else "**NO**"
        print(f"  {comp+'→'+var:<22}{'+' if sign>0 else '−':>7}{delta:>10.2f}    {ok}")


# ======================= (3) REASONING TRACE ================================
def show_trace():
    b = DKABody(); o = randomized_dka(b)
    naive = lambda s: [6.0, 500.0, 0.0]   # treats glucose, forgets K+ (unsafe)
    print("\nreasoning trace (naive policy + Osler shield), sampled steps:")
    for t in range(HORIZON):
        a0 = naive(o)
        a, tr = shield(o, a0)
        if tr:
            print(f"  t={o['t']:4.1f}  K+={o['Ke']:.2f} G={o['G']:.0f} pH={o['pH']:.2f}"
                  f"  | proposed {a0} → safe {a}")
            for line in tr:
                print(f"           · {line}")
        o, r, dead, _ = b.step(a + [0.0], dt=DT)
        if dead:
            print(f"  DIED: ...")
            break
        if o["t"] > 10:
            break
    if b.alive:
        print("  ...survived")


if __name__ == "__main__":
    print("=== (1) SAFETY SHIELD: naive 'insulin+fluids, no K+' policy ===")
    naive = lambda s: [6.0, 500.0, 0.0]
    print(f"  without shield  survived = {run_policy(naive, False)*100:.0f}%  "
          f"(dies of hypokalemia — forgot K+)")
    print(f"  with Osler shield survived = {run_policy(naive, True)*100:.0f}%  "
          f"(shield forces K+ replacement)")

    print("\n=== (2) training L1 world model for mechanism verification ===")
    S, A, S2 = collect(1200)
    wm = WorldModel(); train(wm, S, A, S2, epochs=8)
    verify_mechanisms(wm)

    print("\n=== (3) REASONING TRACE ===")
    show_trace()
