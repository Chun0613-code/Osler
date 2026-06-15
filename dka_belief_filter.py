"""
dka_belief_filter.py  (L1 as explicit belief-filter)
====================================================
Turn the implicit belief the DRQN learned into an EXPLICIT, interpretable
world-model filter (RSSM-style predict-update):

  prior:     z_t^-  = P(z_{t-1}, a_{t-1})           # roll latent forward (L1 predictor)
  posterior: z_t    = U(z_t^-, obs_t, mask_t)        # correct toward the measurement
  decode:    s_hat  = D(z_t)                         # read out FULL physiology incl. hidden

Trained on partial-obs sim rollouts with PRIVILEGED supervision (sim gives the true
full state). At deployment it sees only sparse noisy labs.

Validate:
  (1) hidden-state inference: filter's K+ estimate vs true K+ at steps where K+ was
      NOT measured -- compared to the naive 'last-observed K+'. Also V and I (never
      observed at all).
  (2) MPC on the belief -> survival on the real partial-obs env.

Run:  python dka_belief_filter.py
"""

import itertools
import random
import numpy as np
import torch
import torch.nn as nn
from dka_body import DKABody

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# full state (8) order + normalization
S_MEAN = np.array([250, 7.2, 15, 18, 4.2, 80, 13, 15.], dtype=np.float32)
S_STD = np.array([200, 0.3, 10, 10, 1.5, 30, 4, 25.], dtype=np.float32)
SK = ["G", "pH", "HCO3", "anion_gap", "Ke", "MAP", "V", "I"]   # state idx 0..7
S_DIM = 8
# observable vars: (state_idx, per-step prob, noise std).  HARD regime (rare K+/ABG)
OBSV = {"glucose": (0, 0.20, 18.0), "Ke": (4, 0.08, 0.25), "pH": (1, 0.06, 0.02),
        "HCO3": (2, 0.06, 1.0), "anion_gap": (3, 0.06, 1.0), "MAP": (5, 0.60, 3.0)}
OBS_IDX = [v[0] for v in OBSV.values()]          # state indices that can be observed
O_DIM = len(OBSV)

INSULIN, FLUIDS, KCL = [0., 2., 4., 6.], [0., 250., 500.], [0., 10., 20.]
ACT_GRID = [list(a) for a in itertools.product(INSULIN, FLUIDS, KCL)]
N_ACT, A_SCALE = len(ACT_GRID), np.array([6., 500., 20.], dtype=np.float32)
Z_DIM, DT, HORIZON = 32, 0.5, 48


def true_vec(o):
    return ((np.array([o[k] for k in SK], dtype=np.float32) - S_MEAN) / S_STD)


# measure: partial noisy measurement -> (obs_norm[O_DIM], mask[O_DIM]) at state scale
OBS_ATTR = ["G", "Ke", "pH", "HCO3", "anion_gap", "MAP"]
OBS_P = [0.20, 0.08, 0.06, 0.06, 0.06, 0.60]
OBS_NOISE = [18.0, 0.25, 0.02, 1.0, 1.0, 3.0]
OBS2STATE = [0, 4, 1, 2, 3, 5]


def measure(o):
    om = np.zeros(O_DIM, np.float32); mk = np.zeros(O_DIM, np.float32)
    for i in range(O_DIM):
        if np.random.rand() < OBS_P[i]:
            idx = OBS2STATE[i]
            val = o[OBS_ATTR[i]] + np.random.randn() * OBS_NOISE[i]
            om[i] = (val - S_MEAN[idx]) / S_STD[idx]; mk[i] = 1.0
    return om, mk


def behavior_action():
    if random.random() < 0.5:   # treatmentish -> covers survivable region
        return [random.choice([2., 4., 6.]), random.choice([250., 500.]),
                random.choice([0., 10., 20.])]
    return ACT_GRID[random.randrange(N_ACT)]


def collect(n_roll=1500):
    b = DKABody(); eps = []
    for _ in range(n_roll):
        b.reset()
        b.G = float(np.random.uniform(350, 650)); b.Ke = float(np.random.uniform(4.8, 6.2))
        b.HCO3 = float(np.random.uniform(6, 12)); b.Ket = float(np.random.uniform(8, 14))
        b.V = float(np.random.uniform(10, 13))
        S, OM, MK, PA = [], [], [], []
        prev_a = np.zeros(3, np.float32)
        for _ in range(HORIZON):
            o = b.observe()
            om, mk = measure(o)
            S.append(true_vec(o)); OM.append(om); MK.append(mk); PA.append(prev_a)
            a = behavior_action()
            o2, r, dead, _ = b.step(a + [0.0], dt=DT)
            prev_a = np.array(a, np.float32) / A_SCALE
            if dead:
                break
        eps.append((np.array(S), np.array(OM), np.array(MK), np.array(PA)))
    return eps


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Filter(nn.Module):
    def __init__(self):
        super().__init__()
        self.P = mlp(Z_DIM + 3, Z_DIM)
        self.U = mlp(Z_DIM + O_DIM * 2, Z_DIM)
        self.D = mlp(Z_DIM, S_DIM)
        self.z0 = nn.Parameter(torch.zeros(Z_DIM))

    def prior(self, z, a):
        return self.P(torch.cat([z, a], -1))

    def post(self, zprior, om, mk):
        return self.U(torch.cat([zprior, om, mk], -1))


def vicreg(z):
    std = torch.sqrt(z.var(0) + 1e-4)
    return torch.mean(torch.relu(1.0 - std))


def train(filt, eps, epochs=12, bs=16):
    opt = torch.optim.Adam(filt.parameters(), 1e-3)
    for ep in range(epochs):
        random.shuffle(eps); tot = 0.0; nb = 0
        for i in range(0, len(eps), bs):
            batch = eps[i:i + bs]
            T = max(len(e[0]) for e in batch); B = len(batch)
            S = np.zeros((B, T, S_DIM), np.float32); OM = np.zeros((B, T, O_DIM), np.float32)
            MK = np.zeros((B, T, O_DIM), np.float32); PA = np.zeros((B, T, 3), np.float32)
            M = np.zeros((B, T), np.float32)
            for j, (s, om, mk, pa) in enumerate(batch):
                L = len(s); S[j, :L] = s; OM[j, :L] = om; MK[j, :L] = mk
                PA[j, :L] = pa; M[j, :L] = 1.0
            S = torch.tensor(S); OM = torch.tensor(OM); MK = torch.tensor(MK)
            PA = torch.tensor(PA); M = torch.tensor(M)
            z = filt.z0.expand(B, Z_DIM)
            loss = 0.0
            for t in range(T):
                zp = filt.prior(z, PA[:, t]) if t > 0 else filt.z0.expand(B, Z_DIM)
                zpost = filt.post(zp, OM[:, t], MK[:, t])
                shat = filt.D(zpost); sprior = filt.D(zp)
                m = M[:, t:t + 1]
                l_post = (((shat - S[:, t]) ** 2) * m).sum() / (m.sum() + 1e-6) / S_DIM
                l_prior = (((sprior - S[:, t]) ** 2) * m).sum() / (m.sum() + 1e-6) / S_DIM
                # obs-consistency only on measured dims
                dobs = shat[:, OBS2STATE]
                l_obs = (((dobs - OM[:, t]) ** 2) * MK[:, t] * m).sum() / (MK[:, t].sum() + 1e-6)
                loss = loss + l_post + 0.5 * l_prior + l_obs + 0.05 * vicreg(zpost)
                z = zpost
            loss = loss / T
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(filt.parameters(), 5.0)
            opt.step(); tot += loss.item(); nb += 1
        if (ep + 1) % 4 == 0:
            print(f"  ep {ep+1:2d}  loss={tot/nb:.4f}")


# ---------- validation: hidden-state inference ------------------------------
def validate_inference(filt, n=250):
    b = DKABody()
    f_err = {4: [], 6: [], 7: []}   # Ke, V, I  (state idx)
    last_err = []                   # naive last-observed Ke (only when never re-measured? compare at unmeasured steps)
    for _ in range(n):
        b.reset()
        b.G = float(np.random.uniform(350, 650)); b.Ke = float(np.random.uniform(4.8, 6.2))
        b.HCO3 = float(np.random.uniform(6, 12)); b.Ket = float(np.random.uniform(8, 14))
        b.V = float(np.random.uniform(10, 13))
        z = filt.z0.unsqueeze(0); prev_a = np.zeros(3, np.float32)
        last_ke = None
        for t in range(HORIZON):
            o = b.observe(); om, mk = measure(o)
            zp = filt.prior(z, torch.tensor(prev_a).unsqueeze(0)) if t > 0 else filt.z0.unsqueeze(0)
            z = filt.post(zp, torch.tensor(om).unsqueeze(0), torch.tensor(mk).unsqueeze(0))
            est = filt.D(z).detach().numpy()[0] * S_STD + S_MEAN
            ke_measured = mk[1] > 0
            if ke_measured:
                last_ke = om[1] * S_STD[4] + S_MEAN[4]
            else:   # inference steps only
                f_err[4].append(abs(est[4] - o["Ke"]))
                if last_ke is not None:
                    last_err.append(abs(last_ke - o["Ke"]))
            f_err[6].append(abs(est[6] - o["V"]))   # V never observed
            f_err[7].append(abs(est[7] - o["I"]))   # I never observed
            a = behavior_action()
            b.step(a + [0.0], dt=DT); prev_a = np.array(a, np.float32) / A_SCALE
            if not b.alive:
                break
    print("\nhidden-state inference (filter vs true), MAE:")
    print(f"  K+  at UNMEASURED steps : filter {np.mean(f_err[4]):.2f}   "
          f"naive last-observed {np.mean(last_err):.2f}")
    print(f"  V   (never observed)    : filter {np.mean(f_err[6]):.2f}")
    print(f"  I   (never observed)    : filter {np.mean(f_err[7]):.2f}")


# ---------- MPC on the belief -----------------------------------------------
def reward_batch(s):
    p = s * torch.tensor(S_STD) + torch.tensor(S_MEAN)
    G, pH, Ke, MAP = p[:, 0], p[:, 1], p[:, 4], p[:, 5]
    drive = ((pH - 7.4) / 0.2) ** 2 + ((Ke - 4.2) / 1.5) ** 2 + ((G - 100) / 200) ** 2 + ((MAP - 90) / 30) ** 2
    dead = (pH < 6.8) | (Ke < 2.5) | (Ke > 7.0) | (MAP < 40) | (G < 40) | (G > 1000)
    return torch.where(dead, torch.full_like(drive, -100.0), 1.0 - drive)


def mpc(filt, z, H=5, N=300, gamma=0.97):
    grid = torch.tensor(np.array(ACT_GRID, np.float32))
    aidx = torch.randint(N_ACT, (N, H))
    aseq = grid[aidx] / torch.tensor(A_SCALE)
    zz = z.repeat(N, 1); total = torch.zeros(N)
    for h in range(H):
        zz = filt.prior(zz, aseq[:, h]); total += (gamma ** h) * reward_batch(filt.D(zz))
    return ACT_GRID[int(aidx[int(total.argmax()), 0])]


def eval_mpc(filt, n=60):
    b = DKABody(); surv = 0
    for _ in range(n):
        b.reset()
        b.G = float(np.random.uniform(350, 650)); b.Ke = float(np.random.uniform(4.8, 6.2))
        b.HCO3 = float(np.random.uniform(6, 12)); b.Ket = float(np.random.uniform(8, 14))
        b.V = float(np.random.uniform(10, 13))
        z = filt.z0.unsqueeze(0); prev_a = np.zeros(3, np.float32)
        for t in range(HORIZON):
            o = b.observe(); om, mk = measure(o)
            zp = filt.prior(z, torch.tensor(prev_a).unsqueeze(0)) if t > 0 else filt.z0.unsqueeze(0)
            z = filt.post(zp, torch.tensor(om).unsqueeze(0), torch.tensor(mk).unsqueeze(0))
            with torch.no_grad():
                a = mpc(filt, z)
            b.step(a + [0.0], dt=DT); prev_a = np.array(a, np.float32) / A_SCALE
            if not b.alive:
                break
        surv += b.alive
    print(f"\nMPC on belief   survived 24h on real partial-obs sim = {surv/n*100:.0f}%")
    print("  (memoryless DQN was 63%, recurrent DRQN 98% in this regime)")


if __name__ == "__main__":
    print("collecting partial-obs rollouts (privileged true state for training)...")
    eps = collect()
    print(f"  {len(eps)} episodes")
    print("training RSSM belief-filter...")
    filt = Filter(); train(filt, eps)
    with torch.no_grad():
        validate_inference(filt)
    eval_mpc(filt)
