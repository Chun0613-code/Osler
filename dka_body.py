"""
dka_body.py
===========
A DKA "mini-body": ~5 coupled compartment pools written ONLY from objective,
low-level facts (mass balance + transfer kinetics + what kills you). Nothing about
"DKA" is scripted. The acidotic cascade and the lethal insulin->hypokalemia crisis
EMERGE from the couplings + the agent's (in)action.

State pools
  G    plasma glucose            (mg/dL)
  I    plasma insulin signal     (mU/L)
  Ket  circulating ketoacid      (mEq/L)   -> drives anion gap
  HCO3 bicarbonate               (mEq/L)   -> with pCO2 gives pH
  Ke   plasma (extracellular) K+ (mEq/L)
  Ki   total-body K+ reserve     (mEq-equivalent; not a transcellular pool)
  V    extracellular fluid vol   (L)       -> gives MAP, renal perfusion
  Na   serum sodium               (mEq/L)   -> with glucose gives effective osmolality
  Cr   serum creatinine           (mg/dL)   -> lagging renal-perfusion marker
  CRS  counter-regulatory stress (0..2; resolves over hours with treatment)
  RPS  renal-perfusion state     (0..1; lags instantaneous MAP)
  OI   cumulative hyperosmolar injury      (dimensionless burden)
  D*   rapid/NPH/basal SC insulin depots   (U awaiting absorption)

Actions (administration RATES, held over the decision step)
  IV insulin, rapid-SC insulin, intermediate/NPH insulin, basal insulin,
  fluids, KCl, bicarbonate, and dextrose.

Death requires sustained severity outside the viability kernel. Brief numerical
crossings accumulate a reversible burden rather than causing an instant terminal
event.

Patient profiles vary weight, renal reserve, insulin sensitivity,
counter-regulatory stress, fluid retention, vascular tone, and total-body K+.
The neutral profile is the reference patient; sampled profiles create
patient-level response heterogeneity for world-model training.

Constants are first-pass, tuned for QUALITATIVE correctness. Calibrating them
against real MIMIC DKA trajectories (replay real action sequences, compare emergent
curves) is the "exam" -- that is what MIMIC is for here, not training.
"""

import math
from dataclasses import asdict, dataclass
import numpy as np

from dka_action_contract import ACTION_INDEX, ACTION_KEYS, expand_action

# ---- setpoints / structural constants (objective facts) --------------------
G_NORM, RENAL_G_THRESH = 100.0, 180.0     # mg/dL; kidney spills glucose above thresh
HCO3_NORM = 24.0
KE_NORM = 4.2
V_NORM = 15.0                             # L extracellular fluid
MAP_NORM, MAP_MIN = 90.0, 30.0
AG_BASE = 12.0
I_BASAL = 1.0                             # near-absent endogenous insulin (DKA)
I50 = 20.0                                # insulin signal at half-max effect

# transfer-kinetic rates (per hour unless noted)
K_HEP = 40.0          # basal hepatic glucose output (mg/dL/hr equiv)
K_UPTAKE_BASAL = 0.05 # insulin-independent glucose disposal
K_UPTAKE_INS = 0.13   # insulin-mediated glucose disposal
K_RENAL_G = 0.10      # glucosuria rate above threshold
K_INS_ABS = 60.0      # U/hr -> plasma insulin signal (therapeutic ~6U/hr -> strong effect)
K_INS_CLEAR = 6.0     # IV insulin clearance (~7 min half-life)
K_SC_RAPID_ABS = 0.90
K_SC_INTERMEDIATE_ABS = 0.20
K_SC_BASAL_ABS = 0.055
K_KETO = 1.5          # ketogenesis when insulin absent
K_KETCLEAR_BASAL = 0.02
K_KETCLEAR_INS = 0.45 # insulin promotes ketone disposal (regenerates HCO3)
K_RENAL_KET = 0.02
K_RENAL_HCO3 = 0.008  # weak renal bicarb regen (kidney can't outpace ongoing ketoacidosis)
K_K_INS = 0.15        # insulin shifts K+ INTO cells  (calibrated down from 0.35)
K_K_ACID = 0.30       # acidosis shifts K+ OUT of cells
K_K_BUF = 0.25        # plasma K+ buffered toward equilibrium by intracellular pool
KI_REF = 140.0        # reference intracellular K+ store (buffer capacity scales w/ Ki/KI_REF)
K_URINE_K_BASE = 8.0  # plausible urinary K concentration at low osmotic flow (mEq/L)
K_URINE_K_OSM = 12.0  # additional urinary K concentration with osmotic diuresis
K_STRESS_ADAPT = 0.35 # counter-regulatory stress approaches current illness drive / hr
K_RENAL_PERF_ADAPT = 0.70  # renal perfusion approaches MAP target / hr
K_OSM = 0.020         # osmotic diuresis per unit glucosuria
K_INSENS = 0.06       # insensible fluid loss (L/hr)
PCO2_FLOOR = 12.0     # respiratory compensation limit (Kussmaul fatigue)
NA_INFUSATE = 140.0   # generic isotonic-fluid sodium approximation (mEq/L)
NA_URINE = 100.0      # effective urinary osmole concentration (mEq/L equivalent)
K_CR_ADAPT = 0.25     # creatinine approaches perfusion-dependent target per hour
K_OSM_INJURY_RECOVERY = 0.12
OSM_INJURY_DEATH = 12.0
CRITICAL_BURDEN_DEATH = 1.0
CRITICAL_BURDEN_RECOVERY = 0.75

CRITICAL_LIMITS = {
    "acidosis (pH<6.8)": ("below", 6.8, 0.20),
    "hypokalemia (K<2.5)": ("below", 2.5, 0.50),
    "hyperkalemia (K>7.0)": ("above", 7.0, 1.00),
    "circulatory collapse (MAP<40)": ("below", 40.0, 15.0),
    "hypoglycemia (G<40)": ("below", 40.0, 20.0),
    "extreme hyperglycemia (G>1400)": ("above", 1400.0, 300.0),
}


@dataclass(frozen=True)
class DKAPatientProfile:
    """Patient-level physiology used for simulator domain randomization.

    These are research priors, not clinical parameter estimates. The neutral
    profile supplies the reference physiology; sampled profiles deliberately
    create response heterogeneity that a deterministic average model cannot
    explain from one laboratory snapshot alone.
    """

    weight_kg: float = 75.0
    renal_reserve: float = 1.0
    insulin_sensitivity: float = 1.0
    counterregulatory_drive: float = 1.0
    fluid_retention: float = 1.0
    vascular_tone: float = 1.0
    potassium_store_scale: float = 1.0
    endogenous_insulin: float = I_BASAL
    baseline_sodium: float = 138.0
    baseline_creatinine: float = 0.9

    @classmethod
    def sample(cls, rng):
        return cls(
            weight_kg=float(np.clip(rng.normal(78.0, 18.0), 45.0, 135.0)),
            renal_reserve=float(rng.uniform(0.55, 1.10)),
            insulin_sensitivity=float(np.clip(rng.lognormal(0.0, 0.28), 0.50, 1.70)),
            counterregulatory_drive=float(rng.uniform(0.75, 1.55)),
            fluid_retention=float(rng.uniform(0.65, 1.00)),
            vascular_tone=float(rng.uniform(0.82, 1.15)),
            potassium_store_scale=float(rng.uniform(0.65, 1.05)),
            endogenous_insulin=float(rng.uniform(0.5, 3.0)),
            baseline_sodium=float(rng.uniform(132.0, 145.0)),
            baseline_creatinine=float(rng.uniform(0.6, 1.4)),
        )

    def to_dict(self):
        return asdict(self)


def henderson(hco3):
    pco2 = max(PCO2_FLOOR, min(40.0, 1.5 * hco3 + 8.0))   # Winter's compensation
    hco3 = max(hco3, 1e-3)
    return 6.1 + math.log10(hco3 / (0.03 * pco2))


def estimate_potassium_store(ke, ph, creatinine=1.2, urine_output=100.0,
                             prior_kcl_meq=0.0):
    """Mechanistic prior for latent total-body potassium reserve.

    Serum K alone is misleading in DKA because acidosis shifts intracellular K
    outward. This prior combines acidemia, measured hypokalemia, osmotic urine
    loss, renal impairment, and documented replacement. It is intentionally
    exposed as a belief estimate rather than presented as a measured lab value.
    """
    acid_deficit = max(0.0, 7.35 - float(ph))
    low_serum_penalty = 18.0 * max(0.0, 4.0 - float(ke))
    urine_penalty = 0.04 * max(0.0, float(urine_output) - 100.0)
    renal_retention = 4.0 * max(0.0, float(creatinine) - 1.2)
    replacement = 0.45 * max(0.0, float(prior_kcl_meq))
    return float(np.clip(
        120.0 - 45.0 * acid_deficit - low_serum_penalty - urine_penalty
        + renal_retention + replacement,
        50.0, 150.0,
    ))


class DKABody:
    def __init__(self, rng=None, profile=None, residual_model=None):
        self.rng = rng or np.random.default_rng()
        self.profile = profile or DKAPatientProfile()
        self.residual_model = residual_model
        self.reset()

    @property
    def volume_setpoint(self):
        return V_NORM * self.profile.weight_kg / 75.0

    def set_profile(self, profile, reset=True):
        self.profile = profile
        if reset:
            self.reset()

    def reset(self):
        # a patient presenting in DKA (high glucose, ketoacidosis, dehydrated,
        # plasma K+ high-ish from acidosis despite depleted TOTAL body K+)
        self.G = 480.0
        self.I = self.profile.endogenous_insulin
        self.Ket = 11.0
        self.HCO3 = 8.0
        self.Ke = 5.6
        self.Ki = 120.0 * self.profile.potassium_store_scale
        self.V = 0.8 * self.volume_setpoint
        self.Na = self.profile.baseline_sodium
        self.Cr = self.profile.baseline_creatinine / self.profile.renal_reserve
        self.counterregulatory_stress = self.profile.counterregulatory_drive
        self.renal_perfusion_state = self._instantaneous_renal_perfusion()
        self.insulin_rapid_depot = 0.0
        self.insulin_intermediate_depot = 0.0
        self.insulin_basal_depot = 0.0
        self.osmotic_injury = 0.0
        self.critical_burdens = {
            cause: 0.0 for cause in CRITICAL_LIMITS
        }
        self.urine_output_ml_hr = 0.0
        self.t = 0.0
        self.alive = True
        self.death_cause = None
        return self.observe()

    # --- derived, never stored as independent state -------------------------
    @property
    def pH(self):
        return henderson(self.HCO3)

    @property
    def MAP(self):
        filling = self.V / self.volume_setpoint * self.profile.vascular_tone
        return MAP_MIN + (MAP_NORM - MAP_MIN) * min(1.2, filling)

    @property
    def anion_gap(self):
        return AG_BASE + self.Ket

    @property
    def beta_hydroxybutyrate(self):
        return 0.75 * self.Ket

    @property
    def effective_osmolality(self):
        return 2.0 * self.Na + self.G / 18.0

    @property
    def renal_func(self):
        # Renal clearance follows a lagged perfusion state. This avoids making
        # filtration recover instantly after the first fluid bolus.
        return max(0.0, min(1.0, self.renal_perfusion_state))

    def _instantaneous_renal_perfusion(self):
        perfusion = (self.MAP - MAP_MIN) / (MAP_NORM - MAP_MIN)
        return max(0.0, min(1.0, perfusion * self.profile.renal_reserve))

    def insulin_effect(self):
        effective_i50 = I50 / self.profile.insulin_sensitivity
        return self.I / (self.I + effective_i50)

    # --- one integration substep (dt hours) ---------------------------------
    def _derivs(self, action, fluid_sodium_meq_l=NA_INFUSATE,
                free_water_ml=0.0, nutrition_carbohydrate_g=0.0):
        values = expand_action(action)
        insulin_iv = values[ACTION_INDEX["insulin_iv"]]
        insulin_rapid = values[ACTION_INDEX["insulin_rapid_sc"]]
        insulin_intermediate = values[ACTION_INDEX["insulin_intermediate_sc"]]
        insulin_basal = values[ACTION_INDEX["insulin_basal_sc"]]
        fluid_ml = values[ACTION_INDEX["fluids"]]
        kcl = values[ACTION_INDEX["kcl"]]
        bicarb = values[ACTION_INDEX["bicarbonate"]]
        dextrose_g = values[ACTION_INDEX["dextrose"]]
        ie = self.insulin_effect()
        rf = self.renal_func

        # glucose mass balance
        prod = K_HEP * self.counterregulatory_stress * (1.0 - 0.82 * ie)
        uptake = (K_UPTAKE_BASAL + K_UPTAKE_INS * ie) * self.G
        glucosuria = K_RENAL_G * max(0.0, self.G - RENAL_G_THRESH) * rf

        # volume balance (osmotic diuresis driven by glucosuria)
        osm_diuresis = K_OSM * glucosuria
        basal_urine = 0.0007 * self.profile.weight_kg * rf
        urine_l_hr = basal_urine + osm_diuresis
        retained_iv_fluid = fluid_ml / 1000.0 * self.profile.fluid_retention
        retained_free_water = max(0.0, float(free_water_ml)) / 1000.0 * 0.85
        retained_fluid = retained_iv_fluid + retained_free_water
        dV = retained_fluid - urine_l_hr - K_INSENS
        dil = dV / self.V   # fractional dilution applied to concentrations

        dextrose_input = (
            dextrose_g + max(0.0, float(nutrition_carbohydrate_g))
        ) * 100.0 / self.V
        dG = prod + dextrose_input - uptake - glucosuria - self.G * dil
        weight_scale = 75.0 / self.profile.weight_kg
        rapid_absorbed = K_SC_RAPID_ABS * self.insulin_rapid_depot
        intermediate_absorbed = K_SC_INTERMEDIATE_ABS * self.insulin_intermediate_depot
        basal_absorbed = K_SC_BASAL_ABS * self.insulin_basal_depot
        delivered_insulin = insulin_iv + rapid_absorbed + intermediate_absorbed + basal_absorbed
        dI = (delivered_insulin * K_INS_ABS * weight_scale
              - K_INS_CLEAR * (self.I - self.profile.endogenous_insulin))
        dRapid = insulin_rapid - rapid_absorbed
        dIntermediate = insulin_intermediate - intermediate_absorbed
        dBasal = insulin_basal - basal_absorbed

        # ketoacid balance: production (insulin-suppressed) vs disposal (insulin-promoted)
        kg = K_KETO * self.counterregulatory_stress * (1.0 - ie)
        kl = (K_KETCLEAR_BASAL + K_KETCLEAR_INS * ie) * self.Ket + K_RENAL_KET * self.Ket * rf
        dKet = kg - kl - self.Ket * dil

        # bicarbonate: consumed by new acid, regenerated by ketone disposal + kidney + given
        dHCO3 = (-kg + kl
                 + bicarb / self.V
                 + K_RENAL_HCO3 * (HCO3_NORM - self.HCO3) * rf
                 - self.HCO3 * dil)

        # Potassium uses two distinct concepts. Serum K changes with
        # transcellular shifts, while Ki is a total-body reserve changed only by
        # true intake/output. Insulin therefore cannot destroy potassium mass.
        acid_drive = max(0.0, 7.4 - self.pH)
        shift_in = K_K_INS * ie * self.Ke
        shift_out = K_K_ACID * acid_drive * (self.Ki / 120.0)
        urinary_k_concentration = K_URINE_K_BASE + K_URINE_K_OSM * min(1.0, osm_diuresis)
        renal_k_loss = urine_l_hr * urinary_k_concentration * rf
        renal_k_serum = renal_k_loss / max(self.V, 1.0)
        buf = K_K_BUF * (KE_NORM - self.Ke) * (self.Ki / KI_REF)   # restoring force
        dKe = (-shift_in + shift_out + 0.55 * kcl / self.V
               - renal_k_serum + buf - self.Ke * dil)
        dKi = kcl - renal_k_loss

        # Sodium concentration follows solute and water balance. Bicarbonate is
        # administered as a sodium salt in this simplified IV action contract.
        sodium_in = retained_iv_fluid * float(fluid_sodium_meq_l) + bicarb
        sodium_out = urine_l_hr * NA_URINE
        dNa = (sodium_in - sodium_out - self.Na * dV) / self.V
        # The mini-body does not explicitly model every urinary osmole or
        # intracellular water shift, so bound concentration change to a
        # physiologically plausible hourly rate.
        dNa = float(np.clip(dNa, -1.5, 1.5))

        # Creatinine is a lagging proxy for renal reserve and current perfusion.
        cr_target = (self.profile.baseline_creatinine / self.profile.renal_reserve) * (
            1.0 + 1.8 * (1.0 - rf)
        )
        dCr = K_CR_ADAPT * (cr_target - self.Cr) - self.Cr * dil

        # Stress and renal perfusion are hidden dynamic state, not immutable
        # patient traits. Acidosis, ketosis, hyperglycemia and hypovolemia raise
        # stress; insulin and restoration of volume let it resolve gradually.
        illness_drive = (
            0.20
            + 0.45 * min(1.5, max(0.0, self.G - 180.0) / 300.0)
            + 0.55 * min(1.5, self.Ket / 12.0)
            + 0.45 * min(1.5, max(0.0, 7.35 - self.pH) / 0.35)
            + 0.35 * min(1.5, max(0.0, self.volume_setpoint - self.V)
                         / (0.25 * self.volume_setpoint))
        )
        stress_target = self.profile.counterregulatory_drive * illness_drive * (1.0 - 0.35 * ie)
        dStress = K_STRESS_ADAPT * (stress_target - self.counterregulatory_stress)
        perfusion_target = self._instantaneous_renal_perfusion()
        dRenalPerfusion = K_RENAL_PERF_ADAPT * (perfusion_target - self.renal_perfusion_state)

        # Neurologic hyperosmolar risk depends on severity and duration. It is
        # accumulated rather than triggered by one instantaneous threshold crossing.
        osm_excess = max(0.0, (self.effective_osmolality - 320.0) / 20.0)
        glucose_excess = max(0.0, (self.G - 800.0) / 300.0)
        injury_input = osm_excess ** 2 + glucose_excess ** 2
        recovery = K_OSM_INJURY_RECOVERY * self.osmotic_injury
        dOsmoticInjury = injury_input - recovery

        derivatives = dict(
            G=dG, I=dI, Ket=dKet, HCO3=dHCO3, Ke=dKe, Ki=dKi,
            V=dV, Na=dNa, Cr=dCr, urine_output_ml_hr=urine_l_hr * 1000.0,
            insulin_rapid_depot=dRapid,
            insulin_intermediate_depot=dIntermediate,
            insulin_basal_depot=dBasal,
            osmotic_injury=dOsmoticInjury,
            counterregulatory_stress=dStress,
            renal_perfusion_state=dRenalPerfusion,
        )
        if self.residual_model is not None:
            residual_action = dict(zip(ACTION_KEYS, values.astype(float).tolist()))
            if isinstance(action, dict):
                residual_action.update({
                    key: value for key, value in action.items()
                    if str(key).startswith("_")
                })
            correction = self.residual_model.correction(
                self.observe(), residual_action
            )
            for key in ("G", "Ket", "HCO3", "Ke", "Na", "Cr"):
                derivatives[key] += float(correction.get(key, 0.0))
        return derivatives

    def step(self, action, dt=0.5, substeps=30):
        """Advance physiology under a route-aware or legacy action vector."""
        fluid_sodium_meq_l = (
            float(action.get("_fluid_sodium_meq_l", NA_INFUSATE))
            if isinstance(action, dict) else NA_INFUSATE
        )
        free_water_ml = (
            float(action.get("_free_water_ml", 0.0))
            if isinstance(action, dict) else 0.0
        )
        nutrition_carbohydrate_g = (
            float(action.get("_nutrition_carbohydrate_g", 0.0))
            if isinstance(action, dict) else 0.0
        )
        values = expand_action(action)
        h = dt / substeps
        for _ in range(substeps):
            if not self.alive:
                break
            derivative_action = dict(action) if isinstance(action, dict) else values
            d = self._derivs(
                derivative_action,
                fluid_sodium_meq_l=fluid_sodium_meq_l,
                free_water_ml=free_water_ml,
                nutrition_carbohydrate_g=nutrition_carbohydrate_g,
            )
            self.G = max(0.0, self.G + d["G"] * h)
            self.I = max(0.0, self.I + d["I"] * h)
            self.Ket = max(0.0, self.Ket + d["Ket"] * h)
            self.HCO3 = max(0.1, self.HCO3 + d["HCO3"] * h)
            self.Ke = max(0.0, self.Ke + d["Ke"] * h)
            self.Ki = max(0.0, self.Ki + d["Ki"] * h)
            self.V = max(1.0, self.V + d["V"] * h)
            self.Na = max(100.0, min(180.0, self.Na + d["Na"] * h))
            self.Cr = max(0.1, self.Cr + d["Cr"] * h)
            self.insulin_rapid_depot = max(
                0.0, self.insulin_rapid_depot + d["insulin_rapid_depot"] * h
            )
            self.insulin_intermediate_depot = max(
                0.0, self.insulin_intermediate_depot
                + d["insulin_intermediate_depot"] * h
            )
            self.insulin_basal_depot = max(
                0.0, self.insulin_basal_depot + d["insulin_basal_depot"] * h
            )
            self.osmotic_injury = max(
                0.0, self.osmotic_injury + d["osmotic_injury"] * h
            )
            self.counterregulatory_stress = max(
                0.05, min(2.5, self.counterregulatory_stress
                          + d["counterregulatory_stress"] * h)
            )
            self.renal_perfusion_state = max(
                0.0, min(1.0, self.renal_perfusion_state
                         + d["renal_perfusion_state"] * h)
            )
            self.urine_output_ml_hr = max(0.0, d["urine_output_ml_hr"])
            self.t += h
            self._check_death(h)
        return self.observe(), self.reward(), (not self.alive), {"cause": self.death_cause}

    def _check_death(self, dt):
        """Accumulate duration-sensitive critical physiology burden.

        A brief threshold crossing is not equivalent to sustained organ failure.
        Burden grows with severity and decays after the variable returns to the
        viability region. These are simulator calibration priors, not clinical
        mortality thresholds.
        """
        values = {
            "acidosis (pH<6.8)": self.pH,
            "hypokalemia (K<2.5)": self.Ke,
            "hyperkalemia (K>7.0)": self.Ke,
            "circulatory collapse (MAP<40)": self.MAP,
            "hypoglycemia (G<40)": self.G,
            "extreme hyperglycemia (G>1400)": self.G,
        }
        for cause, (direction, threshold, scale) in CRITICAL_LIMITS.items():
            value = values[cause]
            excess = (
                max(0.0, threshold - value)
                if direction == "below" else max(0.0, value - threshold)
            )
            burden = self.critical_burdens[cause]
            if excess > 0.0:
                severity = excess / scale
                burden += (0.35 + severity) * dt
            else:
                burden = max(0.0, burden - CRITICAL_BURDEN_RECOVERY * dt)
            self.critical_burdens[cause] = burden

        if self.osmotic_injury > OSM_INJURY_DEATH:
            self.alive, self.death_cause = False, "cumulative hyperosmolar injury"
            return
        lethal = [
            (burden, cause) for cause, burden in self.critical_burdens.items()
            if burden >= CRITICAL_BURDEN_DEATH
        ]
        if lethal:
            _, self.death_cause = max(lethal)
            self.alive = False

    def observe(self):
        return dict(
            G=self.G, pH=self.pH, HCO3=self.HCO3, anion_gap=self.anion_gap,
            Ke=self.Ke, MAP=self.MAP, V=self.V, I=self.I, Na=self.Na,
            osmolality=self.effective_osmolality, creatinine=self.Cr,
            urine_output=self.urine_output_ml_hr,
            BHB=self.beta_hydroxybutyrate,
            K_store=self.Ki,
            counterregulatory_stress=self.counterregulatory_stress,
            renal_perfusion_state=self.renal_perfusion_state,
            osmotic_injury=self.osmotic_injury,
            critical_burden=max(self.critical_burdens.values(), default=0.0),
            critical_burdens=dict(self.critical_burdens),
            insulin_rapid_depot=self.insulin_rapid_depot,
            insulin_intermediate_depot=self.insulin_intermediate_depot,
            insulin_basal_depot=self.insulin_basal_depot,
            t=self.t,
        )

    def reward(self):
        """Homeostatic survival reward: stay alive, stay near viable setpoints."""
        if not self.alive:
            return -100.0
        drive = ((self.pH - 7.4) / 0.2) ** 2 \
            + ((self.Ke - KE_NORM) / 1.5) ** 2 \
            + ((self.G - G_NORM) / 200.0) ** 2 \
            + ((self.MAP - MAP_NORM) / 30.0) ** 2
        return 1.0 - drive   # +1/step alive, minus distance-from-viable


# ---- demo: emergence of the three classic outcomes (no scripting) ----------
def run(policy, label, hours=24, dt=0.5):
    b = DKABody()
    log = []
    n = int(hours / dt)
    for _ in range(n):
        a = policy(b.observe())
        obs, r, done, info = b.step(a, dt=dt)
        log.append(obs)
        if done:
            break
    o = b.observe()
    status = f"DIED @ {b.t:4.1f}h [{b.death_cause}]" if not b.alive else f"survived {hours}h"
    print(f"{label:<26}{status:<34} "
          f"G={o['G']:6.1f} pH={o['pH']:.2f} HCO3={o['HCO3']:4.1f} "
          f"K={o['Ke']:.2f} MAP={o['MAP']:4.0f}")
    return b


if __name__ == "__main__":
    print("Initial DKA state: G=480 pH=%.2f HCO3=8 K=5.6 (dehydrated)\n"
          % henderson(8.0))

    run(lambda s: [0, 0, 0, 0], "1. no treatment")
    run(lambda s: [8, 0, 0, 0], "2. insulin only")
    run(lambda s: [6, 500, 0, 0], "3. insulin + fluids (no K)")
    run(lambda s: [6, 500, 10, 0], "4. insulin + fluids + KCl")

    print("\nNote: nobody coded 'insulin causes hypokalemia'. It emerges from the")
    print("K+ shift rule + renal wasting. Scenario 2/3 should die of low K+;")
    print("only scenario 4 (replacing K+) should survive -- which is real DKA care.")
