"""Interpretable heme/coagulation belief features for candidate-only audits.

These are inferred personalization features, not measured labs.  They summarize
bleeding pressure, coagulation reserve, oxygen-carrying reserve, platelet
reserve, and observation confidence from currently available hematology,
coagulation, perfusion, and treatment evidence.  They are useful only if they
improve downstream observable prediction versus both a baseline and a
capacity-matched placebo.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


HEME_COAG_BELIEF_COLUMNS = (
    "belief_bleeding_burden",
    "belief_coagulation_reserve",
    "belief_oxygen_carrying_reserve",
    "belief_platelet_reserve",
    "belief_transfusion_pressure",
    "belief_anticoagulation_pressure",
    "belief_rbc_transfusion_exposure",
    "belief_hemostatic_product_exposure",
    "belief_transfusion_dose_intensity",
    "belief_hemostatic_stress",
    "belief_observation_confidence",
)

HEME_COAG_STATE_COLUMNS = (
    "state_belief_coag_reserve_mean",
    "state_belief_coag_reserve_sd",
    "state_belief_bleeding_burden",
    "state_belief_oxygen_reserve",
    "state_belief_platelet_reserve",
    "state_belief_observation_confidence",
    "state_belief_delta_since_prior",
)


def _num(frame: pd.DataFrame, column: str, default=np.nan) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _finite_clip(values: np.ndarray, low: float, high: float, fill: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    output[~np.isfinite(output)] = fill
    return np.clip(output, low, high)


def _clip01(values: np.ndarray | float) -> np.ndarray | float:
    return np.clip(values, 0.0, 1.0)


def _scalar(row: pd.Series, column: str, default: float = np.nan) -> float:
    value = row.get(column, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _flag(row: pd.Series, column: str) -> bool:
    return _scalar(row, column, 0.0) > 0.0


def _action_any(frame: pd.DataFrame, stem: str) -> np.ndarray:
    return (
        (_num(frame, f"hist_{stem}", default=0.0) > 0.0)
        | (_num(frame, f"act_{stem}", default=0.0) > 0.0)
    ).astype(np.float64)


def _transfusion_unit_exposure(frame: pd.DataFrame, product: str | None = None) -> np.ndarray:
    stems = ["transfusion"] if product is None else [f"transfusion_{product}"]
    values = np.zeros(len(frame), dtype=np.float64)
    for stem in stems:
        for prefix in ("hist", "act"):
            unit_column = f"{prefix}_{stem}_unit_like_count"
            volume_column = f"{prefix}_{stem}_volume_like_ml"
            count_column = f"{prefix}_{stem}_evidence_count"
            flag_column = f"{prefix}_{stem}"
            if unit_column in frame:
                values += _finite_clip(_num(frame, unit_column), 0.0, 20.0, 0.0)
            elif volume_column in frame:
                values += _finite_clip(_num(frame, volume_column), 0.0, 6000.0, 0.0) / 300.0
            elif count_column in frame:
                values += _finite_clip(_num(frame, count_column), 0.0, 20.0, 0.0)
            elif flag_column in frame:
                values += (_num(frame, flag_column, default=0.0) > 0.0).astype(np.float64)
    return np.clip(values, 0.0, 20.0)


def _observation_confidence(frame: pd.DataFrame) -> np.ndarray:
    ages = []
    for column in (
        "hemoglobin_age_hr",
        "hematocrit_age_hr",
        "platelets_age_hr",
        "inr_age_hr",
        "ptt_age_hr",
        "fibrinogen_age_hr",
    ):
        if column in frame:
            ages.append(_finite_clip(_num(frame, column), 0.0, 72.0, 72.0))
    if not ages:
        return np.full(len(frame), 0.15, dtype=np.float64)
    stacked = np.vstack(ages)
    freshness = np.clip(1.0 - stacked / 72.0, 0.0, 1.0)
    availability = np.isfinite(stacked).astype(np.float64)
    confidence = 0.20 + 0.80 * np.nanmean(freshness * availability, axis=0)
    return np.clip(confidence, 0.05, 1.0)


def heme_coag_belief_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return candidate heme/coagulation belief features for downstream gates."""

    hemoglobin = _finite_clip(_num(frame, "hemoglobin_t"), 1.0, 25.0, 10.0)
    hematocrit = _finite_clip(_num(frame, "hematocrit_t"), 3.0, 75.0, 30.0)
    platelets = _finite_clip(_num(frame, "platelets_t"), 1.0, 1000.0, 180.0)
    inr = _finite_clip(_num(frame, "inr_t"), 0.4, 20.0, 1.2)
    ptt = _finite_clip(_num(frame, "ptt_t"), 10.0, 250.0, 35.0)
    fibrinogen = _finite_clip(_num(frame, "fibrinogen_t"), 20.0, 1500.0, 250.0)
    map_value = _finite_clip(_num(frame, "map_t"), 25.0, 180.0, 75.0)
    lactate = _finite_clip(_num(frame, "lactate_t"), 0.2, 25.0, 1.5)

    transfusion = _action_any(frame, "transfusion")
    anticoagulant = _action_any(frame, "anticoagulant")
    antiplatelet = _action_any(frame, "antiplatelet")
    vasopressor = _action_any(frame, "vasopressor")
    fluids = _action_any(frame, "fluids")
    rbc_units = _transfusion_unit_exposure(frame, "prbc")
    plasma_units = _transfusion_unit_exposure(frame, "plasma")
    platelet_units = _transfusion_unit_exposure(frame, "platelet")
    cryo_units = _transfusion_unit_exposure(frame, "cryo")
    total_units = _transfusion_unit_exposure(frame)
    rbc_exposure = _clip01(rbc_units / 3.0)
    hemostatic_product_exposure = _clip01((plasma_units + platelet_units + cryo_units) / 3.0)
    transfusion_dose_intensity = _clip01(total_units / 5.0)

    anemia_risk = 0.65 * _clip01((8.0 - hemoglobin) / 3.0) + 0.35 * _clip01((24.0 - hematocrit) / 9.0)
    platelet_risk = _clip01((100.0 - platelets) / 80.0)
    inr_risk = _clip01((inr - 1.4) / 2.6)
    ptt_risk = _clip01((ptt - 45.0) / 75.0)
    fibrinogen_risk = _clip01((180.0 - fibrinogen) / 130.0)
    shock_risk = 0.55 * _clip01((65.0 - map_value) / 30.0) + 0.45 * _clip01((lactate - 2.0) / 6.0)

    coagulation_reserve = _clip01(1.0 - (0.40 * inr_risk + 0.25 * ptt_risk + 0.25 * fibrinogen_risk + 0.10 * anticoagulant))
    oxygen_carrying_reserve = _clip01(0.60 * ((hemoglobin - 7.0) / 5.0) + 0.40 * ((hematocrit - 21.0) / 15.0))
    platelet_reserve = _clip01((platelets - 50.0) / 150.0)
    transfusion_pressure = _clip01(0.45 * transfusion + 0.25 * transfusion_dose_intensity + 0.20 * anemia_risk + 0.10 * shock_risk)
    anticoagulation_pressure = _clip01(0.65 * anticoagulant + 0.25 * antiplatelet + 0.10 * ptt_risk)
    hemostatic_stress = _clip01(
        0.30 * platelet_risk
        + 0.25 * inr_risk
        + 0.15 * ptt_risk
        + 0.15 * fibrinogen_risk
        + 0.10 * shock_risk
        + 0.05 * vasopressor
    )
    bleeding_burden = _clip01(
        0.30 * anemia_risk
        + 0.25 * hemostatic_stress
        + 0.20 * transfusion_pressure
        + 0.15 * shock_risk
        + 0.10 * fluids
    )
    observation_confidence = _observation_confidence(frame)

    return pd.DataFrame({
        "belief_bleeding_burden": bleeding_burden,
        "belief_coagulation_reserve": coagulation_reserve,
        "belief_oxygen_carrying_reserve": oxygen_carrying_reserve,
        "belief_platelet_reserve": platelet_reserve,
        "belief_transfusion_pressure": transfusion_pressure,
        "belief_anticoagulation_pressure": anticoagulation_pressure,
        "belief_rbc_transfusion_exposure": rbc_exposure,
        "belief_hemostatic_product_exposure": hemostatic_product_exposure,
        "belief_transfusion_dose_intensity": transfusion_dose_intensity,
        "belief_hemostatic_stress": hemostatic_stress,
        "belief_observation_confidence": observation_confidence,
    }, index=frame.index)


def _row_observation(row: pd.Series) -> tuple[float, float, float, float, float, float]:
    features = heme_coag_belief_features(pd.DataFrame([row]))
    bleeding = float(features.iloc[0]["belief_bleeding_burden"])
    coag_reserve = float(features.iloc[0]["belief_coagulation_reserve"])
    oxygen = float(features.iloc[0]["belief_oxygen_carrying_reserve"])
    platelet = float(features.iloc[0]["belief_platelet_reserve"])
    confidence = float(features.iloc[0]["belief_observation_confidence"])
    variance = float(np.clip(0.04 + (1.0 - confidence) * 0.35 + bleeding * 0.08, 0.03, 1.0))
    return coag_reserve, variance, bleeding, oxygen, platelet, confidence


@dataclass(frozen=True)
class CoagulationReserveBelief:
    """Gaussian belief over hemostatic reserve.

    This is an inferred state for online personalization.  It is not a lab and
    must pass downstream observable gates before use.
    """

    mean: float
    variance: float
    source: str = "coagulation_observation_prior"

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(max(float(self.variance), 1e-6))

    @classmethod
    def from_row(cls, row: pd.Series) -> "CoagulationReserveBelief":
        mean, variance, _bleeding, _oxygen, _platelet, _confidence = _row_observation(row)
        return cls(mean=mean, variance=variance, source="current_observation")

    def predict(self, row: pd.Series, delta_hours: float) -> "CoagulationReserveBelief":
        hours = max(0.0, float(delta_hours))
        _reserve, _variance, bleeding, _oxygen, _platelet, _confidence = _row_observation(row)
        transfusion = 1.0 if _flag(row, "hist_transfusion") or _flag(row, "act_transfusion") else 0.0
        dose_units = _scalar(row, "hist_transfusion_unit_like_count", 0.0) + _scalar(row, "act_transfusion_unit_like_count", 0.0)
        dose_intensity = float(np.clip(dose_units / 5.0, 0.0, 1.0))
        anticoag = 1.0 if _flag(row, "hist_anticoagulant") or _flag(row, "act_anticoagulant") else 0.0
        antiplatelet = 1.0 if _flag(row, "hist_antiplatelet") or _flag(row, "act_antiplatelet") else 0.0
        drift = (0.006 * transfusion + 0.010 * dose_intensity - 0.008 * anticoag - 0.006 * antiplatelet - 0.006 * bleeding) * hours
        mean = float(np.clip(self.mean + drift, 0.0, 1.0))
        process_variance = (0.006 + 0.015 * bleeding + 0.006 * anticoag + 0.003 * transfusion + 0.003 * dose_intensity) * max(hours, 0.25)
        return CoagulationReserveBelief(
            mean=mean,
            variance=float(np.clip(self.variance + process_variance, 0.03, 2.0)),
            source="predicted_from_previous_belief",
        )

    def update(self, row: pd.Series) -> "CoagulationReserveBelief":
        observation, observation_variance, _bleeding, _oxygen, _platelet, _confidence = _row_observation(row)
        prior_var = max(float(self.variance), 1e-6)
        obs_var = max(float(observation_variance), 1e-6)
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
        posterior_mean = posterior_var * (self.mean / prior_var + observation / obs_var)
        return CoagulationReserveBelief(
            mean=float(np.clip(posterior_mean, 0.0, 1.0)),
            variance=float(np.clip(posterior_var, 0.01, 2.0)),
            source="predict_update_observation",
        )

    def to_features(self, row: pd.Series, prior_mean: float | None = None) -> dict[str, float]:
        _reserve, _variance, bleeding, oxygen, platelet, confidence = _row_observation(row)
        return {
            "state_belief_coag_reserve_mean": float(self.mean),
            "state_belief_coag_reserve_sd": float(self.standard_deviation),
            "state_belief_bleeding_burden": float(bleeding),
            "state_belief_oxygen_reserve": float(oxygen),
            "state_belief_platelet_reserve": float(platelet),
            "state_belief_observation_confidence": float(confidence),
            "state_belief_delta_since_prior": (
                float(self.mean - prior_mean) if prior_mean is not None else 0.0
            ),
        }


def heme_coag_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return explicit predict-update coagulation reserve belief features."""

    output = pd.DataFrame(
        np.nan,
        index=frame.index,
        columns=HEME_COAG_STATE_COLUMNS,
        dtype=np.float64,
    )
    if frame.empty:
        return output
    sort_column = "hours_since_onset" if "hours_since_onset" in frame else None
    group_column = "stay_id" if "stay_id" in frame else None
    groups = frame.groupby(group_column, sort=False) if group_column else [(None, frame)]
    for _stay_id, group in groups:
        ordered = group.sort_values(sort_column) if sort_column else group
        belief: CoagulationReserveBelief | None = None
        previous_hour: float | None = None
        previous_row: pd.Series | None = None
        for index, row in ordered.iterrows():
            hour = _scalar(row, "hours_since_onset", 0.0)
            if belief is None:
                belief = CoagulationReserveBelief.from_row(row)
                prior_mean = None
            else:
                delta_hours = max(0.0, hour - (previous_hour if previous_hour is not None else hour))
                prior = belief.predict(previous_row if previous_row is not None else row, delta_hours)
                prior_mean = prior.mean
                belief = prior.update(row)
            features = belief.to_features(row, prior_mean=prior_mean)
            for column, value in features.items():
                output.loc[index, column] = value
            previous_hour = hour
            previous_row = row
    return output.astype(np.float64)


def placebo_heme_coag_belief_features(
    frame: pd.DataFrame,
    seed: int,
    columns: tuple[str, ...] = HEME_COAG_BELIEF_COLUMNS,
) -> pd.DataFrame:
    """Return capacity-matched uninformative placebo columns."""

    rng = np.random.default_rng(int(seed))
    values = rng.normal(0.0, 1.0, size=(len(frame), len(columns)))
    return pd.DataFrame(values, columns=[f"placebo_{name}" for name in columns], index=frame.index)
