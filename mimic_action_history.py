"""Normalize MIMIC-IV inputevents and eMAR administrations for DKA.

The output contract is one row per canonical action contribution with a total
amount over [starttime, endtime]. Dextrose-containing fluids intentionally emit
both a fluid-volume row and a dextrose-grams row.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from dka_action_contract import ACTION_KEYS, INSULIN_KEYS
from osler_jepa.actions import TREATMENT_EVENT_DIM

ACTION_NAMES = ACTION_KEYS
MAINTENANCE_NAMES = (
    "free_water", "oral_intake", "enteral_nutrition",
    "parenteral_nutrition", "carbohydrate",
)
ACTION_UNITS = {
    **{name: "unit" for name in INSULIN_KEYS},
    "fluids": "ml",
    "kcl": "meq",
    "bicarbonate": "meq",
    "dextrose": "g",
}


def _number(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _integer(value):
    number = _number(value)
    return int(number) if number is not None else 0


def _text(*values):
    return " ".join(str(value or "") for value in values).lower()


def classify_actions(label):
    text = _text(label)
    actions = []
    if "insulin" in text:
        actions.append("insulin")
    if "potassium chloride" in text or re.search(r"\bkcl\b", text):
        actions.append("kcl")
    if "bicarbonate" in text or re.search(r"\bnahco3\b", text):
        actions.append("bicarbonate")
    dextrose = any(token in text for token in (
        "dextrose", "glucose", "d5w", "d10w", "d20w", "d50w",
    ))
    if dextrose:
        actions.append("dextrose")
    fluid = any(token in text for token in (
        "sodium chloride", "normal saline", "saline", "lactated ringer",
        "ringer's", "ringer ", "plasma-lyte", "plasmalyte", "iv fluid",
        "d5w", "d10w", "d20w", "1/2ns",
    )) or bool(re.search(r"\bnacl\b|\bl\.??r\.??\b|\blr\b|\bns\b", text))
    if fluid or (dextrose and "d50" not in text):
        actions.append("fluids")
    return tuple(dict.fromkeys(actions))


def insulin_channel(label, route=None, source=None, rate_uom=None, duration_hours=0.0):
    text = _text(label, route)
    if any(token in text for token in ("glargine", "lantus", "detemir", "levemir", "degludec")):
        return "insulin_basal_sc"
    if any(token in text for token in ("nph", "70/30", "75/25", "protamine")):
        return "insulin_intermediate_sc"
    if any(token in text for token in ("humalog", "novolog", "lispro", "aspart", "apidra", "glulisine")):
        return "insulin_rapid_sc"
    route_text = _text(route)
    if "intraven" in route_text or route_text.strip() == "iv":
        return "insulin_iv"
    if any(token in route_text for token in ("subcut", " sc", "sq")):
        return "insulin_rapid_sc"
    rate_text = _text(rate_uom)
    if source == "inputevents" and (
        "/hr" in rate_text or "hour" in rate_text or duration_hours > 0.75
    ):
        return "insulin_iv"
    return "insulin_rapid_sc"


def default_duration_hours(action, label, route=None):
    """Approximate exposure duration for instantaneous administration records."""
    text = _text(label, route)
    if action in INSULIN_KEYS:
        return 0.5
    if action == "kcl":
        return 4.0 if any(token in text for token in ("tablet", "packet", " po", "oral")) else 1.0
    if action == "bicarbonate":
        return 1.0
    if action == "dextrose":
        return 0.5 if "d50" in text or "50%" in text else 1.0
    if action == "fluids":
        return 0.5
    return 0.5


def dextrose_fraction(label):
    text = _text(label)
    compact = text.replace(" ", "")
    for token, fraction in (
        ("d50", 0.50), ("50%", 0.50), ("d20", 0.20), ("20%", 0.20),
        ("d10", 0.10), ("10%", 0.10), ("d5", 0.05), ("5%", 0.05),
    ):
        if token in compact:
            return fraction
    return None


def fluid_sodium_concentration(label):
    """Approximate sodium concentration for common crystalloid carriers."""
    text = _text(label).replace(" ", "")
    if any(token in text for token in ("d5w", "d10w", "d20w", "sterilewater")):
        return 0.0
    if any(token in text for token in ("1/2ns", "0.45%", "halfnormal")):
        return 77.0
    if any(token in text for token in ("lactatedringer", "ringer's", "ringer")) or text == "lr":
        return 130.0
    if any(token in text for token in ("plasma-lyte", "plasmalyte")):
        return 140.0
    if any(token in text for token in ("normalsaline", "0.9%", "sodiumchloride", "nacl")) or text == "ns":
        return 154.0
    return 140.0


def canonical_amount(action, amount, uom, label, duration_hours=None,
                     rate=None, rate_uom=None, itemid=None):
    """Return total canonical amount for one event, or None if unsafe to infer."""
    amount = _number(amount)
    rate = _number(rate)
    unit = _text(uom).replace(" ", "")
    rate_unit = _text(rate_uom).replace(" ", "")
    duration_hours = max(float(duration_hours or 0.0), 0.0)

    if amount is None and rate is not None and duration_hours > 0:
        amount = rate * duration_hours
        unit = rate_unit.split("/")[0]
    if amount is None:
        return None

    if action in INSULIN_KEYS:
        if "unit" in unit or unit in ("u", "iu", ""):
            return amount
    elif action == "fluids":
        if unit in ("l", "liter", "liters"):
            return amount * 1000.0
        if "ml" in unit or "milliliter" in unit or unit == "":
            return amount
    elif action in ("kcl", "bicarbonate"):
        if "mmol" in unit or "meq" in unit or unit == "":
            return amount
        # MIMIC item 227533 is sodium bicarbonate 8.4% (1 mEq/mL).
        # KCl bolus carrier volume is deliberately not converted to mEq.
        if action == "bicarbonate" and _integer(itemid) == 227533 and "ml" in unit:
            return amount
    elif action == "dextrose":
        if unit in ("g", "gram", "grams") or "gm" in unit:
            return amount
        if unit in ("mg", "milligram", "milligrams"):
            return amount / 1000.0
        if "ml" in unit or unit == "":
            fraction = dextrose_fraction(label)
            if fraction is not None:
                return amount * fraction
    return None


def normalize_events(frame):
    rows = []
    for event in frame.to_dict("records"):
        start = pd.to_datetime(event.get("starttime"), errors="coerce")
        end = pd.to_datetime(event.get("endtime"), errors="coerce")
        if pd.isna(start):
            continue
        if pd.isna(end) or end < start:
            end = start
        duration = max((end - start).total_seconds() / 3600.0, 0.0)
        duration_is_exact = bool(duration > 0.0)
        label = _text(event.get("label"), event.get("product_description"))
        for action in classify_actions(label):
            if action == "insulin":
                action = insulin_channel(
                    label,
                    event.get("route"),
                    event.get("source"),
                    event.get("rate_uom"),
                    duration,
                )
            if action == "fluids" and "flush" in label:
                continue
            if action in INSULIN_KEYS and action != "insulin_iv":
                effective_duration = 0.5
            else:
                effective_duration = duration or default_duration_hours(
                    action, label, event.get("route")
                )
            total = canonical_amount(
                action,
                event.get("amount"),
                event.get("uom"),
                label,
                effective_duration,
                event.get("rate"),
                event.get("rate_uom"),
                event.get("itemid"),
            )
            if total is None or not np.isfinite(total) or total <= 0:
                continue
            rows.append({
                "subject_id": event.get("subject_id"),
                "stay_id": event.get("stay_id"),
                "starttime": start,
                "endtime": (
                    start + pd.Timedelta(hours=effective_duration)
                    if action in INSULIN_KEYS and action != "insulin_iv"
                    else end if duration > 0
                    else start + pd.Timedelta(hours=effective_duration)
                ),
                "action": action,
                "amount": float(total),
                "uom": ACTION_UNITS[action],
                "source": event.get("source", "unknown"),
                "route": event.get("route"),
                "original_label": label,
                "itemid": event.get("itemid"),
                "orderid": event.get("orderid"),
                "timing_source": "recorded_interval" if duration_is_exact else "default_duration",
                "timing_confidence": 1.0 if duration_is_exact else 0.5,
                "dose_source": "recorded_amount" if _number(event.get("amount")) is not None else "rate_x_duration",
                "dose_confidence": 1.0 if _number(event.get("amount")) is not None else 0.75,
                "fluid_sodium_meq_l": (
                    fluid_sodium_concentration(label) if action == "fluids" else np.nan
                ),
            })
    columns = [
        "subject_id", "stay_id", "starttime", "endtime", "action",
        "amount", "uom", "source", "route", "original_label", "itemid",
        "orderid", "timing_source", "timing_confidence", "dose_source",
        "dose_confidence", "fluid_sodium_meq_l",
    ]
    return pd.DataFrame(rows, columns=columns)


def normalize_maintenance_events(frame):
    """Normalize observed nutrition/free-water ingredients without guessing dose.

    Volumes remain contextual maintenance inputs. Only explicitly charted
    glucose/carbohydrate mass is converted to grams; calories are never reverse
    engineered into carbohydrate.
    """
    rows = []
    for event in frame.to_dict("records"):
        start = pd.to_datetime(event.get("starttime"), errors="coerce")
        end = pd.to_datetime(event.get("endtime"), errors="coerce")
        if pd.isna(start):
            continue
        if pd.isna(end) or end <= start:
            end = start + pd.Timedelta(hours=0.5)
        itemid = _integer(event.get("ingredient_itemid") or event.get("itemid"))
        input_label = _text(event.get("input_label"))
        amount = _number(event.get("amount"))
        unit = _text(event.get("uom") or event.get("amountuom")).replace(" ", "")
        if amount is None or amount <= 0:
            continue
        name = None
        canonical_amount = amount
        canonical_unit = "ml"
        if itemid == 226221:
            name = "enteral_nutrition"
        elif itemid == 227079:
            name = "parenteral_nutrition"
        elif itemid == 227075:
            if "free water" in input_label or "flush" in input_label:
                name = "free_water"
            elif "po intake" in input_label:
                name = "oral_intake"
        elif itemid in (220395, 220364):
            if not any(token in input_label for token in ("tpn", "parenteral", "nutrition")):
                continue
            name = "carbohydrate"
            canonical_unit = "g"
            if unit in ("mg", "milligram", "milligrams"):
                canonical_amount = amount / 1000.0
            elif unit not in ("g", "gram", "grams", "gm"):
                continue
        if name is None:
            continue
        rows.append({
            "subject_id": event.get("subject_id"),
            "stay_id": event.get("stay_id"),
            "starttime": start,
            "endtime": end,
            "maintenance": name,
            "amount": float(canonical_amount),
            "uom": canonical_unit,
            "source": "ingredientevents",
            "orderid": event.get("orderid"),
            "input_label": input_label,
            "dose_confidence": 1.0,
            "timing_confidence": 1.0,
        })
    return pd.DataFrame(rows, columns=[
        "subject_id", "stay_id", "starttime", "endtime", "maintenance",
        "amount", "uom", "source", "orderid", "input_label",
        "dose_confidence", "timing_confidence",
    ])


def deduplicate_events(frame, tolerance_minutes=20):
    """Prefer ICU inputevents when the same administration is also in eMAR."""
    if frame.empty or not {"inputevents", "emar"}.issubset(set(frame["source"])):
        return frame.sort_values(["stay_id", "starttime"]).reset_index(drop=True)
    input_rows = frame[frame["source"] == "inputevents"]
    keep = np.ones(len(frame), dtype=bool)
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    for position, (_, event) in enumerate(frame.iterrows()):
        if event["source"] != "emar":
            continue
        candidates = input_rows[
            (input_rows["stay_id"] == event["stay_id"])
            & (input_rows["action"] == event["action"])
            & ((input_rows["starttime"] - event["starttime"]).abs() <= tolerance)
        ]
        if candidates.empty:
            continue
        amount_tolerance = max(1.0, 0.25 * float(event["amount"]))
        if ((candidates["amount"] - float(event["amount"])).abs() <= amount_tolerance).any():
            keep[position] = False
    return frame.iloc[np.flatnonzero(keep)].sort_values(
        ["stay_id", "starttime"]
    ).reset_index(drop=True)


def action_rate_grid(events, start, hours, dt=0.5):
    """Distribute event totals over a regular action-rate grid."""
    cells = int(round(hours / dt))
    grid = np.zeros((cells, len(ACTION_NAMES)), dtype=np.float32)
    end = start + pd.Timedelta(hours=hours)
    for event in events.to_dict("records"):
        event_start = max(pd.Timestamp(event["starttime"]), start)
        event_end = min(pd.Timestamp(event["endtime"]), end)
        if event_end <= event_start:
            event_end = min(event_start + pd.Timedelta(hours=dt), end)
        full_duration = max(
            (pd.Timestamp(event["endtime"]) - pd.Timestamp(event["starttime"])).total_seconds()
            / 3600.0,
            dt,
        )
        action_index = ACTION_NAMES.index(event["action"])
        for cell in range(cells):
            lo = start + pd.Timedelta(hours=cell * dt)
            hi = lo + pd.Timedelta(hours=dt)
            overlap = max(0.0, (min(event_end, hi) - max(event_start, lo)).total_seconds() / 3600.0)
            if overlap:
                grid[cell, action_index] += event["amount"] * overlap / full_duration / dt
    return grid


def maintenance_rate_grid(events, start, hours, dt=0.5):
    cells = int(round(hours / dt))
    grid = np.zeros((cells, len(MAINTENANCE_NAMES)), dtype=np.float32)
    end = start + pd.Timedelta(hours=hours)
    for event in events.to_dict("records"):
        event_start = max(pd.Timestamp(event["starttime"]), start)
        event_end = min(pd.Timestamp(event["endtime"]), end)
        if event_end <= event_start:
            continue
        full_duration = max(
            (pd.Timestamp(event["endtime"]) - pd.Timestamp(event["starttime"])).total_seconds()
            / 3600.0,
            dt,
        )
        index = MAINTENANCE_NAMES.index(event["maintenance"])
        for cell in range(cells):
            lo = start + pd.Timedelta(hours=cell * dt)
            hi = lo + pd.Timedelta(hours=dt)
            overlap = max(0.0, (min(event_end, hi) - max(event_start, lo)).total_seconds() / 3600.0)
            if overlap:
                grid[cell, index] += float(event["amount"]) * overlap / full_duration / dt
    return grid


def maintenance_window_summary(events, anchor, history_hours=6.0,
                               future_hours=6.0, dt=0.5):
    history_start = anchor - pd.Timedelta(hours=history_hours)
    future_end = anchor + pd.Timedelta(hours=future_hours)
    history = events[
        (events["endtime"] >= history_start) & (events["starttime"] < anchor)
    ]
    future = events[
        (events["endtime"] >= anchor) & (events["starttime"] < future_end)
    ]
    history_grid = maintenance_rate_grid(history, history_start, history_hours, dt)
    future_grid = maintenance_rate_grid(future, anchor, future_hours, dt)
    output = {
        "history_maintenance_grid": history_grid.tolist(),
        "future_maintenance_grid": future_grid.tolist(),
        "maintenance_sources": sorted(events["source"].dropna().unique().tolist())
        if not events.empty else [],
    }
    for index, name in enumerate(MAINTENANCE_NAMES):
        output[f"hist_{name}_total"] = float(history_grid[:, index].sum() * dt)
        output[f"act_{name}_total"] = float(future_grid[:, index].sum() * dt)
    return output


def fluid_sodium_grid(events, start, hours, dt=0.5):
    """Volume-weighted fluid sodium concentration for each action cell."""
    cells = int(round(hours / dt))
    sodium_mass = np.zeros(cells, dtype=np.float32)
    fluid_volume = np.zeros(cells, dtype=np.float32)
    end = start + pd.Timedelta(hours=hours)
    for event in events[events["action"] == "fluids"].to_dict("records"):
        event_start = max(pd.Timestamp(event["starttime"]), start)
        event_end = min(pd.Timestamp(event["endtime"]), end)
        if event_end <= event_start:
            event_end = min(event_start + pd.Timedelta(hours=dt), end)
        full_duration = max(
            (pd.Timestamp(event["endtime"]) - pd.Timestamp(event["starttime"])).total_seconds()
            / 3600.0,
            dt,
        )
        concentration = float(event.get("fluid_sodium_meq_l", 140.0))
        for cell in range(cells):
            lo = start + pd.Timedelta(hours=cell * dt)
            hi = lo + pd.Timedelta(hours=dt)
            overlap = max(0.0, (min(event_end, hi) - max(event_start, lo)).total_seconds() / 3600.0)
            if overlap:
                volume = float(event["amount"]) * overlap / full_duration
                fluid_volume[cell] += volume
                sodium_mass[cell] += volume * concentration
    return np.divide(
        sodium_mass,
        fluid_volume,
        out=np.full(cells, 140.0, dtype=np.float32),
        where=fluid_volume > 0,
    )


def action_quality_summary(events):
    if events.empty:
        return {
            "event_count": 0,
            "exact_timing_fraction": 0.0,
            "high_confidence_dose_fraction": 0.0,
            "sources": [],
        }
    return {
        "event_count": int(len(events)),
        "exact_timing_fraction": float((events["timing_source"] == "recorded_interval").mean()),
        "high_confidence_dose_fraction": float((events["dose_confidence"] >= 0.75).mean()),
        "sources": sorted(events["source"].dropna().astype(str).unique().tolist()),
    }


def treatment_event_records(events, start, hours):
    """Return aggregate start/stop lifecycle events within a time window."""
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(hours=hours)
    records = []
    for action in ACTION_NAMES:
        selected = events[events["action"] == action]
        active = int(((selected["starttime"] < start) & (selected["endtime"] > start)).sum())
        if active:
            records.append({
                "hour": 0.0, "event_type": "start", "action": action,
                "carried_in": True,
            })
        boundaries = {}
        for event in selected.to_dict("records"):
            event_start = pd.Timestamp(event["starttime"])
            event_end = pd.Timestamp(event["endtime"])
            if start <= event_start < end:
                boundaries[event_start] = boundaries.get(event_start, 0) + 1
            if start < event_end < end:
                boundaries[event_end] = boundaries.get(event_end, 0) - 1
        for timestamp, delta in sorted(boundaries.items()):
            before = active
            active = max(0, active + delta)
            if before == 0 and active > 0:
                event_type = "start"
            elif before > 0 and active == 0:
                event_type = "stop"
            else:
                continue
            records.append({
                "hour": round((timestamp - start).total_seconds() / 3600.0, 6),
                "event_type": event_type,
                "action": action,
                "carried_in": False,
            })
    return sorted(records, key=lambda item: (item["hour"], item["action"], item["event_type"]))


def treatment_event_grid(events, start, hours, dt=0.5):
    """Map exact lifecycle records to start/stop event channels."""
    cells = int(round(hours / dt))
    grid = np.zeros((cells, TREATMENT_EVENT_DIM), dtype=np.float32)
    for event in treatment_event_records(events, start, hours):
        cell = min(cells - 1, max(0, int(float(event["hour"]) // dt)))
        offset = 0 if event["event_type"] == "start" else len(ACTION_NAMES)
        grid[cell, offset + ACTION_NAMES.index(event["action"])] = 1.0
    return grid


def action_window_summary(events, anchor, history_hours=6.0, future_hours=6.0, dt=0.5):
    history_start = anchor - pd.Timedelta(hours=history_hours)
    future_end = anchor + pd.Timedelta(hours=future_hours)
    history_events = events[
        (events["endtime"] >= history_start) & (events["starttime"] < anchor)
    ]
    future_events = events[
        (events["endtime"] >= anchor) & (events["starttime"] < future_end)
    ]
    history_grid = action_rate_grid(history_events, history_start, history_hours, dt)
    future_grid = action_rate_grid(future_events, anchor, future_hours, dt)
    history_event_grid = treatment_event_grid(
        history_events, history_start, history_hours, dt
    )
    future_event_grid = treatment_event_grid(
        future_events, anchor, future_hours, dt
    )
    output = {
        "history_action_grid": history_grid.tolist(),
        "future_action_grid": future_grid.tolist(),
        "history_treatment_event_grid": history_event_grid.tolist(),
        "future_treatment_event_grid": future_event_grid.tolist(),
        "history_treatment_events": treatment_event_records(
            history_events, history_start, history_hours
        ),
        "future_treatment_events": treatment_event_records(
            future_events, anchor, future_hours
        ),
        "history_fluid_sodium_grid": fluid_sodium_grid(
            history_events, history_start, history_hours, dt
        ).tolist(),
        "future_fluid_sodium_grid": fluid_sodium_grid(
            future_events, anchor, future_hours, dt
        ).tolist(),
        "history_action_quality": action_quality_summary(history_events),
        "future_action_quality": action_quality_summary(future_events),
    }
    for index, action in enumerate(ACTION_NAMES):
        output[f"hist_{action}_total"] = float(history_grid[:, index].sum() * dt)
        output[f"act_{action}_total"] = float(future_grid[:, index].sum() * dt)
        active = np.flatnonzero(history_grid[:, index] > 0)
        output[f"hist_{action}_hours_since"] = (
            float((len(history_grid) - 1 - active[-1]) * dt) if len(active)
            else float(history_hours)
        )
    insulin_indices = [ACTION_NAMES.index(name) for name in INSULIN_KEYS]
    output["hist_insulin_total"] = float(history_grid[:, insulin_indices].sum() * dt)
    output["act_insulin_total"] = float(future_grid[:, insulin_indices].sum() * dt)
    insulin_active = np.flatnonzero(history_grid[:, insulin_indices].sum(axis=1) > 0)
    output["hist_insulin_hours_since"] = (
        float((len(history_grid) - 1 - insulin_active[-1]) * dt)
        if len(insulin_active) else float(history_hours)
    )
    return output
