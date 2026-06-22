"""
prescription.py — turn a symbolic-engine recommendation into a (sandbox) e-prescription.

The engine only ever RECOMMENDS; a licensed clinician REVIEWS, edits the sig, picks a
pharmacy and SIGNS inside the prescribe UI (Photon's certified workflow, or — with no
credentials — a local mock that mirrors it). That human-in-the-loop split is the same
philosophy the rest of Osler is built on, and it's also what the law requires.

Responsibilities:
  • map an engine `candidate` (drug, label dose, safety) → a prescription draft
  • SAFETY GATE: refuse to start an Rx the engine flagged avoid/block, and refuse
    controlled substances in the demo (those need EPCS / DEA 2-factor — out of scope)
  • hold prescriptions in memory (like demo_app._CACHE) — a demo session starts fresh
  • PHOTON mode when credentials exist, MOCK mode otherwise (offline-demoable)

State is intentionally process-local; production swaps this for a real datastore.
"""
from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional

import photon_client as PH

# Engine safety decisions that hard-block prescribing (mirrors DrugCard's `isBad`).
BLOCKING_DECISIONS = {"avoid", "block"}

# Controlled substances are excluded from the sandbox demo: e-prescribing them needs
# EPCS (DEA two-factor + audited software). Keep this list aligned with data/drugs_pkpd.json.
CONTROLLED = {
    "morphine", "fentanyl", "oxycodone", "hydrocodone", "hydromorphone",
    "codeine", "methadone", "buprenorphine", "tramadol", "diazepam",
    "lorazepam", "alprazolam", "clonazepam", "midazolam", "phenobarbital",
    "ketamine", "methylphenidate", "amphetamine", "lisdexamfetamine",
}

# In-memory stores. _RX: rx_id → record. _BY_PATIENT: patient_id → [rx_id...].
_RX: Dict[str, Dict[str, Any]] = {}
_BY_PATIENT: Dict[str, List[str]] = {}


class RxError(Exception):
    """Raised when a prescription cannot be started (safety gate / bad input).
    demo_app turns this into a 400 with .message."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _find_candidate(bundle: Dict[str, Any], drug: str) -> Optional[Dict[str, Any]]:
    want = (drug or "").strip().lower()
    for c in (bundle.get("result") or {}).get("candidates", []):
        if (c.get("drug") or "").strip().lower() == want:
            return c
    return None


def _prefill_sig(candidate: Dict[str, Any]) -> str:
    """Seed the sig from the FDA-label dose, but ONLY the verbatim text the engine
    cleared for display. This is NOT a structured sig — the clinician confirms/edits
    it in the prescribe UI. Empty string when the safety gate withheld a dose."""
    dose = candidate.get("dose") or {}
    if dose.get("verbatim") and dose.get("patient_specific_allowed"):
        return dose["verbatim"]
    return ""


def _synthesize_identity(patient_id: str, result_patient: Dict[str, Any]) -> Dict[str, str]:
    """Demo cases are anonymous (e.g. "64M ACS"). Photon needs a real patient record,
    so we mint a stable sandbox identity from the case. Sandbox data only."""
    age = result_patient.get("age")
    # AWSDate wants YYYY-MM-DD; approximate a DOB from age against a fixed ref year so
    # the same case always maps to the same sandbox patient.
    birth_year = (2025 - int(age)) if isinstance(age, (int, float)) else 1980
    return {
        "external_id": f"osler-demo-{patient_id}",
        "given": "Demo",
        "family": f"Patient {patient_id}"[:35],
        "dob": f"{birth_year:04d}-01-01",
        "sex": result_patient.get("sex") or "unknown",
        "phone": "+12025550102",  # sandbox placeholder — valid NANP area code (555 area is rejected)
    }


def start(patient_id: str, drug: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Begin a prescription for `drug` from the cached analysis `bundle`.
    Returns a record the client opens in the prescribe WebView. Raises RxError if the
    safety gate blocks it."""
    candidate = _find_candidate(bundle, drug)
    if not candidate:
        raise RxError(f"'{drug}' is not a candidate in this analysis.")

    decision = (candidate.get("safety") or {}).get("decision", "").lower()
    if decision in BLOCKING_DECISIONS:
        raise RxError(f"Safety gate blocks prescribing {drug} (decision: {decision}).")

    if (drug or "").strip().lower() in CONTROLLED:
        raise RxError(f"{drug} is a controlled substance — excluded from the sandbox "
                      "demo (requires EPCS / DEA two-factor).")

    rx_id = "rx_" + uuid.uuid4().hex[:10]
    result_patient = (bundle.get("result") or {}).get("patient") or {}

    photon_patient_id = None
    mode = "mock"
    if PH.is_enabled():
        mode = "photon"
        ident = _synthesize_identity(patient_id, result_patient)
        # Failure here shouldn't 500 the demo — fall back to mock so the UX still flows.
        try:
            # Unique external id per Rx so repeat prescribing of the same case never
            # collides on createPatient (sandbox patient proliferation is harmless).
            photon_patient_id = PH.create_or_sync_patient(
                external_id=f"{ident['external_id']}-{rx_id}", given=ident["given"],
                family=ident["family"], dob=ident["dob"], sex=ident["sex"],
                phone=ident["phone"])
        except Exception as e:  # noqa: BLE001 — degrade gracefully in a demo
            mode = "mock"
            photon_patient_id = None
            print(f"[prescribe] Photon patient sync failed, using mock: {e}")

    # Photon hosted prescribe page (opened in the system browser) when synced.
    photon_prescribe_url = (PH.prescribe_url(photon_patient_id)
                            if mode == "photon" and photon_patient_id else None)

    record = {
        "rx_id": rx_id,
        "patient_id": patient_id,
        "drug": candidate.get("drug"),
        "clinical_role": (candidate.get("clinical_role") or {}).get("label"),
        "safety_decision": decision or "ok",
        "sig": _prefill_sig(candidate),       # editable in the UI
        "dispense_quantity": None,
        "dispense_unit": None,
        "days_supply": None,
        "refills": 0,
        "pharmacy": None,                      # {id, name}
        "status": "draft",                     # draft → sent → filled (or error)
        "mode": mode,
        "photon_patient_id": photon_patient_id,
        "photon_prescribe_url": photon_prescribe_url,
        "photon_prescription_id": None,
        "photon_order_id": None,
        "est_price": None,
        "source_rationale": candidate.get("final_answer") or candidate.get("rationale"),
        "mechanism_chain": candidate.get("mechanism_chain"),
    }
    _RX[rx_id] = record
    _BY_PATIENT.setdefault(patient_id, []).append(rx_id)
    return record


# ── native Sign & Send (provider-token path) ─────────────────────────────────
# The catalog search ranks combination products oddly (e.g. "Zegerid With Magnesium
# Hydroxide (… Omeprazole 40 mg …)" outranks plain omeprazole), so we re-rank toward the
# single-ingredient generic before offering a default treatment to sign.
def _ingredient_count(name: str) -> int:
    """Rough active-ingredient count from a Photon catalog name (each strength = one ' mg'
    / ' mcg' / ' unit' token)."""
    n = name.lower()
    return max(1, sum(n.count(u) for u in (" mg", " mcg", " unit", " %")))


def _rank_candidates(drug: str, meds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order catalog hits so a plain generic of `drug` sorts first (fewest ingredients,
    name actually starts with the drug, shortest)."""
    d = (drug or "").strip().lower()

    def score(m: Dict[str, Any]) -> tuple:
        name = (m.get("name") or "")
        nl = name.lower()
        combo = (_ingredient_count(name) - 1) * 10
        if " with " in nl or "/" in nl:
            combo += 15
        starts = 0 if nl.startswith(d) else 5
        absent = 0 if d in nl else 100
        return (absent, combo, starts, len(name))

    return sorted(meds, key=score)


def _dispense_unit_for_form(form: Optional[str]) -> str:
    """A sane default Photon dispenseUnit from the dosage form (clinician can edit)."""
    f = (form or "").lower()
    if any(k in f for k in ("tablet", "capsule", "patch", "suppository", "lozenge")):
        return "Each"
    if any(k in f for k in ("solution", "suspension", "syrup", "liquid", "drops", "/ml", " ml")):
        return "Milliliter"
    if any(k in f for k in ("cream", "ointment", "gel", "powder")):
        return "Gram"
    return "Each"


def options(rx_id: str) -> Dict[str, Any]:
    """Catalog choices + prefilled defaults for the native review screen. Uses the M2M
    token (search is read-only); signing later uses the provider token. In mock/offline
    mode, return editable defaults instead of failing the review screen."""
    rec = _RX.get(rx_id)
    if not rec:
        raise RxError("Unknown prescription.")
    drug = rec["drug"] or ""
    # Fetch wide (20) then rank: the catalog buries single-ingredient generics behind
    # combination products, so a small `first` would default to a combo. Offer the top 10.
    catalog_error = None
    meds: List[Dict[str, Any]] = []
    if PH.is_enabled():
        try:
            meds = _rank_candidates(drug, PH.search_medications(drug, 20))[:10]
        except Exception as e:  # noqa: BLE001 - keep the clinician review screen usable
            catalog_error = str(e)
            print(f"[prescribe] Photon catalog search failed, using manual defaults: {e}")
    else:
        catalog_error = "Photon is not configured; manual mock defaults are shown."
    candidates = [{"treatment_id": m.get("id"), "name": m.get("name"),
                   "strength": m.get("strength"), "form": m.get("form")} for m in meds]
    best = candidates[0] if candidates else None
    return {
        "rx_id": rx_id,
        "drug": drug,
        "clinical_role": rec.get("clinical_role"),
        "rationale": rec.get("source_rationale"),
        "catalog_error": catalog_error,
        "candidates": candidates,
        "default": {
            "treatment_id": best["treatment_id"] if best else None,
            "sig": rec.get("sig") or "Take as directed.",
            "dispense_quantity": 30,
            "dispense_unit": _dispense_unit_for_form(best["form"]) if best else "Each",
            "days_supply": 30,
            "refills": 0,
        },
    }


def sign(rx_id: str, *, provider_token: str, treatment_id: str, sig: str,
         dispense_quantity: float, dispense_unit: str, days_supply: int,
         refills: int, address: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Provider's in-app Sign & Send: createPrescription → createOrder under the provider
    token, then finalize the local record. Re-checks the safety gate as defense in depth.
    Raises RxError on a gate/input problem; lets Photon transport errors propagate."""
    rec = _RX.get(rx_id)
    if not rec:
        raise RxError("Unknown prescription.")
    if (rec.get("safety_decision") or "").lower() in BLOCKING_DECISIONS:
        raise RxError(f"Safety gate blocks prescribing {rec.get('drug')}.")
    if (rec.get("drug") or "").strip().lower() in CONTROLLED:
        raise RxError(f"{rec.get('drug')} is a controlled substance — not available in the demo.")
    if not treatment_id:
        raise RxError("Pick a medication to prescribe.")
    patient_id = rec.get("photon_patient_id")
    if not patient_id:
        raise RxError("Patient is not synced to Photon — re-open this prescription.")

    rx_photon_id = PH.create_prescription(
        patient_id=patient_id, treatment_id=treatment_id, sig=sig,
        dispense_quantity=float(dispense_quantity), dispense_unit=dispense_unit,
        days_supply=int(days_supply), refills=int(refills), token=provider_token)
    order_id = PH.create_order(
        patient_id=patient_id, prescription_id=rx_photon_id,
        address=address, token=provider_token)

    return complete(rx_id, sig=sig, dispense_quantity=float(dispense_quantity),
                    dispense_unit=dispense_unit, days_supply=int(days_supply),
                    refills=int(refills), pharmacy=None,
                    photon_prescription_id=rx_photon_id, photon_order_id=order_id)


def context(rx_id: str) -> Dict[str, Any]:
    """Everything the embed page needs to render the prescribe UI (mock or Photon)."""
    rec = _RX.get(rx_id)
    if not rec:
        raise RxError("Unknown prescription.")
    ctx: Dict[str, Any] = {
        "rx_id": rec["rx_id"],
        "mode": rec["mode"],
        "drug": rec["drug"],
        "clinical_role": rec["clinical_role"],
        "safety_decision": rec["safety_decision"],
        "sig_prefill": rec["sig"],
        "rationale": rec["source_rationale"],
        "status": rec["status"],
    }
    if rec["mode"] == "photon":
        # Photon Elements authenticate the provider via the SPA client id (NOT the M2M
        # one) + org id; both are public client-side. The M2M secret stays server-side.
        ctx["photon"] = {
            "spaClientId": PH.SPA_CLIENT_ID,
            "orgId": PH.ORG_ID,
            "devMode": not PH.IS_PROD,
            "env": PH.env_label(),
            "patientId": rec["photon_patient_id"],
            "appOrigin": PH.APP_ORIGIN,
        }
    return ctx


def complete(rx_id: str, *, sig: str, dispense_quantity: Optional[float],
             dispense_unit: Optional[str], days_supply: Optional[int],
             refills: int, pharmacy: Optional[Dict[str, str]],
             photon_prescription_id: Optional[str] = None,
             photon_order_id: Optional[str] = None) -> Dict[str, Any]:
    """Clinician signed & sent. Records the finalized prescription. In MOCK mode this
    is the terminal 'sent' state; in PHOTON mode the webhook may further advance it."""
    rec = _RX.get(rx_id)
    if not rec:
        raise RxError("Unknown prescription.")
    rec.update({
        "sig": sig or rec["sig"],
        "dispense_quantity": dispense_quantity,
        "dispense_unit": dispense_unit,
        "days_supply": days_supply,
        "refills": int(refills or 0),
        "pharmacy": pharmacy,
        "status": "sent",
        "photon_prescription_id": photon_prescription_id or rec["photon_prescription_id"],
        "photon_order_id": photon_order_id or rec["photon_order_id"],
    })
    return rec


def apply_webhook(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort: advance an Rx from a Photon Order/Prescription webhook event.
    We match on the Photon order/prescription id we stored at completion time.
    Exact event schema confirmed in Phase 0 (docs: order-events)."""
    data = event.get("data") or event
    order_id = data.get("orderId") or data.get("id")
    status = (data.get("status") or event.get("type") or "").lower()
    if not order_id:
        return None
    for rec in _RX.values():
        if rec.get("photon_order_id") == order_id:
            if "fill" in status or "dispens" in status:
                rec["status"] = "filled"
            elif "error" in status or "cancel" in status:
                rec["status"] = "error"
            return rec
    return None


def get(rx_id: str) -> Optional[Dict[str, Any]]:
    return _RX.get(rx_id)


def list_for(patient_id: str) -> List[Dict[str, Any]]:
    """Prescriptions for a patient, newest first."""
    ids = _BY_PATIENT.get(patient_id, [])
    return [_RX[i] for i in reversed(ids) if i in _RX]
