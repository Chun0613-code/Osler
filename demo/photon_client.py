"""
photon_client.py — Photon Health (e-prescribing) integration for the Rx demo.

Photon is the prescription infrastructure behind Doximity Prescribe: it carries a
prescription onto the Surescripts network and on to the patient's chosen pharmacy.
We talk to its SANDBOX (Neutron) so the whole flow is exercisable end-to-end
without sending a real prescription anywhere.

Design rules that matter here:
  • The client_id / client_secret live ONLY in the backend env — never on a device.
  • Everything is gated behind is_enabled(): with no credentials the app still runs
    and prescription.py falls back to a MOCK flow (mirrors how this demo runs without
    an LLM key). A teammate can demo the end-to-end UX offline, then drop in real
    sandbox keys to light up the certified rails — no code change.
  • M2M OAuth2 client_credentials → Bearer token (~24h), cached and reused.
  • stdlib only (urllib), so `pip install flask` is still all you need to run.

Switch sandbox⇄prod with PHOTON_ENV. Docs:
  https://docs.photon.health/docs/authentication
  https://docs.photon.health/docs/integration-guide
"""
from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

# Sandbox (Neutron) vs production (Photon). v1 ships sandbox-only.
_ENV = (os.environ.get("PHOTON_ENV") or "sandbox").strip().lower()
IS_PROD = _ENV in ("prod", "production", "photon")

AUTH_URL = "https://auth.photon.health/oauth/token" if IS_PROD else "https://auth.neutron.health/oauth/token"
AUDIENCE = "https://api.photon.health" if IS_PROD else "https://api.neutron.health"
GRAPHQL_URL = "https://api.photon.health/graphql" if IS_PROD else "https://api.neutron.health/graphql"
# Provider-facing app the WebView can deep-link into (review/sign/pharmacy-select).
APP_ORIGIN = "https://app.photon.health" if IS_PROD else "https://app.neutron.health"

CLIENT_ID = (os.environ.get("PHOTON_CLIENT_ID") or "").strip()
CLIENT_SECRET = (os.environ.get("PHOTON_CLIENT_SECRET") or "").strip()
ORG_ID = (os.environ.get("PHOTON_ORG_ID") or "").strip()
# A single sandbox test prescriber the demo prescribes under (optional — with Elements
# the signed-in provider is the prescriber).
PROVIDER_ID = (os.environ.get("PHOTON_PROVIDER_ID") or "").strip()

# Single Page Application client id for Photon Elements auth inside the WebView.
# DISTINCT from the M2M client id above: Elements sign the *provider* in via this SPA app
# (its `id` attribute); the M2M pair is backend-only. Both the SPA id and org id are public.
SPA_CLIENT_ID = (os.environ.get("PHOTON_SPA_CLIENT_ID") or "").strip()


def is_enabled() -> bool:
    """True only when sandbox/prod credentials are configured. Otherwise the caller
    (prescription.py) uses its MOCK path so the demo still runs offline."""
    return bool(CLIENT_ID and CLIENT_SECRET)


def env_label() -> str:
    return "production" if IS_PROD else "sandbox"


# ── auth ───────────────────────────────────────────────────────────────────
_token: Dict[str, Any] = {"value": None, "exp": 0.0}


def _post_json(url: str, payload: dict, headers: dict, timeout: float = 20.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_token() -> Optional[str]:
    """Cached M2M bearer token. Photon tokens last ~86400s; refresh 5 min early."""
    if not is_enabled():
        return None
    now = time.time()
    if _token["value"] and now < _token["exp"]:
        return _token["value"]
    try:
        resp = _post_json(AUTH_URL, {
            "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
            "audience": AUDIENCE, "grant_type": "client_credentials",
        }, {"content-type": "application/json"})
    except urllib.error.URLError as e:  # offline / bad creds
        raise RuntimeError(f"Photon auth request failed: {e}") from e
    tok = resp.get("access_token")
    if not tok:
        raise RuntimeError(f"Photon auth returned no token: {resp}")
    _token["value"] = tok
    _token["exp"] = now + float(resp.get("expires_in", 86400)) - 300
    return tok


def graphql(query: str, variables: dict, token: Optional[str] = None) -> dict:
    """POST a GraphQL operation with a bearer token. Raises on transport or
    GraphQL-level errors.

    `token` lets a caller supply a PROVIDER (user) access token — required for
    createPrescription, whose prescriber is derived from the authenticated user
    (the backend M2M token deliberately lacks `write:prescription`). When omitted,
    the cached M2M token is used (patient sync, catalog search)."""
    token = token or get_token()
    if not token:
        raise RuntimeError("Photon is not configured (no client credentials).")
    resp = _post_json(GRAPHQL_URL, {"query": query, "variables": variables},
                      {"content-type": "application/json",
                       "authorization": f"Bearer {token}"})
    if resp.get("errors"):
        raise RuntimeError(f"Photon GraphQL error: {resp['errors']}")
    return resp.get("data") or {}


# ── patient sync ───────────────────────────────────────────────────────────
# Documented integration step #1 is "Sync Patients". We create a sandbox patient
# from the (anonymous) demo case so the prescribe UI is pre-bound to a real Photon
# patientId.
#
# ⚠️ CONFIRM IN PHASE 0: the exact mutation name + argument shape against the live
# sandbox schema (GraphQL introspection / docs.photon.health/reference/clinical-api).
# Photon's schema is AppSync-style; the shape below matches its documented model and
# is centralized so a single edit realigns it. This path is only reached when
# is_enabled() is True — the MOCK demo never calls it.
_CREATE_PATIENT = """
mutation oslerCreatePatient($externalId: ID, $name: NameInput!, $dateOfBirth: AWSDate!,
                            $sex: SexType!, $phone: AWSPhone!) {
  createPatient(externalId: $externalId, name: $name, dateOfBirth: $dateOfBirth,
                sex: $sex, phone: $phone) {
    id
  }
}
"""

# Photon SexType enum. Demo cases carry "M"/"F"/free-text → normalize.
_SEX = {"m": "MALE", "male": "MALE", "f": "FEMALE", "female": "FEMALE"}


def normalize_sex(sex: Optional[str]) -> str:
    return _SEX.get((sex or "").strip().lower(), "UNKNOWN")


def create_or_sync_patient(*, external_id: str, given: str, family: str,
                           dob: str, sex: str, phone: str) -> str:
    """Create a Photon sandbox patient; returns the Photon patient id.
    `dob` is YYYY-MM-DD, `phone` is E.164 (+15555550123)."""
    data = graphql(_CREATE_PATIENT, {
        "externalId": external_id,
        "name": {"first": given, "last": family},
        "dateOfBirth": dob,
        "sex": normalize_sex(sex),
        "phone": phone,
    })
    return data["createPatient"]["id"]


def prescribe_url(patient_id: str) -> str:
    """Hosted prescribe page (fallback / non-native path) — opened in the system browser.
    Sandbox → app.neutron.health. The native API path below is preferred."""
    return f"{APP_ORIGIN}/prescriptions/new?patientId={patient_id}"


# ── medication catalog search ──────────────────────────────────────────────
# medications(filter:{drug:{name}}) returns prescribable treatments; Medication.id IS the
# treatmentId for createPrescription. We omit `controlled` from the bulk query — the
# sandbox catalog has null `controlled` values that violate its own non-null schema and
# would fail the whole query.
_MEDS_SEARCH = """
query oslerSearchMeds($name: String!, $first: Int!) {
  medications(filter: {drug: {name: $name}}, first: $first) { id name strength form }
}
"""


def search_medications(name: str, first: int = 8) -> list:
    """Search the Photon catalog by drug name → [{id,name,strength,form}]. id = treatmentId."""
    return graphql(_MEDS_SEARCH, {"name": name, "first": first}).get("medications") or []


# ── API-direct prescribe (native in-app sign, no hosted UI) ─────────────────
# createPrescription has NO prescriberId arg — the prescriber is derived from the
# authenticated context. The clinician's in-app "Sign & Send" is the authorizing action.
# Valid for NON-controlled drugs; controlled substances legally require EPCS two-factor.
# Photon deprecated `refillsAllowed` — a prescription now declares total `fillsAllowed`
# (the initial fill + the refills), so fillsAllowed = refills + 1.
_CREATE_RX = """
mutation oslerCreateRx($patientId: ID!, $treatmentId: ID!, $dispenseQuantity: Float!,
                       $dispenseUnit: String!, $daysSupply: Int!, $fillsAllowed: Int!,
                       $instructions: String!) {
  createPrescription(patientId: $patientId, treatmentId: $treatmentId,
                     dispenseQuantity: $dispenseQuantity, dispenseUnit: $dispenseUnit,
                     daysSupply: $daysSupply,
                     fillsAllowed: $fillsAllowed, instructions: $instructions) { id }
}
"""

# pharmacyId omitted → Photon routes to the optimal pharmacy; an address lets it route.
_CREATE_ORDER = """
mutation oslerCreateOrder($patientId: ID!, $prescriptionId: ID!, $address: AddressInput) {
  createOrder(patientId: $patientId, fills: [{prescriptionId: $prescriptionId}],
              address: $address) { id }
}
"""

# Sandbox routing address (demo). Production uses the patient's real address.
_DEMO_ADDRESS = {"street1": "123 Main St", "city": "Washington", "state": "DC",
                 "postalCode": "20001", "country": "US"}


def create_prescription(*, patient_id: str, treatment_id: str, sig: str,
                        dispense_quantity: float, dispense_unit: str,
                        days_supply: int, refills: int,
                        token: Optional[str] = None) -> str:
    """Create a prescription (the clinician's in-app sign authorizes this). Returns rx id.
    `token` MUST be a provider access token carrying `write:prescription`; the M2M
    token lacks that scope and Photon rejects it with MISSING_PERMISSIONS."""
    data = graphql(_CREATE_RX, {
        "patientId": patient_id, "treatmentId": treatment_id,
        "dispenseQuantity": float(dispense_quantity), "dispenseUnit": dispense_unit,
        "daysSupply": int(days_supply), "fillsAllowed": int(refills) + 1,
        "instructions": sig,
    }, token=token)
    return data["createPrescription"]["id"]


def create_order(*, patient_id: str, prescription_id: str,
                 address: Optional[dict] = None,
                 token: Optional[str] = None) -> str:
    """Place the order (pharmacyId omitted → Photon routes to the optimal pharmacy).
    Returns order id. Pass the provider token so the order is attributed to the same
    authenticated clinician who signed the prescription."""
    data = graphql(_CREATE_ORDER, {
        "patientId": patient_id, "prescriptionId": prescription_id,
        "address": address or _DEMO_ADDRESS,
    }, token=token)
    return data["createOrder"]["id"]
