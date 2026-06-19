"""
photon_oauth.py — provider (clinician) login so the app can SIGN prescriptions.

Why this exists: the backend M2M token can sync patients and search the catalog, but
Photon deliberately withholds `write:prescription` from machine tokens — a prescription
must be bound to an authenticated, authorized provider. So a clinician logs in ONCE via
their Neutron/Photon account (Authorization Code + PKCE, in the system browser — embedded
WebView login is blocked by Auth0/Google), and we exchange that for a *provider* access
token carrying `write:prescription`. createPrescription/createOrder then run natively
under that token.

Design choices (matches the rest of the demo):
  • The whole OAuth dance runs BACKEND-side. The device never holds the provider token;
    it only learns connected:true/false. Keeps every secret server-side and gives a
    STABLE redirect_uri (http://127.0.0.1:5000/...) that works the same in Expo Go and a
    dev build — far easier to whitelist than per-machine exp:// URLs.
  • PKCE with stdlib only (secrets + hashlib + base64) — no new dependency.
  • Provider token is process-local (like demo_app._CACHE / prescription._RX): a demo
    session reconnects fresh. Production swaps this for per-user encrypted storage.

Auth0 tenant (confirmed via the OIDC discovery doc):
  authorize : https://auth.neutron.health/authorize     (S256 PKCE supported)
  token     : https://auth.neutron.health/oauth/token   (authorization_code, refresh_token)
  API scopes (read:patient … write:prescription) live on the audience https://api.neutron.health
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

import photon_client as PH

# Endpoints derived from photon_client's token URL so sandbox⇄prod stays in one place.
_AUTH_BASE = PH.AUTH_URL.rsplit("/oauth/token", 1)[0]   # https://auth.neutron.health
AUTHORIZE_URL = f"{_AUTH_BASE}/authorize"
TOKEN_URL = PH.AUTH_URL
AUDIENCE = PH.AUDIENCE

# Where Neutron redirects after login. Photon's SPA "Whitelisted URLs" are bare ORIGINS
# (no path/query) — `http://127.0.0.1:5000` is already whitelisted — so we redirect to the
# origin and catch the ?code&state at the Flask ROOT route (see demo_app.index). This needs
# zero dashboard changes. MUST be byte-identical in the authorize request and the token
# exchange. Configurable for LAN/real-device runs (add that origin to the whitelist too).
REDIRECT_URI = (os.environ.get("PHOTON_OAUTH_REDIRECT")
                or "http://127.0.0.1:5000").strip()

# offline_access → refresh_token; the write:* scopes are granted only to authorized
# prescribers (Auth0 hands back the subset the provider's role actually permits).
SCOPES = ("openid profile email offline_access "
          "read:patient write:patient read:prescription "
          "read:order write:order write:prescription")

# Pending authorizations: state → {verifier, return_url, ts}. Short-lived; pruned on use.
_PENDING: Dict[str, Dict[str, Any]] = {}
# The signed-in provider's token. exp is an absolute epoch (refreshed 60s early).
_PROVIDER: Dict[str, Any] = {"access": None, "refresh": None, "exp": 0.0, "claims": {}}


def is_enabled() -> bool:
    """OAuth is possible only when the SPA client id is configured."""
    return bool(PH.SPA_CLIENT_ID)


# ── PKCE ─────────────────────────────────────────────────────────────────────
def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _pkce_pair() -> tuple[str, str]:
    """(code_verifier, code_challenge) for S256. Verifier is 43 chars of URL-safe
    entropy; challenge = base64url(sha256(verifier))."""
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


# ── authorize ────────────────────────────────────────────────────────────────
def build_authorize_url(return_url: str) -> str:
    """Start a login: mint PKCE + state, remember them, return the Neutron authorize URL
    the app opens in the system browser. `return_url` is the app deep link we bounce back
    to after the callback (e.g. oslianrx://photon-connected)."""
    if not is_enabled():
        raise RuntimeError("Photon SPA client id is not configured (PHOTON_SPA_CLIENT_ID).")
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(16)
    _PENDING[state] = {"verifier": verifier, "return_url": return_url, "ts": time.time()}
    _prune_pending()
    params = {
        "response_type": "code",
        "client_id": PH.SPA_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "audience": AUDIENCE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if PH.ORG_ID:
        # Auth0 Organizations: pre-selects the org so the provider isn't asked to pick.
        params["organization"] = PH.ORG_ID
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def consume_state(state: str) -> Optional[Dict[str, Any]]:
    """Pop a pending authorization by its state (one-time use). None if unknown/expired."""
    return _PENDING.pop(state, None)


def _prune_pending(max_age: float = 600.0) -> None:
    now = time.time()
    for s in [s for s, p in _PENDING.items() if now - p.get("ts", 0) > max_age]:
        _PENDING.pop(s, None)


# ── token exchange / refresh ──────────────────────────────────────────────────
def _token_request(payload: dict) -> dict:
    """POST the Auth0 token endpoint, surfacing the JSON error body on 4xx (Auth0 returns
    {error, error_description}) so callbacks can show a real reason instead of 'HTTP 403'."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(TOKEN_URL, data=data,
                                 headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
            msg = body.get("error_description") or body.get("error") or str(body)
        except Exception:  # noqa: BLE001
            msg = f"HTTP {e.code}"
        raise RuntimeError(f"Photon token exchange failed: {msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Photon token request failed: {e}") from e


def _store(resp: dict) -> None:
    """Persist the token response into _PROVIDER (refresh token rotates if returned)."""
    _PROVIDER["access"] = resp.get("access_token")
    if resp.get("refresh_token"):
        _PROVIDER["refresh"] = resp["refresh_token"]
    _PROVIDER["exp"] = time.time() + float(resp.get("expires_in", 86400)) - 60
    if resp.get("id_token"):
        _PROVIDER["claims"] = _decode_jwt_claims(resp["id_token"])


def exchange_code(code: str, verifier: str) -> None:
    """Authorization code → provider token. Raises RuntimeError on failure."""
    resp = _token_request({
        "grant_type": "authorization_code",
        "client_id": PH.SPA_CLIENT_ID,
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
    })
    if not resp.get("access_token"):
        raise RuntimeError(f"Token exchange returned no access_token: {resp}")
    _store(resp)


def _refresh() -> None:
    resp = _token_request({
        "grant_type": "refresh_token",
        "client_id": PH.SPA_CLIENT_ID,
        "refresh_token": _PROVIDER["refresh"],
    })
    if not resp.get("access_token"):
        raise RuntimeError("Refresh returned no access_token")
    _store(resp)


def provider_token() -> Optional[str]:
    """A valid provider access token, or None if not connected. Refreshes transparently
    when expired and a refresh token is on hand."""
    if _PROVIDER["access"] and time.time() < _PROVIDER["exp"]:
        return _PROVIDER["access"]
    if _PROVIDER["refresh"]:
        try:
            _refresh()
            return _PROVIDER["access"]
        except Exception as e:  # noqa: BLE001 — expired/revoked refresh ⇒ treat as disconnected
            print(f"[photon-oauth] refresh failed, disconnecting: {e}")
            disconnect()
    return None


def disconnect() -> None:
    _PROVIDER.update({"access": None, "refresh": None, "exp": 0.0, "claims": {}})


# ── status / claims ───────────────────────────────────────────────────────────
def _decode_jwt_claims(token: str) -> Dict[str, Any]:
    """Decode a JWT payload WITHOUT verifying (display only — name/email for the UI).
    The token's authority comes from Photon checking it server-side, not from us."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # restore base64 padding
        return json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def granted_scopes() -> str:
    """The scope string actually granted in the provider access token (Auth0 grants only
    the subset the provider's role permits). Empty if not connected. Used to confirm
    `write:prescription` was granted before the clinician tries to sign."""
    tok = _PROVIDER.get("access")
    if not tok:
        return ""
    claims = _decode_jwt_claims(tok)
    scope = claims.get("scope") or ""
    if not scope and isinstance(claims.get("permissions"), list):
        scope = " ".join(claims["permissions"])
    return scope


def status() -> Dict[str, Any]:
    """Connection state for the Settings UI."""
    connected = provider_token() is not None
    claims = _PROVIDER["claims"] if connected else {}
    provider = None
    scopes = ""
    if connected:
        provider = {
            "name": claims.get("name") or claims.get("nickname"),
            "email": claims.get("email"),
        }
        scopes = granted_scopes()
    return {
        "connected": connected,
        "provider": provider,
        "env": PH.env_label(),
        "enabled": is_enabled(),
        "can_prescribe": "write:prescription" in scopes,
        "scopes": scopes,
    }
