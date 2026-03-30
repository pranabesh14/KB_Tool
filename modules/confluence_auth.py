“””
confluence_auth.py - Azure AD OAuth 2.0 token management.

Shared by both Confluence and SharePoint since they use the same
Azure AD App Registration.

Authentication flow:

1. Try silent token acquisition from MSAL cache
1. If client secret present: client credentials flow (non-interactive)
1. Otherwise: interactive browser login (triggers Microsoft Authenticator MFA)
1. Fallback: device code flow (headless environments)
1. Token cached to disk - survives app restarts, auto-refreshed when expired

## Public API

get_headers(service)   -> dict    # “confluence” or “sharepoint”
clear_token_cache()               # force re-login on next call
“””

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

from typing import Dict
from modules.config import (
AZURE_AD_TENANT_ID, AZURE_AD_CLIENT_ID, AZURE_AD_CLIENT_SECRET,
AZURE_AD_CONFLUENCE_SCOPE, AZURE_AD_SHAREPOINT_SCOPE,
DATA_DIR, get_logger,
)

logger = get_logger(“confluence_auth”)

_TOKEN_CACHE_FILE = os.path.join(DATA_DIR, “.azure_token_cache.json”)
_msal_app = None

def _get_msal_app():
global _msal_app
if _msal_app is not None:
return _msal_app

try:
    import msal
except ImportError as exc:
    raise RuntimeError("msal not installed. Run: pip install msal") from exc

cache = msal.SerializableTokenCache()
if os.path.exists(_TOKEN_CACHE_FILE):
    try:
        with open(_TOKEN_CACHE_FILE, "r", encoding="utf-8") as f:
            cache.deserialize(f.read())
    except Exception as exc:
        logger.warning("Could not load token cache: %s", exc)

if AZURE_AD_CLIENT_SECRET:
    _msal_app = msal.ConfidentialClientApplication(
        client_id=AZURE_AD_CLIENT_ID,
        client_credential=AZURE_AD_CLIENT_SECRET,
        authority=f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}",
        token_cache=cache,
    )
    logger.info("MSAL: ConfidentialClientApplication (client secret)")
else:
    _msal_app = msal.PublicClientApplication(
        client_id=AZURE_AD_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}",
        token_cache=cache,
    )
    logger.info("MSAL: PublicClientApplication (interactive)")
return _msal_app

def _save_cache() -> None:
app = _msal_app
if app is None:
return
try:
if hasattr(app, “token_cache”) and app.token_cache.has_state_changed:
with open(_TOKEN_CACHE_FILE, “w”, encoding=“utf-8”) as f:
f.write(app.token_cache.serialize())
except Exception as exc:
logger.warning(“Could not save token cache: %s”, exc)

def _acquire_token(scope: str) -> str:
“””
Acquire an access token for the given scope.

For shared server deployments: uses client credentials flow only.
AZURE_AD_CLIENT_SECRET must be set in .env.

For single-user/dev: falls back to interactive browser login if no secret set.
"""
import msal
app    = _get_msal_app()
scopes = [scope]

# 1. Silent from cache (works for both flows)
accounts = app.get_accounts()
if accounts:
    result = app.acquire_token_silent(scopes, account=accounts[0])
    if result and "access_token" in result:
        logger.info("Token from cache (silent).")
        _save_cache()
        return result["access_token"]

# 2. Client credentials (shared server - requires AZURE_AD_CLIENT_SECRET)
if AZURE_AD_CLIENT_SECRET and isinstance(app, msal.ConfidentialClientApplication):
    result = app.acquire_token_for_client(scopes=scopes)
    if result and "access_token" in result:
        logger.info("Token via client credentials.")
        _save_cache()
        return result["access_token"]
    raise RuntimeError(
        "Client credentials flow failed: "
        + result.get("error_description", result.get("error", "unknown"))
        + "\n\nCheck that:\n"
        + "  1. AZURE_AD_CLIENT_SECRET is correct in .env\n"
        + "  2. The App Registration has the correct API permissions\n"
        + "  3. Admin consent has been granted in Azure Portal"
    )

# 3. Interactive browser (single-user / dev only - will not work on shared server)
if AZURE_AD_CLIENT_SECRET:
    # Secret was set but app is not ConfidentialClientApplication - config error
    raise RuntimeError(
        "AZURE_AD_CLIENT_SECRET is set but MSAL app is not confidential. "
        "Check AZURE_AD_TENANT_ID and AZURE_AD_CLIENT_ID in .env."
    )

# No secret - try interactive (only works if someone is at the machine)
logger.warning(
    "AZURE_AD_CLIENT_SECRET not set. "
    "Attempting interactive login - this WILL NOT WORK on a shared server. "
    "Set AZURE_AD_CLIENT_SECRET in .env for shared deployments."
)
try:
    result = app.acquire_token_interactive(scopes=scopes, prompt="select_account")
    if result and "access_token" in result:
        uname = result.get("id_token_claims", {}).get("preferred_username", "?")
        logger.info("Interactive login OK for: %s", uname)
        _save_cache()
        return result["access_token"]
except Exception as exc:
    raise RuntimeError(
        f"Interactive login failed: {exc}\n\n"
        "For shared server deployments, set AZURE_AD_CLIENT_SECRET in .env "
        "and use client credentials flow instead."
    ) from exc

raise RuntimeError("Authentication failed - no access token obtained.")

def get_headers(service: str = “confluence”) -> Dict[str, str]:
“””
Return Bearer auth headers for the given service.

Parameters
----------
service : str
    "confluence" or "sharepoint"

Returns
-------
dict with Authorization, Content-Type, Accept headers.
Triggers interactive login + MFA if no valid token is cached.
"""
scope = (
    AZURE_AD_CONFLUENCE_SCOPE
    if service == "confluence"
    else AZURE_AD_SHAREPOINT_SCOPE
)
if not scope:
    raise RuntimeError(
        f"AZURE_AD_{service.upper()}_SCOPE is not set in .env\n"
        f"Example: api://<app-registration-client-id>/.default"
    )
token = _acquire_token(scope)
return {
    "Authorization": f"Bearer {token}",
    "Content-Type":  "application/json",
    "Accept":        "application/json",
}


def clear_token_cache() -> None:
“”“Clear cached tokens and MSAL app - forces re-authentication on next call.”””
global _msal_app
_msal_app = None
if os.path.exists(_TOKEN_CACHE_FILE):
try:
os.remove(_TOKEN_CACHE_FILE)
logger.info(“Azure AD token cache cleared.”)
except Exception as exc:
logger.warning(“Could not delete token cache file: %s”, exc)