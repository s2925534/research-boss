from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ZOTERO_API_BASE_URL = "https://api.zotero.org"

# Cap every Web API call so a stalled connection raises URLError instead of
# hanging indefinitely (applies only to the real urlopen; injected openers,
# e.g. in tests, are used as-is).
ZOTERO_API_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class ZoteroApiCredentials:
    api_key: str
    user_id: str


class ZoteroApiError(RuntimeError):
    pass


def load_dotenv_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def write_dotenv_values(path: Path, updates: dict[str, str]) -> None:
    """Upsert key=value pairs into a dotenv file, preserving every other line.

    Used to let the CLI/web UI save credentials a user submits interactively
    (e.g. linking a Zotero account) without hand-editing `.env`. Never called
    with, and must never be used to persist, anything that should instead
    live only in-memory or in session storage.
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    remaining = dict(updates)
    result = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                result.append(f"{key}={remaining.pop(key)}")
                continue
        result.append(raw_line)
    for key, value in remaining.items():
        result.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(result) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def remove_dotenv_values(path: Path, keys: list[str]) -> None:
    """Remove the given keys from a dotenv file, leaving every other line untouched."""
    if not path.is_file():
        return
    keys_set = set(keys)
    result = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in keys_set:
                continue
        result.append(raw_line)
    path.write_text("\n".join(result) + ("\n" if result else ""), encoding="utf-8")


def save_zotero_api_credentials(workspace: Path, api_key: str, user_id: str) -> None:
    """Persist Zotero Web API credentials into the workspace's local `.env` file.

    This is the CLI/web-UI "link your Zotero account" entry point, replacing
    hand-editing `.env`. Never returns, logs, or echoes the credential value
    back to any caller — callers only learn whether the save succeeded.
    """
    api_key = api_key.strip()
    user_id = user_id.strip()
    if not api_key:
        raise ZoteroApiError("API key is required.")
    if not user_id:
        raise ZoteroApiError("User ID is required.")
    write_dotenv_values(workspace / ".env", {"ZOTERO_API_KEY": api_key, "ZOTERO_USER_ID": user_id})


def clear_zotero_api_credentials(workspace: Path) -> None:
    """Remove any saved Zotero Web API credentials from the workspace's `.env` file."""
    remove_dotenv_values(workspace / ".env", ["ZOTERO_API_KEY", "ZOTERO_USER_ID"])


def zotero_api_credentials(workspace: Path | None = None) -> ZoteroApiCredentials:
    env_values = load_dotenv_values(Path.cwd() / ".env")
    if workspace is not None:
        env_values = {**env_values, **load_dotenv_values(workspace / ".env")}
    api_key = os.environ.get("ZOTERO_API_KEY") or env_values.get("ZOTERO_API_KEY") or ""
    user_id = os.environ.get("ZOTERO_USER_ID") or env_values.get("ZOTERO_USER_ID") or ""
    if not api_key:
        raise ZoteroApiError("Missing ZOTERO_API_KEY")
    if not user_id:
        raise ZoteroApiError("Missing ZOTERO_USER_ID")
    return ZoteroApiCredentials(api_key=api_key, user_id=user_id)


def zotero_api_get(
    path: str,
    credentials: ZoteroApiCredentials,
    *,
    opener: Callable[[Request], Any] | None = None,
    base_url: str = ZOTERO_API_BASE_URL,
) -> Any:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    request = Request(
        url,
        headers={
            "Zotero-API-Key": credentials.api_key,
            "Zotero-API-Version": "3",
            "Accept": "application/json",
        },
        method="GET",
    )
    fetch = opener or partial(urlopen, timeout=ZOTERO_API_TIMEOUT_SECONDS)
    try:
        with fetch(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ZoteroApiError(f"Zotero API request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise ZoteroApiError(f"Zotero API request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise ZoteroApiError("Zotero API returned invalid JSON") from exc


def zotero_api_key_info(
    credentials: ZoteroApiCredentials,
    *,
    opener: Callable[[Request], Any] | None = None,
) -> dict[str, Any]:
    data = zotero_api_get(f"keys/{credentials.api_key}", credentials, opener=opener)
    return data if isinstance(data, dict) else {}


def zotero_api_collections(
    credentials: ZoteroApiCredentials,
    *,
    opener: Callable[[Request], Any] | None = None,
) -> list[dict[str, Any]]:
    data = zotero_api_get(f"users/{credentials.user_id}/collections?limit=100", credentials, opener=opener)
    if not isinstance(data, list):
        return []
    collections = []
    for item in data:
        if not isinstance(item, dict):
            continue
        item_data = item.get("data") if isinstance(item.get("data"), dict) else {}
        collections.append(
            {
                "key": item.get("key") or item_data.get("key"),
                "name": item_data.get("name"),
                "parent_key": item_data.get("parentCollection"),
                "version": item.get("version"),
                "source": "zotero_web_api",
            }
        )
    return collections


def zotero_api_readiness(
    credentials: ZoteroApiCredentials,
    *,
    opener: Callable[[Request], Any] | None = None,
) -> dict[str, Any]:
    info = zotero_api_key_info(credentials, opener=opener)
    access = info.get("access") if isinstance(info.get("access"), dict) else {}
    user_access = access.get("user") if isinstance(access.get("user"), dict) else {}
    return {
        "version": 1,
        "user_id": credentials.user_id,
        "key_loaded": True,
        "key_has_write_access": bool(user_access.get("write")),
        "library_access": bool(user_access.get("library")),
        "notes_access": bool(user_access.get("notes")),
        "policy": "read_only_local_write_via_web_api_opt_in",
    }


# --------------------------------------------------------------------------
# Zotero Web API write path (opt-in).
#
# These functions create collections and items through https://api.zotero.org
# only. They never touch the user's local Zotero directory (zotero.sqlite,
# storage/), so the AGENTS.md hard rule "never modify anything inside the
# user's local Zotero directory" is preserved: server-side library mutation
# via the authenticated Web API is a distinct, explicitly opt-in capability.
# A write-scoped API key is required; require_zotero_write_access() refuses to
# proceed with a read-only key.
# --------------------------------------------------------------------------


def zotero_api_post(
    path: str,
    credentials: ZoteroApiCredentials,
    body: Any,
    *,
    opener: Callable[[Request], Any] | None = None,
    base_url: str = ZOTERO_API_BASE_URL,
    write_token: str | None = None,
    if_unmodified_since_version: int | None = None,
) -> Any:
    """POST a JSON body to the Zotero Web API and return the decoded response.

    Idempotency: pass write_token (a client-generated token Zotero caches for
    12 hours) for new-object creation, or if_unmodified_since_version for
    versioned updates. Never send both.
    """
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {
        "Zotero-API-Key": credentials.api_key,
        "Zotero-API-Version": "3",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if write_token is not None:
        headers["Zotero-Write-Token"] = write_token
    if if_unmodified_since_version is not None:
        headers["If-Unmodified-Since-Version"] = str(if_unmodified_since_version)
    data = json.dumps(body).encode("utf-8")
    request = Request(url, data=data, headers=headers, method="POST")
    fetch = opener or partial(urlopen, timeout=ZOTERO_API_TIMEOUT_SECONDS)
    try:
        with fetch(request) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:500]
        except Exception:
            detail = ""
        suffix = f": {detail}" if detail else ""
        raise ZoteroApiError(f"Zotero API write failed with HTTP {exc.code}{suffix}") from exc
    except URLError as exc:
        raise ZoteroApiError(f"Zotero API write failed: {exc.reason}") from exc
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ZoteroApiError("Zotero API returned invalid JSON") from exc


def require_zotero_write_access(
    credentials: ZoteroApiCredentials,
    *,
    opener: Callable[[Request], Any] | None = None,
) -> dict[str, Any]:
    """Confirm the linked key can write, or raise with remediation guidance."""
    report = zotero_api_readiness(credentials, opener=opener)
    if not report.get("key_has_write_access"):
        raise ZoteroApiError(
            "The linked Zotero API key does not have write access. Create a key "
            "with 'Allow write access' at https://www.zotero.org/settings/keys "
            "and re-link it with `corroborly zotero api-link`."
        )
    return report


def create_zotero_collection(
    credentials: ZoteroApiCredentials,
    name: str,
    *,
    parent_key: str | None = None,
    opener: Callable[[Request], Any] | None = None,
    write_token: str | None = None,
) -> dict[str, Any]:
    """Create one collection in the user's Web API library. Returns {key, name}."""
    name = name.strip()
    if not name:
        raise ZoteroApiError("Collection name is required.")
    token = write_token or uuid.uuid4().hex
    body = [{"name": name, "parentCollection": parent_key if parent_key else False}]
    result = zotero_api_post(
        f"users/{credentials.user_id}/collections", credentials, body,
        opener=opener, write_token=token,
    )
    result = result if isinstance(result, dict) else {}
    failed = result.get("failed") or {}
    if failed:
        raise ZoteroApiError(f"Zotero rejected the collection: {json.dumps(failed)[:300]}")
    saved = (result.get("successful") or {}).get("0")
    if not isinstance(saved, dict):
        raise ZoteroApiError("Zotero did not return the created collection.")
    saved_data = saved.get("data") if isinstance(saved.get("data"), dict) else {}
    return {"key": saved.get("key") or saved_data.get("key"), "name": saved_data.get("name") or name}


def find_or_create_zotero_collection(
    credentials: ZoteroApiCredentials,
    name: str,
    *,
    opener: Callable[[Request], Any] | None = None,
) -> dict[str, Any]:
    """Return an existing top-level collection of this name, or create it.

    Makes the mirror re-runnable: a second run reuses the same collection
    rather than creating a duplicate.
    """
    target = name.strip().lower()
    for col in zotero_api_collections(credentials, opener=opener):
        if (col.get("name") or "").strip().lower() == target and not col.get("parent_key"):
            return {"key": col.get("key"), "name": col.get("name"), "created": False}
    created = create_zotero_collection(credentials, name, opener=opener)
    return {"key": created["key"], "name": created["name"], "created": True}


def zotero_api_collection_items(
    credentials: ZoteroApiCredentials,
    collection_key: str,
    *,
    opener: Callable[[Request], Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the `data` dicts of items already in a collection (for dedupe)."""
    data = zotero_api_get(
        f"users/{credentials.user_id}/collections/{collection_key}/items?limit=100",
        credentials, opener=opener,
    )
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("data"), dict):
            out.append(item["data"])
    return out


def _item_dedupe_keys(item_data: dict[str, Any]) -> set[str]:
    keys = set()
    doi = (item_data.get("DOI") or "").strip().lower()
    if doi:
        keys.add("doi:" + doi)
    url = (item_data.get("url") or "").strip().lower()
    if url:
        keys.add("url:" + url)
    title = (item_data.get("title") or "").strip().lower()
    if title:
        keys.add("title:" + title)
    return keys


def create_zotero_items(
    credentials: ZoteroApiCredentials,
    items: list[dict[str, Any]],
    *,
    opener: Callable[[Request], Any] | None = None,
    write_token: str | None = None,
) -> dict[str, Any]:
    """Create items in the user's Web API library. Returns created/unchanged/failed maps."""
    if not items:
        return {"created": {}, "unchanged": {}, "failed": {}}
    token = write_token or uuid.uuid4().hex
    result = zotero_api_post(
        f"users/{credentials.user_id}/items", credentials, list(items),
        opener=opener, write_token=token,
    )
    result = result if isinstance(result, dict) else {}
    return {
        "created": result.get("success") or {},
        "unchanged": result.get("unchanged") or {},
        "failed": result.get("failed") or {},
    }


def mirror_items_to_zotero_collection(
    credentials: ZoteroApiCredentials,
    collection_name: str,
    items: list[dict[str, Any]],
    *,
    opener: Callable[[Request], Any] | None = None,
) -> dict[str, Any]:
    """High-level mirror: ensure a single named collection exists and add items
    to it, skipping items already present (by DOI, url or title). Idempotent
    and re-runnable. Requires a write-scoped key. Never touches local Zotero.
    """
    require_zotero_write_access(credentials, opener=opener)
    col = find_or_create_zotero_collection(credentials, collection_name, opener=opener)
    existing: set[str] = set()
    for data in zotero_api_collection_items(credentials, col["key"], opener=opener):
        existing |= _item_dedupe_keys(data)
    to_create: list[dict[str, Any]] = []
    skipped: list[str | None] = []
    for raw_item in items:
        item = dict(raw_item)
        cols = list(item.get("collections") or [])
        if col["key"] not in cols:
            cols.append(col["key"])
        item["collections"] = cols
        if _item_dedupe_keys(item) & existing:
            skipped.append(item.get("title"))
            continue
        to_create.append(item)
    result = create_zotero_items(credentials, to_create, opener=opener)
    return {
        "collection": col,
        "attempted": len(to_create),
        "created": result["created"],
        "skipped": skipped,
        "failed": result["failed"],
    }
