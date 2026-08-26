import json
from pathlib import Path
from urllib.request import Request

import pytest

from corroborly.engine.zotero_api import (
    ZoteroApiCredentials,
    ZoteroApiError,
    clear_zotero_api_credentials,
    create_zotero_collection,
    create_zotero_items,
    load_dotenv_values,
    mirror_items_to_zotero_collection,
    require_zotero_write_access,
    save_zotero_api_credentials,
    write_dotenv_values,
    zotero_api_collections,
    zotero_api_credentials,
    zotero_api_readiness,
)

CREDS = ZoteroApiCredentials(api_key="secret", user_id="42")


class RouterOpener:
    """Dispatches a fake response by (method, url) predicate, for multi-call tests."""

    def __init__(self, routes):
        self.routes = routes  # list of (predicate, data)
        self.calls = []

    def __call__(self, request: Request):
        self.calls.append((request.get_method(), request.full_url))
        for predicate, data in self.routes:
            if predicate(request):
                return FakeResponse(data)
        raise AssertionError(f"no route for {request.get_method()} {request.full_url}")


class FakeResponse:
    def __init__(self, data: object):
        self.data = json.dumps(data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.data


def test_load_dotenv_values_reads_zotero_credentials(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("ZOTERO_API_KEY=abc\nZOTERO_USER_ID=123\n", encoding="utf-8")

    values = load_dotenv_values(env_path)

    assert values["ZOTERO_API_KEY"] == "abc"
    assert values["ZOTERO_USER_ID"] == "123"


def test_write_dotenv_values_preserves_other_lines(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("# comment\nOPENAI_API_KEY=keep-me\nZOTERO_API_KEY=old\n", encoding="utf-8")

    write_dotenv_values(env_path, {"ZOTERO_API_KEY": "new-key", "ZOTERO_USER_ID": "42"})

    values = load_dotenv_values(env_path)
    assert values["OPENAI_API_KEY"] == "keep-me"
    assert values["ZOTERO_API_KEY"] == "new-key"
    assert values["ZOTERO_USER_ID"] == "42"
    assert "# comment" in env_path.read_text(encoding="utf-8")


def test_write_dotenv_values_creates_missing_file(tmp_path: Path) -> None:
    env_path = tmp_path / "nested" / ".env"

    write_dotenv_values(env_path, {"ZOTERO_API_KEY": "abc"})

    assert load_dotenv_values(env_path)["ZOTERO_API_KEY"] == "abc"


def test_save_and_clear_zotero_api_credentials_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # zotero_api_credentials() also merges in Path.cwd()/.env; isolate from
    # the real repo-root .env (a developer's actual Zotero credentials) so
    # this test's "cleared -> raises" assertion can't pass or fail based on
    # whatever happens to be configured on the machine running it.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.delenv("ZOTERO_USER_ID", raising=False)

    save_zotero_api_credentials(tmp_path, "  my-key  ", " 999 ")

    credentials = zotero_api_credentials(tmp_path)
    assert credentials.api_key == "my-key"
    assert credentials.user_id == "999"

    clear_zotero_api_credentials(tmp_path)
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "ZOTERO_API_KEY" not in env_text
    assert "ZOTERO_USER_ID" not in env_text
    with pytest.raises(ZoteroApiError):
        zotero_api_credentials(tmp_path)


def test_save_zotero_api_credentials_rejects_blank_values(tmp_path: Path) -> None:
    with pytest.raises(ZoteroApiError):
        save_zotero_api_credentials(tmp_path, "", "999")
    with pytest.raises(ZoteroApiError):
        save_zotero_api_credentials(tmp_path, "key", "  ")


def test_zotero_api_readiness_reports_write_access_without_exposing_key() -> None:
    def opener(request: Request):
        assert request.get_method() == "GET"
        assert request.headers["Zotero-api-key"] == "secret"
        return FakeResponse({"access": {"user": {"library": True, "notes": True, "write": False}}})

    report = zotero_api_readiness(ZoteroApiCredentials(api_key="secret", user_id="42"), opener=opener)

    assert report["key_loaded"] is True
    assert report["key_has_write_access"] is False
    assert "secret" not in str(report)


def test_zotero_api_collections_maps_response() -> None:
    def opener(_request: Request):
        return FakeResponse(
            [
                {"key": "ABC", "version": 1, "data": {"name": "Thesis", "parentCollection": False}},
                {"key": "DEF", "version": 2, "data": {"name": "Chapter", "parentCollection": "ABC"}},
            ]
        )

    rows = zotero_api_collections(ZoteroApiCredentials(api_key="secret", user_id="42"), opener=opener)

    assert rows == [
        {"key": "ABC", "name": "Thesis", "parent_key": False, "version": 1, "source": "zotero_web_api"},
        {"key": "DEF", "name": "Chapter", "parent_key": "ABC", "version": 2, "source": "zotero_web_api"},
    ]


def test_zotero_api_requires_json_response() -> None:
    class BadResponse(FakeResponse):
        def __init__(self):
            self.data = b"not-json"

    with pytest.raises(ZoteroApiError, match="invalid JSON"):
        zotero_api_readiness(ZoteroApiCredentials(api_key="secret", user_id="42"), opener=lambda _request: BadResponse())


def test_create_zotero_collection_returns_saved_key() -> None:
    def opener(request: Request):
        assert request.get_method() == "POST"
        assert request.full_url.endswith("/users/42/collections")
        assert request.headers.get("Zotero-write-token")  # idempotency token set
        body = json.loads(request.data.decode("utf-8"))
        assert body == [{"name": "AES", "parentCollection": False}]
        return FakeResponse(
            {"successful": {"0": {"key": "NEWKEY", "data": {"key": "NEWKEY", "name": "AES"}}}, "failed": {}}
        )

    assert create_zotero_collection(CREDS, "AES", opener=opener) == {"key": "NEWKEY", "name": "AES"}


def test_create_zotero_collection_raises_on_failed() -> None:
    def opener(_request: Request):
        return FakeResponse({"successful": {}, "failed": {"0": {"code": 400, "message": "bad"}}})

    with pytest.raises(ZoteroApiError, match="rejected the collection"):
        create_zotero_collection(CREDS, "AES", opener=opener)


def test_require_write_access_raises_for_read_only_key() -> None:
    def opener(_request: Request):
        return FakeResponse({"access": {"user": {"library": True, "write": False}}})

    with pytest.raises(ZoteroApiError, match="does not have write access"):
        require_zotero_write_access(CREDS, opener=opener)


def test_create_zotero_items_maps_success() -> None:
    def opener(request: Request):
        assert request.get_method() == "POST"
        assert request.full_url.endswith("/users/42/items")
        return FakeResponse({"success": {"0": "ITEMKEY"}, "unchanged": {}, "failed": {}})

    out = create_zotero_items(CREDS, [{"itemType": "preprint", "title": "X"}], opener=opener)
    assert out["created"] == {"0": "ITEMKEY"}
    assert out["failed"] == {}


def test_mirror_creates_new_and_skips_existing_and_assigns_collection() -> None:
    posted_body = {}

    def is_keys(r: Request) -> bool:
        return r.get_method() == "GET" and "/keys/" in r.full_url

    def is_collections(r: Request) -> bool:
        return r.get_method() == "GET" and r.full_url.endswith("/users/42/collections?limit=100")

    def is_coll_items(r: Request) -> bool:
        return r.get_method() == "GET" and "/collections/COL1/items" in r.full_url

    def is_post_items(r: Request) -> bool:
        posted = r.get_method() == "POST" and r.full_url.endswith("/users/42/items")
        if posted:
            posted_body["items"] = json.loads(r.data.decode("utf-8"))
        return posted

    router = RouterOpener(
        [
            (is_keys, {"access": {"user": {"library": True, "write": True}}}),
            (is_collections, [{"key": "COL1", "version": 1, "data": {"name": "AES", "parentCollection": False}}]),
            (is_coll_items, [{"data": {"DOI": "10.1/existing", "title": "Existing"}}]),
            (is_post_items, {"success": {"0": "NEWITEM"}, "unchanged": {}, "failed": {}}),
        ]
    )

    items = [
        {"itemType": "journalArticle", "title": "Existing", "DOI": "10.1/existing"},  # deduped -> skipped
        {"itemType": "preprint", "title": "New", "url": "https://arxiv.org/abs/2607.10059"},  # created
    ]
    report = mirror_items_to_zotero_collection(CREDS, "AES", items, opener=router)

    assert report["collection"] == {"key": "COL1", "name": "AES", "created": False}
    assert report["skipped"] == ["Existing"]
    assert report["attempted"] == 1
    assert report["created"] == {"0": "NEWITEM"}
    # the created item was assigned to the target collection
    assert posted_body["items"][0]["collections"] == ["COL1"]
