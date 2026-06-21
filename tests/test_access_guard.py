import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.access_guard import AccessGuardMiddleware


@pytest.fixture
def protected_client(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN", "secret-token")
    app = FastAPI()
    app.add_middleware(AccessGuardMiddleware)

    @app.get("/")
    async def root():
        return {"ok": True}

    with patch("services.access_guard.ACCESS_TOKEN", "secret-token"):
        yield TestClient(app)


def test_open_when_access_token_unset(monkeypatch):
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    app = FastAPI()
    app.add_middleware(AccessGuardMiddleware)

    @app.get("/")
    async def root():
        return {"ok": True}

    with patch("services.access_guard.ACCESS_TOKEN", ""):
        client = TestClient(app)
        assert client.get("/").status_code == 200


def test_blocks_without_token(protected_client):
    assert protected_client.get("/").status_code == 401


def test_allows_bearer_token(protected_client):
    response = protected_client.get(
        "/",
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 200


def test_allows_query_token_and_sets_cookie(protected_client):
    response = protected_client.get("/?token=secret-token")
    assert response.status_code == 200
    assert protected_client.cookies.get("access_token") == "secret-token"
