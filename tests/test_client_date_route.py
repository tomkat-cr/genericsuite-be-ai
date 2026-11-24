"""
Route-level test: "client_date" in the JSON body of the FastAPI /chatbot
route reaches the model layer. The model layer is mocked (no LLM, DB, or
network).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from genericsuite.util.utilities import get_request_body
from genericsuite_ai.fastapilib.endpoints import ai_chatbot_endpoint as route

CONVERSATION = '[{"role": "user", "content": "hi"}]'


@pytest.fixture(name="client_and_calls")
def fixture_client_and_calls(monkeypatch):
    calls = []

    def fake_model(request, **_kwargs):
        calls.append(get_request_body(request))
        return {"response": "ok"}

    monkeypatch.setattr(route, "ai_chatbot_endpoint_model", fake_model)
    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[route.get_current_user] = lambda: {"public_id": "user-1"}
    return TestClient(app), calls


def test_client_date_reaches_model_layer(client_and_calls):
    client, calls = client_and_calls
    response = client.post("/chatbot", json={
        "conversation": CONVERSATION,
        "cid": "c1",
        "client_date": "2026-03-15",
    })
    assert response.status_code == 200
    assert calls[0]["client_date"] == "2026-03-15"
    assert calls[0]["conversation"] == CONVERSATION
    assert calls[0]["cid"] == "c1"


def test_body_without_client_date_is_as_before(client_and_calls):
    client, calls = client_and_calls
    response = client.post("/chatbot", json={
        "conversation": CONVERSATION,
        "cid": "c1",
    })
    assert response.status_code == 200
    assert "client_date" not in calls[0]
    assert calls[0]["conversation"] == CONVERSATION
    assert calls[0]["cid"] == "c1"
