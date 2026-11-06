import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import google_mcp_server as gm


class _Request:
    def __init__(self, message_id):
        self.message_id = message_id

    def execute(self):
        return {
            "id": self.message_id,
            "payload": {"headers": [{"name": "From", "value": "Sender <sender@example.com>"}]},
            "sizeEstimate": 42,
        }


class _SequentialService:
    def __init__(self):
        self.get_calls = 0

    def users(self):
        return self

    def messages(self):
        return self

    def get(self, **kwargs):
        self.get_calls += 1
        return _Request(kwargs["id"])


def test_metadata_fallback_caps_sequential_provider_calls():
    service = _SequentialService()
    with pytest.raises(ValueError, match="128-item execution limit"):
        gm._gmail_get_metadata_batch(
            service,
            [f"m{index}" for index in range(129)],
            max_messages=129,
        )
    assert service.get_calls == 0

    results = gm._gmail_get_metadata_batch(
        service,
        [f"m{index}" for index in range(128)],
        max_messages=128,
    )
    assert service.get_calls == 128
    assert len(results) == 128
    assert results[-1]["id"] == "m127"


def test_metadata_helper_batches_500_ids_in_100_provider_calls():
    batch_sizes = []

    class _BatchRequest:
        def __init__(self, message_id):
            self.message_id = message_id

        def execute(self):
            raise AssertionError("sequential GET should not execute")

    class _Batch:
        def __init__(self, callback):
            self.callback = callback
            self.requests = []

        def add(self, request, request_id):
            self.requests.append((request_id, request))

        def execute(self):
            batch_sizes.append(len(self.requests))
            for request_id, request in self.requests:
                self.callback(
                    request_id,
                    {"id": request.message_id, "payload": {"headers": []}},
                    None,
                )

    class _BatchService(_SequentialService):
        def new_batch_http_request(self, callback):
            return _Batch(callback)

        def get(self, **kwargs):
            self.get_calls += 1
            assert "fields" not in kwargs
            return _BatchRequest(kwargs["id"])

    batch_service = _BatchService()
    results = gm._gmail_get_metadata_batch(
        batch_service,
        [f"m{index}" for index in range(500)],
        max_messages=500,
        preserve_raw_response=True,
    )
    assert batch_sizes == [5] * 100
    assert batch_service.get_calls == 500
    assert len(results) == 500
    assert results[-1]["id"] == "m499"
    assert results[-1]["payload"]["headers"] == []
    assert "headers" not in results[-1]


def test_public_metadata_batch_preserves_raw_gmail_response():
    service = _SequentialService()

    class _Client:
        def get_service(self, api_name, api_version):
            assert (api_name, api_version) == ("gmail", "v1")
            return service, False

        def is_session_cached(self):
            return False

    token = gm.ACTIVE_GOOGLE_CLIENT.set(_Client())
    try:
        payload = json.loads(asyncio.run(gm.gmail_batch_get_metadata(["m1"])))
    finally:
        gm.ACTIVE_GOOGLE_CLIENT.reset(token)

    assert payload["ok"] is True
    assert payload["data"]["messages"] == [_Request("m1").execute()]
    assert payload["data"]["messages"][0]["payload"]["headers"][0]["name"] == "From"
