from __future__ import annotations

from autonomous_sre.interfaces import simulate_prod


def test_build_payload_shape():
    payload = simulate_prod._build_payload(spike_mode=False)
    assert payload["service"]
    assert payload["alert_name"]
    assert payload["severity"] in {"low", "high", "critical"}
    assert "cpu_pct" in payload["metrics"]
    assert "latency_ms" in payload["metrics"]


def test_headers_include_api_key_when_provided():
    headers = simulate_prod._headers("abc123")
    assert headers["Content-Type"] == "application/json"
    assert headers["x-api-key"] == "abc123"


def test_headers_without_api_key():
    headers = simulate_prod._headers(None)
    assert headers == {"Content-Type": "application/json"}
