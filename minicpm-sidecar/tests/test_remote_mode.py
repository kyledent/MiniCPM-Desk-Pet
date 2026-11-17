"""Remote mode: MINICPM_REMOTE_URL points the gateway at an existing
OpenAI-compatible server (llama-server or vLLM elsewhere) instead of
spawning llama-server. Locks down:

  - URL normalisation (root, trailing slash and /v1 all accepted)
  - nothing is spawned, and stop() never touches the local pid file
  - model id: explicit MINICPM_REMOTE_MODEL wins, else learnt from /v1/models
  - request body: model id, repetition_penalty for vLLM, never a `lora` array
  - warmup uses /v1/completions (llama-server's /completion is not portable)
  - API key sent as a Bearer token only when configured
  - /api/health and /api/onboarding report remote mode, so the Electron
    host does not push the user into the model download
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway import server as server_mod
from gateway.llama_client import LlamaServer, normalise_remote_url


_SSE = (
    b'data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
    b'data: [DONE]\n\n'
)


class _FakeRemote:
    """Minimal OpenAI-compatible server. `has_health=False` behaves like
    vLLM without llama-server's /health route."""

    def __init__(self, *, has_health: bool = True, models=("served-model",)) -> None:
        self.has_health = has_health
        self.models = list(models)
        self.requests: list[httpx.Request] = []
        self.bodies: dict[str, dict] = {}

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200 if self.has_health else 404, json={"status": "ok"})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        if path == "/v1/chat/completions":
            self.bodies[path] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, content=_SSE, headers={"content-type": "text/event-stream"})
        if path == "/v1/completions":
            self.bodies[path] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"choices": [{"text": " "}]})
        return httpx.Response(404)


def _remote_server(fake: _FakeRemote, monkeypatch, *, model: str | None = None,
                   api_key: str | None = None) -> LlamaServer:
    if model:
        monkeypatch.setenv("MINICPM_REMOTE_MODEL", model)
    else:
        monkeypatch.delenv("MINICPM_REMOTE_MODEL", raising=False)
    if api_key:
        monkeypatch.setenv("MINICPM_REMOTE_API_KEY", api_key)
    else:
        monkeypatch.delenv("MINICPM_REMOTE_API_KEY", raising=False)
    srv = LlamaServer(model_path=None, remote_url="http://remote.test:8080/v1/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    srv._client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler),
                                    base_url=srv.remote_url, headers=headers)
    return srv


async def _drain(srv: LlamaServer, **kw) -> list[tuple[str, str]]:
    out = []
    async for item in srv.stream_chat(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8, temperature=0.5, top_p=0.9, top_k=20,
        repetition_penalty=1.05, enable_thinking=False, **kw,
    ):
        out.append(item)
    return out


# ── URL normalisation ────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "http://h:8080", "http://h:8080/", "http://h:8080/v1", "http://h:8080/v1/", " http://h:8080/V1 ",
])
def test_normalise_remote_url_accepts_common_forms(raw):
    assert normalise_remote_url(raw) == "http://h:8080"


def test_normalise_remote_url_empty_is_local_mode():
    assert normalise_remote_url("") is None
    assert normalise_remote_url(None) is None


def test_normalise_remote_url_rejects_non_http():
    with pytest.raises(ValueError):
        normalise_remote_url("ftp://h/")
    with pytest.raises(ValueError):
        normalise_remote_url("h:8080")


def test_local_mode_is_default(monkeypatch):
    monkeypatch.delenv("MINICPM_REMOTE_URL", raising=False)
    srv = LlamaServer(model_path=None)
    assert srv.remote is False


def test_remote_mode_from_env_drops_adapters(monkeypatch, tmp_path):
    monkeypatch.setenv("MINICPM_REMOTE_URL", "http://h:1/v1")
    srv = LlamaServer(model_path=None, adapters=[tmp_path / "a.gguf"])
    assert srv.remote is True
    assert srv.remote_url == "http://h:1"
    assert srv.adapter_paths == []


# ── chat / warmup / health ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_chat_learns_model_and_sends_portable_body(monkeypatch):
    fake = _FakeRemote(has_health=False, models=["openbmb/MiniCPM5-1B"])
    srv = _remote_server(fake, monkeypatch)
    out = await _drain(srv, lora=[{"id": 0, "scale": 1.0}])
    assert out == [("content", "hel"), ("content", "lo")]
    body = fake.bodies["/v1/chat/completions"]
    assert body["model"] == "openbmb/MiniCPM5-1B"
    assert body["repetition_penalty"] == pytest.approx(1.05)
    assert body["repeat_penalty"] == pytest.approx(1.05)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "lora" not in body


@pytest.mark.asyncio
async def test_explicit_model_wins_over_models_list(monkeypatch):
    fake = _FakeRemote(models=["other"])
    srv = _remote_server(fake, monkeypatch, model="minicpm-dev")
    await _drain(srv)
    assert fake.bodies["/v1/chat/completions"]["model"] == "minicpm-dev"
    health = await srv.health()
    assert health["status"] == "ok" and health["model"] == "minicpm-dev"


@pytest.mark.asyncio
async def test_api_key_is_bearer_only_when_set(monkeypatch):
    fake = _FakeRemote()
    srv = _remote_server(fake, monkeypatch, model="m", api_key="sekrit")
    await _drain(srv)
    assert fake.requests[-1].headers["authorization"] == "Bearer sekrit"

    fake2 = _FakeRemote()
    srv2 = _remote_server(fake2, monkeypatch, model="m")
    await _drain(srv2)
    assert "authorization" not in fake2.requests[-1].headers


@pytest.mark.asyncio
async def test_warmup_uses_v1_completions(monkeypatch):
    fake = _FakeRemote(models=["m1"])
    srv = _remote_server(fake, monkeypatch)
    await srv.complete_once(prompt=" ", max_tokens=1)
    body = fake.bodies["/v1/completions"]
    assert body == {"model": "m1", "prompt": " ", "max_tokens": 1, "temperature": 0.0, "stream": False}


@pytest.mark.asyncio
async def test_health_none_when_unreachable(monkeypatch):
    async def down(request):
        raise httpx.ConnectError("refused", request=request)
    srv = LlamaServer(model_path=None, remote_url="http://remote.test:1")
    srv._client = httpx.AsyncClient(transport=httpx.MockTransport(down), base_url=srv.remote_url)
    assert await srv.health() is None
    assert srv.alive is True   # configured; reachability is health()'s job


@pytest.mark.asyncio
async def test_chat_error_names_the_remote(monkeypatch):
    async def bad(request):
        return httpx.Response(500, content=b"boom")
    srv = LlamaServer(model_path=None, remote_url="http://remote.test:1")
    srv.remote_model = "m"
    srv._client = httpx.AsyncClient(transport=httpx.MockTransport(bad), base_url=srv.remote_url)
    with pytest.raises(RuntimeError, match="remote server http://remote.test:1"):
        await _drain(srv)


# ── lifecycle ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_spawns_nothing_and_tolerates_down_server(monkeypatch, tmp_path):
    monkeypatch.setenv("MINICPM_REMOTE_WAIT", "0")
    srv = LlamaServer(model_path=None, remote_url="http://127.0.0.1:9")
    srv._pid_file = tmp_path / "llama.pid"
    srv._pid_file.write_text("12345")

    def boom(*a, **k):
        raise AssertionError("remote mode must not spawn llama-server")
    monkeypatch.setattr("subprocess.Popen", boom)

    await srv.start()
    assert srv.alive is True and srv._proc is None
    await srv.swap_model(tmp_path / "x.gguf")         # recorded, no restart
    assert srv.model_path == (tmp_path / "x.gguf").resolve()
    await srv.reload_adapters([tmp_path / "a.gguf"])  # ignored
    assert srv.adapter_paths == []
    await srv.stop()
    assert srv.alive is False
    assert srv._pid_file.read_text() == "12345"       # not ours to clear


# ── HTTP surface the Electron host reads ─────────────────────────────

def test_health_and_onboarding_report_remote(monkeypatch, tmp_path):
    monkeypatch.setenv("MINICPM_REMOTE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("MINICPM_REMOTE_MODEL", "minicpm-dev")
    monkeypatch.setenv("MINICPM_REMOTE_WAIT", "0")
    monkeypatch.setenv("MINICPM_ADAPTER_DIR", str(tmp_path / "adapters"))
    monkeypatch.setattr(server_mod.ClawdBridge, "post", lambda *a, **k: None)
    app = server_mod.build_app(initial_model=tmp_path / "missing" / "model.gguf")
    with TestClient(app) as client:
        h = client.get("/api/health").json()
        assert h["ok"] is True
        assert h["alive"] is True
        assert h["backend"] == "remote"
        assert h["remote_url"] == "http://127.0.0.1:9"
        assert h["model_name"] == "minicpm-dev"
        o = client.get("/api/onboarding").json()
        assert o["model_present"] is True
        assert o["stage_hint"] == "ready"


# ── security hardening (review findings F5 / F6) ────────────────────

@pytest.mark.parametrize("raw", [
    "http://user:s3cretpw@h:8080", "http://s3cretpw@h:8080/v1", "http://h:8080/v1?key=s3cretpw", "http://h:8080/#s3cretpw",
])
def test_remote_url_rejects_credentials_query_fragment(raw):
    with pytest.raises(ValueError) as ei:
        normalise_remote_url(raw)
    assert "s3cretpw" not in str(ei.value)


def test_malformed_remote_wait_falls_back(monkeypatch):
    monkeypatch.setenv("MINICPM_REMOTE_WAIT", "soon")
    srv = LlamaServer(model_path=None, remote_url="http://h:1")
    assert srv._remote_wait == 10.0
    monkeypatch.setenv("MINICPM_REMOTE_WAIT", "inf")
    assert LlamaServer(model_path=None, remote_url="http://h:1")._remote_wait == 10.0


def test_remote_mode_refuses_web_origins(monkeypatch, tmp_path):
    monkeypatch.setenv("MINICPM_REMOTE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("MINICPM_REMOTE_MODEL", "m")
    monkeypatch.setenv("MINICPM_REMOTE_WAIT", "0")
    monkeypatch.setenv("MINICPM_ADAPTER_DIR", str(tmp_path / "adapters"))
    monkeypatch.setattr(server_mod.ClawdBridge, "post", lambda *a, **k: None)
    app = server_mod.build_app(initial_model=None)
    with TestClient(app) as client:
        assert client.get("/api/health", headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.post("/api/chat", json={"messages": [{"role": "user", "content": "x"}]},
                           headers={"Origin": "http://evil.example"}).status_code == 403
        # The pet's renderer (file:// page) and its main process (no Origin) still work.
        assert client.get("/api/health", headers={"Origin": "null"}).status_code == 200
        assert client.get("/api/health", headers={"Origin": "file://"}).status_code == 200
        assert client.get("/api/health").status_code == 200


def test_local_mode_cors_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("MINICPM_REMOTE_URL", raising=False)
    monkeypatch.setenv("MINICPM_ADAPTER_DIR", str(tmp_path / "adapters"))
    monkeypatch.setattr(server_mod.ClawdBridge, "post", lambda *a, **k: None)
    app = server_mod.build_app(initial_model=None)
    with TestClient(app) as client:
        assert client.get("/api/health", headers={"Origin": "https://example.org"}).status_code == 200


def test_remote_mode_refuses_lan_host_without_opt_in(monkeypatch, capsys):
    from gateway import __main__ as entry
    monkeypatch.setenv("MINICPM_REMOTE_URL", "http://127.0.0.1:9")
    monkeypatch.delenv("MINICPM_ALLOW_LAN", raising=False)
    assert entry.main(["--host", "0.0.0.0", "--port", "18799"]) == 2
    assert "MINICPM_ALLOW_LAN" in capsys.readouterr().err
