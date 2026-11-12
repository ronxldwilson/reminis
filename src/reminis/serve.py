"""The database is the model; this hands it to HTTP.

``reminis serve`` exposes a model stored in a reminis database as an
OpenAI-compatible API, so anything that can talk to a chat completions
endpoint -- openai, langchain, a frontend, curl -- can talk to the rows.
The model is loaded once at startup and every request is a forward pass
against it; nothing is re-read from SQLite per request.

Endpoints:

  GET  /health               -> {"status": "ok"}
  GET  /v1/models            -> the served model's id and metadata
  POST /v1/chat/completions  -> chat, with or without streaming
  POST /v1/completions       -> raw prompt completion

Streaming is server-sent events, one ``data:`` chunk per token, terminated
by ``data: [DONE]``. One generation runs at a time; concurrent clients are
queued on a lock rather than raced against a shared cache.

Chat prompts are rendered by the Jinja template stored in the database,
when ``jinja2`` is importable (``pip install reminis[serve]``). Without
it, a plain ChatML wrap is used instead -- sound for ChatML models, and
the only option on a three-package install. ``chat_template_kwargs`` in
the request body is passed through to the template; ``enable_thinking``
defaults to false so responses arrive directly rather than after a
reasoning preamble.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from reminis.errors import UnsupportedModel
from reminis.kvcache import KVCache

__all__ = ["Session", "run_server"]


class _ClientGone(Exception):
    """The client disconnected mid-stream; unwind without writing more."""


def _config_stop_ids(meta: dict) -> set[int]:
    """Every end-of-text id the model declares, wherever it was recorded.

    ``tokenizer.ggml.eos_token_id`` is the single GGUF-style record;
    ``config.eos_token_id`` is the config.json one and may list several
    (Qwen chat models stop on both ``<|im_end|>`` and ``<|endoftext|>``).
    A chat turn that ends on the id nothing stopped on would otherwise
    keep going into the next im_start block.
    """
    ids = set()
    eos = meta.get("tokenizer.ggml.eos_token_id")
    if eos is not None:
        try:
            ids.add(int(eos))
        except ValueError:
            pass
    raw = meta.get("config.eos_token_id")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [int(raw)]
        if isinstance(parsed, int):
            parsed = [parsed]
        ids.update(int(x) for x in parsed)
    return ids


def _fallback_chat(messages: list[dict], think: bool) -> str:
    """A plain ChatML wrap, for a database with no usable template."""
    out = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):  # content parts -> text only
            content = "".join(part.get("text", "") for part in content
                              if isinstance(part, dict))
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    out.append("<|im_start|>assistant\n"
               + ("<think>\n" if think else "<think>\n\n</think>\n\n"))
    return "".join(out)


def _render_chat(tok, messages: list[dict], template_kwargs: dict | None) -> str:
    """Render the chat template stored in the database, if possible."""
    kwargs = dict(template_kwargs or {})
    src = getattr(tok, "chat_template", "") or ""
    if src:
        try:
            import jinja2
        except ImportError:
            pass
        else:
            def raise_exception(msg):
                raise UnsupportedModel(f"The model's chat template failed: {msg}")

            env = jinja2.Environment(
                trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
            )
            env.globals["raise_exception"] = raise_exception
            env.filters["tojson"] = (
                lambda v: json.dumps(v, ensure_ascii=False))
            tools = kwargs.pop("tools", None)
            rendered = env.from_string(src).render(
                messages=messages,
                add_generation_prompt=True,
                tools=tools,
                **kwargs,
            )
            return rendered
    think = bool(kwargs.get("enable_thinking", False))
    return _fallback_chat(messages, think)


def _json_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The request body is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("The request body must be a JSON object")
    return parsed


class Session:
    """One model, loaded once, generating behind a lock."""

    def __init__(self, db_path: str, backend=None, pack_bits=None,
                 pack_group: int = 128, kv_bits: int | None = None,
                 draft: str | None = None, draft_tokens: int = 4,
                 name: str | None = None):
        from reminis.backend import select as select_backend
        from reminis.model import Model

        self.db_path = db_path
        self.chosen = select_backend("inference", backend)
        self.model = Model(db_path, backend=self.chosen, pack_bits=pack_bits,
                           pack_group=pack_group)
        self.tok = self.model.tokenizer
        self.id = name or Path(db_path).stem
        self.pack_bits = pack_bits
        self.kv_bits = kv_bits
        self.draft = draft
        self.draft_tokens = int(draft_tokens)
        self.stop_ids = _config_stop_ids(dict(self.model.meta))
        self.lock = threading.Lock()

    # ------------------------------------------------------------ requests

    def _generate(self, text: str, max_tokens: int, temperature: float,
                  top_p: float, seed: int | None, stop: list[str],
                  on_piece) -> dict:
        """Run the decode loop; call ``on_piece(text)`` per token.

        Returns dict(completion, prompt_tokens, completion_tokens,
        finish_reason). Mirrors ``reminis.infer.generate`` without the
        printing, so nothing is re-read per request and the model stays
        resident between calls.
        """
        from reminis.infer import _sample

        model = self.model
        tok = self.tok
        model.reset_sequence_state()
        tokens = tok.encode(text)
        if not tokens:
            raise ValueError("The prompt encoded to zero tokens")
        if len(tokens) >= model.cfg.context_length:
            raise ValueError(
                f"The prompt is {len(tokens)} tokens and this model's "
                f"context is {model.cfg.context_length}"
            )

        rng = np.random.default_rng(seed)
        draft = self.draft
        headroom = 3 * self.draft_tokens + 5 if draft else 0
        capacity = len(tokens) + max_tokens + headroom
        cache = KVCache(
            model.cfg.n_layers, capacity=capacity, backend=self.chosen,
            quantize_bits=self.kv_bits,
            last_append_layer=getattr(model.cfg, "last_kv_layer", None),
        )

        speculator = None
        if draft:
            from reminis.speculative import Speculator, open_drafter
            drafter = open_drafter(draft, model, capacity=capacity,
                                   temperature=temperature, top_p=top_p,
                                   rng=rng, backend=self.chosen,
                                   pack_bits=self.pack_bits)
            speculator = Speculator(model, drafter, self.draft_tokens)

        logits = model.forward(tokens, cache, offset=0)

        produced: list[int] = []
        text_so_far = ""
        finish = "length"

        def emit(token_id: int) -> bool:
            """Decode and publish one token; False stops the run."""
            nonlocal text_so_far, finish
            piece = tok.decode_one(token_id)
            joined = text_so_far + piece
            for needle in stop:
                at = joined.find(needle)
                if at != -1:
                    # Cut the emitted text at the stop string. Tokens before
                    # it have already gone out on the stream; the remainder
                    # is dropped rather than sent.
                    remainder = joined[:at]
                    if len(remainder) > len(text_so_far):
                        on_piece(remainder[len(text_so_far):])
                    text_so_far = remainder
                    finish = "stop"
                    return False
            on_piece(piece)
            text_so_far = joined
            return True

        if speculator is not None:
            def on_id(token_id: int):
                if not emit(token_id):
                    # Signal the speculator to stop by returning early from
                    # keep(): raising is caught below.
                    raise _ClientGone

            try:
                produced = speculator.generate(
                    logits, tokens, cache, max_tokens, temperature, top_p,
                    rng, eos_id=self.stop_ids, stop_at_eos=True,
                    on_token=on_id,
                )
            except _ClientGone:
                pass
            if finish == "length" and len(produced) and produced[-1] in self.stop_ids:
                finish = "stop"
        else:
            for _ in range(max_tokens):
                token_id = _sample(logits, temperature, top_p, rng)
                if token_id in self.stop_ids:
                    finish = "stop"
                    break
                produced.append(token_id)
                if not emit(token_id):
                    break
                logits = model.forward([token_id], cache, offset=cache.length)

        if speculator is not None:
            speculator.drafter.close()

        return {
            "completion": text_so_far if stop else tok.decode(produced),
            "prompt_tokens": len(tokens),
            "completion_tokens": len(produced),
            "finish_reason": finish,
        }

    # ------------------------------------------------------------ control

    def warmup(self):
        """Touch every weight once, so the first request pays no load."""
        cache = KVCache(self.model.cfg.n_layers, capacity=8,
                        backend=self.chosen)
        prompt = self.tok.encode("<|im_start|>user\n\nhi", add_special=False) or [0]
        self.model.forward(prompt[:1], cache, offset=0)

    def models_document(self) -> dict:
        return {
            "object": "list",
            "data": [{
                "id": self.id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "reminis",
            }],
        }


# ------------------------------------------------------------------- HTTP


def _sse_send(handler, payload: dict):
    body = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
    handler._chunk(body)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "reminis"

    # -- plumbing ---------------------------------------------------------

    @property
    def session(self) -> Session:
        return self.server.session  # set by run_server

    def log_message(self, fmt, *args):  # keep access logs on stderr, brief
        pass

    def _chunk(self, data: bytes):
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _end_chunked(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, kind="invalid_request_error"):
        self._json({"error": {"message": message, "type": kind}}, status)

    # -- routing ----------------------------------------------------------

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            return self._json({"status": "ok"})
        if self.path == "/v1/models":
            return self._json(self.session.models_document())
        return self._error(404, f"Unknown path: {self.path}", "not_found")

    def do_POST(self):
        try:
            body = _json_body(self)
        except ValueError as exc:
            return self._error(400, str(exc))

        if self.path == "/v1/chat/completions":
            return self._chat(body)
        if self.path == "/v1/completions":
            return self._completions(body)
        return self._error(404, f"Unknown path: {self.path}", "not_found")

    # -- endpoints ----------------------------------------------------------

    def _params(self, body: dict) -> dict:
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        try:
            max_tokens = int(max_tokens) if max_tokens is not None else 512
        except (TypeError, ValueError):
            raise ValueError("max_tokens must be an integer")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        def num(key, default):
            value = body.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number")

        stop = body.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        stop = [str(s) for s in stop]

        seed = body.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                raise ValueError("seed must be an integer")

        return {
            "max_tokens": max_tokens,
            "temperature": num("temperature", 1.0),
            "top_p": num("top_p", 0.95),
            "seed": seed,
            "stop": stop,
            "stream": bool(body.get("stream", False)),
        }

    def _chat(self, body: dict):
        session = self.session
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._error(400, "messages must be a non-empty list")
        try:
            params = self._params(body)
        except ValueError as exc:
            return self._error(400, str(exc))

        kwargs = body.get("chat_template_kwargs")
        if kwargs is not None and not isinstance(kwargs, dict):
            return self._error(400, "chat_template_kwargs must be an object")
        if kwargs is None:
            kwargs = {"enable_thinking": False}
        text = _render_chat(session.tok, messages, kwargs)
        if "tools" in body and isinstance(body["tools"], list):
            # re-render with tools available to the template
            text = _render_chat(session.tok, messages,
                                {**kwargs, "tools": body["tools"]})
        return self._run(text, params, kind="chat")

    def _completions(self, body: dict):
        session = self.session
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            prompt = "".join(prompt)
        if not isinstance(prompt, str) or not prompt:
            return self._error(400, "prompt must be a non-empty string")
        try:
            params = self._params(body)
        except ValueError as exc:
            return self._error(400, str(exc))
        return self._run(prompt, params, kind="completion")

    def _run(self, prompt_text: str, params: dict, kind: str):
        session = self.session
        stream = params.pop("stream")
        chat = kind == "chat"
        stop_ids = session.stop_ids

        with session.lock:
            start = time.time()
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def on_piece(piece: str):
                    delta: dict = {"content": piece}
                    if not on_piece.role_sent:  # type: ignore[attr-defined]
                        delta["role"] = "assistant"
                        on_piece.role_sent = True  # type: ignore[attr-defined]
                    _sse_send(self, {
                        "id": completion_id,
                        "object": "chat.completion.chunk" if chat
                        else "text_completion.chunk",
                        "created": int(start),
                        "model": session.id,
                        "choices": [{
                            "index": 0,
                            "delta": delta if chat else None,
                            "text": None if chat else piece,
                            "finish_reason": None,
                        }],
                    })

                on_piece.role_sent = False  # type: ignore[attr-defined]
                try:
                    result = session._generate(
                        prompt_text, stop=params["stop"],
                        on_piece=on_piece, **{
                            "max_tokens": params["max_tokens"],
                            "temperature": params["temperature"],
                            "top_p": params["top_p"],
                            "seed": params["seed"],
                        })
                except _ClientGone:
                    result = None
                except (UnsupportedModel, ValueError) as exc:
                    _sse_send(self, {"error": {"message": str(exc)}})
                    self._chunk(b"data: [DONE]\n\n")
                    self._end_chunked()
                    return
                finish = result["finish_reason"] if result else "stop"
                _sse_send(self, {
                    "id": completion_id,
                    "object": "chat.completion.chunk" if chat
                    else "text_completion.chunk",
                    "created": int(start),
                    "model": session.id,
                    "choices": [{
                        "index": 0,
                        "delta": {} if chat else None,
                        "text": None if chat else "",
                        "finish_reason": finish,
                    }],
                })
                self._chunk(b"data: [DONE]\n\n")
                self._end_chunked()
                return

            try:
                result = session._generate(
                    prompt_text, stop=params["stop"],
                    on_piece=lambda piece: None, **{
                        "max_tokens": params["max_tokens"],
                        "temperature": params["temperature"],
                        "top_p": params["top_p"],
                        "seed": params["seed"],
                    })
            except (UnsupportedModel, ValueError) as exc:
                return self._error(400, str(exc))
            except BrokenPipeError:
                return

            if chat:
                obj = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(start),
                    "model": session.id,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": result["completion"]},
                        "finish_reason": result["finish_reason"],
                    }],
                    "usage": {
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens": result["prompt_tokens"]
                        + result["completion_tokens"],
                    },
                }
            else:
                obj = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": int(start),
                    "model": session.id,
                    "choices": [{
                        "index": 0,
                        "text": result["completion"],
                        "finish_reason": result["finish_reason"],
                    }],
                    "usage": {
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens": result["prompt_tokens"]
                        + result["completion_tokens"],
                    },
                }
            return self._json(obj)


def run_server(db_path: str, host: str = "127.0.0.1", port: int = 8080,
               backend=None, pack_bits=None, pack_group: int = 128,
               kv_bits: int | None = None,
               draft: str | None = None, draft_tokens: int = 4,
               name: str | None = None, warmup: bool = True):
    """Load the model and serve it until interrupted."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    print(f"Loading {db_path} ...", flush=True)
    session = Session(db_path, backend=backend, pack_bits=pack_bits,
                      pack_group=pack_group, kv_bits=kv_bits,
                      draft=draft, draft_tokens=draft_tokens, name=name)
    print(f"serving '{session.id}' | {session.model.cfg.arch} | "
          f"{session.model.cfg.n_layers} layers | "
          f"{session.chosen.describe()}", flush=True)
    if warmup:
        print("warming up ...", flush=True)
        session.warmup()
        print("ready", flush=True)

    server = ThreadingHTTPServer((host, port), _Handler)
    server.session = session  # type: ignore[attr-defined]
    server.daemon_threads = True
    print(f"listening on http://{host}:{port} "
          f"(POST /v1/chat/completions, /v1/completions)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()
        session.model.close()
