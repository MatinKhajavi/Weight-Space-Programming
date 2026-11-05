"""Local stand-in for the Anthropic API: records every request Claude Code sends, returns a tiny valid reply.

Usage: python -I capture_server.py <port> <capture_dir>
Nothing is forwarded anywhere. API key headers are redacted before saving.
"""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, OUT = int(sys.argv[1]), sys.argv[2]
REPLIES = json.load(open(os.environ["CAPTURE_REPLIES"])) if os.environ.get("CAPTURE_REPLIES") else ["CAPTURED-REPLY"]
_k = [0]
os.makedirs(OUT, exist_ok=True)
_lock, _n = threading.Lock(), [0]
REDACT = {"x-api-key", "authorization"}


def _save(handler, body_bytes):
    with _lock:
        _n[0] += 1
        n = _n[0]
    try:
        body = json.loads(body_bytes) if body_bytes else None
    except Exception:
        body = {"_raw": body_bytes.decode("utf-8", "replace")}
    rec = {"n": n, "method": handler.command, "path": handler.path,
           "headers": {k: ("<redacted>" if k.lower() in REDACT else v) for k, v in handler.headers.items()},
           "body": body}
    with open(os.path.join(OUT, f"{n:03d}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return body


def _sse(model):
    text = REPLIES[min(_k[0], len(REPLIES) - 1)]
    _k[0] += 1
    ev = [
        ("message_start", {"type": "message_start", "message": {"id": "msg_capture", "type": "message",
          "role": "assistant", "model": model, "content": [], "stop_reason": None, "stop_sequence": None,
          "usage": {"input_tokens": 1, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
          "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in ev).encode()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        _save(self, b"")
        if self.path.startswith("/v1/models"):
            m = {"id": "claude-opus-5-5", "type": "model", "display_name": "Claude Opus 5.5",
                 "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1000000, "max_tokens": 128000}
            return self._send(200, m if self.path.rstrip("/") != "/v1/models" else {"data": [m], "has_more": False})
        self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": "capture server"}})

    def do_POST(self):
        body = _save(self, self.rfile.read(int(self.headers.get("content-length") or 0))) or {}
        path = self.path.split("?")[0]
        if path.endswith("/count_tokens"):
            return self._send(200, {"input_tokens": 1})
        if path.endswith("/v1/messages"):
            model = body.get("model", "claude-opus-5-5")
            if body.get("stream"):
                return self._send(200, _sse(model), "text/event-stream")
            return self._send(200, {"id": "msg_capture", "type": "message", "role": "assistant", "model": model,
                                    "content": [{"type": "text", "text": "CAPTURED-REPLY"}], "stop_reason": "end_turn",
                                    "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})
        self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": "capture server"}})


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
