"""Local Jev-style decision server — llama.cpp logits → calibrated probabilities.

Exposes the graphwiz-jev option-scoring pipeline (see ``llamacpp.py``) as a tiny
HTTP decision endpoint, so consumers like predictive-agent get calibrated
choice probabilities from the laptop GPU without touching llama.cpp directly.

Constraints: stdlib only (matches ``llamacpp.py``), one decision per request
(one forward pass), fitted temperature applied from ``checkpoint/temperature.json``
when present (the ``calibrate-lm-head.py`` output).

Usage:
    python scripts/decide-server.py --port 8787 --llama-url http://127.0.0.1:8080

Test:
    curl -s :8787/decide -H 'Content-Type: application/json' \\
         -d '{"question": "...", "options": [{"id":"low"},{"id":"high"}]}'
"""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llamacpp import OPTIONS, option_logprobs

FLOOR = -30.0


def build_prompt(question, options):
    """Gemma 4 prompt with thinking disabled, option list rendered as letters —
    the same template ``llamacpp.py`` uses for CommonsenseQA, generalised to any
    1..5 options (letters A..E)."""
    text = f"<|turn>user\n{question}\n"
    for letter, option in zip(OPTIONS, options):
        text += f"{letter}. {option}\n"
    text += "Answer:<turn|>\n"
    text += "<|turn>model\n<|channel>thought\n<channel|>"
    return text


def scaled_probs(log_probs, temperature):
    """Logits → probabilities at a given temperature (overflow-free; -inf-safe
    via FLOOR). Temperature 1.0 returns the raw model distribution."""
    scaled = [max(lp, FLOOR) / temperature for lp in log_probs]
    z = sum(math.exp(x) for x in scaled)
    return [math.exp(x) / z for x in scaled]


class DecisionServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, addr, llama_url, temperature, calibrated):
        super().__init__(addr, Handler)
        self.llama_url = llama_url
        self.temperature = temperature
        self.calibrated = calibrated


class Handler(BaseHTTPRequestHandler):
    server_version = "decide-server/0.1"

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"status": "ok", "calibrated": self.server.calibrated})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.rstrip("/") != "/decide":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        question = body.get("question", "")
        options = body.get("options", [])
        if not question or not isinstance(options, list) or not 1 <= len(options) <= len(OPTIONS):
            self._json(400, {"error": "requires 'question' and 1..5 'options' entries"})
            return
        texts = [o.get("text", str(o)) if isinstance(o, dict) else str(o) for o in options]
        temperature = body.get("temperature", self.server.temperature)

        try:
            log_probs = [
                max(lp, FLOOR)
                for lp in option_logprobs(build_prompt(question, texts), base_url=self.server.llama_url)
            ]
        except Exception as exc:  # llama.cpp unreachable / malformed response
            self._json(502, {"error": f"model backend error: {exc}"})
            return

        raw = scaled_probs(log_probs, 1.0)
        probs = scaled_probs(log_probs, temperature)
        choice = max(range(len(options)), key=lambda i: probs[i])
        self._json(200, {
            "choice": choice,
            "option_ids": [o.get("id", str(i)) for i, o in enumerate(options)],
            "probabilities": probs[: len(options)],
            "raw_probabilities": raw[: len(options)],
            "log_probs": log_probs[: len(options)],
            "temperature": temperature,
            "calibrated": self.server.calibrated,
        })

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # decisions are one line of JSON each; keep the log quiet


def load_temperature(checkpoint):
    """Fitted temperature from calibrate-lm-head.py, or 1.0 (uncalibrated)."""
    path = Path(checkpoint) / "temperature.json"
    if path.exists():
        return float(json.loads(path.read_text())["temperature"]), True
    return 1.0, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8080")
    parser.add_argument("--checkpoint", default="checkpoint")
    args = parser.parse_args()

    temperature, calibrated = load_temperature(args.checkpoint)
    server = DecisionServer((args.host, args.port), args.llama_url, temperature, calibrated)
    print(f"decide-server on :{args.port} -> llama {args.llama_url}"
          f" (temperature={temperature:.3f}, calibrated={calibrated})")
    server.serve_forever()
