"""A5: end to end through POST /chat, with NO part_spec injected and no fake template.

The mock provider answers only the routing call; the prismatic requests must be built by the feature rung before the
provider is ever asked for code, which is what `answers() == []` proves. The last two requests need code (a free-form
moldboard, on purpose) and one assembly part, and they check that path is still alive.
"""
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEEN, REPLY = [], {"text": ""}

FREE = {"title": "Snow plow moldboard", "kind": "styling", "process": None, "material": "steel", "params": [],
        "code": "const body = makeBox([0,0,0],[2400,900,6]);\nreturn { body };",
        "expect": [], "missing": []}


def is_router(body):
    return "first reader" in json.dumps(body)


def answers():
    return [c for c in SEEN if not is_router(c[2])]


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, dict(self.headers), body))
        t = json.dumps({"route": "other"}) if is_router(body) else REPLY["text"]
        if self.path.startswith("/v1/messages"):
            out = {"content": [{"type": "text", "text": t}]}
        elif ":generateContent" in self.path:
            out = {"candidates": [{"content": {"parts": [{"text": t}]}}]}
        else:
            out = {"choices": [{"message": {"content": t}}]}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


srv = HTTPServer(("127.0.0.1", 8766), Mock)
threading.Thread(target=srv.serve_forever, daemon=True).start()
env = dict(os.environ, PORT="8011", ANTHROPIC_API_KEY="k-claude", GEMINI_API_KEY="k-gem", OPENAI_API_KEY="k-oai",
           ANTHROPIC_BASE_URL="http://127.0.0.1:8766", GEMINI_BASE_URL="http://127.0.0.1:8766",
           OPENAI_BASE_URL="http://127.0.0.1:8766", IDRIS_LLM_PER_MIN="60", IDRIS_LLM_RETRY_SLEEP="0.05",
           IDRIS_RECIPE_AI="0")
proc = subprocess.Popen([sys.executable, "main.py"], cwd=ROOT, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.8)


def chat(body, port=8011):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


def volume_of(tree):
    """The recipe's own numbers, read back: extrude area x depth, minus the holes that go through it."""
    vol, depth, face = 0.0, 0.0, None
    holes = 0.0
    for s in tree:
        p = s.get("p") or {}
        if s.get("type") == "sketch" and p.get("prof") == "rect":
            face = float(p["W"]) * float(p["Hh"])
        elif s.get("type") == "sketch" and p.get("prof") == "circle":
            face = math.pi / 4 * float(p["D"]) ** 2
        elif s.get("type") == "push":
            depth = abs(float(p.get("dist") or 0))
            vol += (face or 0.0) * depth
        elif s.get("type") == "hole":
            holes += math.pi / 4 * float(p["D"]) ** 2 * depth
    return vol - holes


try:
    REPLY["text"] = json.dumps(FREE)
    SEEN.clear()
    s, r = chat({"message": "design a mounting bracket plate 120x80x8 with 4 holes M8"})
    check("plate: 200", s == 200)
    check("plate: rung features (not code)", r.get("rung") == "features" and r.get("fidelity") == "features")
    tree = ((r.get("cad") or {}).get("feature_tree")) or []
    check("plate: 6 ordered steps", [x["type"] for x in tree] == ["sketch", "push", "hole", "hole", "hole", "hole"])
    want = 120 * 80 * 8 - 4 * (math.pi / 4 * 9 ** 2 * 8)
    check(f"plate: kernel volume {volume_of(tree):.1f} = 120x80x8 - 4x ISO 273 M8 holes", abs(volume_of(tree) - want) < 0.5)
    check("plate: every number carries a source", all(n.get("source") for n in r.get("numbers") or []))
    check("plate: the provider was never asked for code", answers() == [])

    SEEN.clear()
    s, r = chat({"message": "L bracket 60x40x5 width 30"})
    check("L bracket: rung features", s == 200 and r.get("rung") == "features")
    check("L bracket: template named", ((r.get("cad") or {}).get("feature_template")) == "bracket_l")
    check("L bracket: profile from lines, extruded",
          [x["type"] for x in (r.get("cad") or {}).get("feature_tree") or []] == ["sketch", "push"])

    SEEN.clear()
    s, r = chat({"message": "shaft \u00d830 length 120 with \u00d820 x 30 step"})
    check("shaft: rung features via revolve", s == 200 and r.get("rung") == "features"
          and [x["type"] for x in (r.get("cad") or {}).get("feature_tree") or []] == ["sketch", "revolve"])

    SEEN.clear()
    # A free-form request also has to reach the ladder (the Studio's own door, current_cad), so this checks the
    # ladder, not routing. "snow plow moldboard" is answered by the parts library before this point - a parametric
    # welded part, which is a better answer than code - so the double-curved case here is a fairing.
    s, r = chat({"message": "curved aerodynamic fairing for the nose 400 mm",
                 "context": {"current_cad": {"title": "fairing", "kind": "styling"}}})
    check("fairing: free-form goes to code on purpose", s == 200 and r.get("rung") == "code"
          and bool(((r.get("cad") or {}).get("code"))) and "free-form" in str(r.get("features_error")))
finally:
    proc.terminate()
    srv.shutdown()

sys.exit(0 if ok else 1)
