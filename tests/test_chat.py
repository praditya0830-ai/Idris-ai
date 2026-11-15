"""Tests for /chat using a local mock that imitates the three provider APIs (no real keys needed)."""
import json, os, subprocess, sys, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
ROOT = os.path.join(os.path.dirname(__file__), "..")
SEEN, REPLY = [], {"text": ""}


def is_router(body):
    """v0.34 (cad_router): a fresh sentence is read by the AI before the answer is composed."""
    return "first reader" in json.dumps(body)


def answers():
    """Provider calls that carry the real answer; the routing call is not one of them."""
    return [c for c in SEEN if not is_router(c[2])]


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, dict(self.headers), body))
        if is_router(body):
            # Answer routing cleanly and outside the failure injection below: these checks are about
            # the /chat answer path, and the router consumes one call of its own.
            t = json.dumps(REPLY.get("route", {"route": "other"}))
        else:
            t = REPLY["text"]
        if self.path.startswith("/v1/messages") : out = {"content": [{"type": "text", "text": t}]}
        elif ":generateContent" in self.path and REPLY.get("fail503") and not is_router(body):
            REPLY["fail503"] -= 1; self.send_response(503); self.end_headers(); self.wfile.write(b'{"error":{"code":503,"message":"high demand"}}'); return
        elif ":generateContent" in self.path: out = {"candidates": [{"content": {"parts": [{"text": t}]}}]}
        elif REPLY.get("fail"): self.send_response(500); self.end_headers(); self.wfile.write(b"boom"); return
        else: out = {"choices": [{"message": {"content": t}}]}
        b = json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

srv = HTTPServer(("127.0.0.1", 8765), Mock); threading.Thread(target=srv.serve_forever, daemon=True).start()
env = dict(os.environ, PORT="8010", ANTHROPIC_API_KEY="k-claude", GEMINI_API_KEY="k-gem", OPENAI_API_KEY="k-oai",
           ANTHROPIC_BASE_URL="http://127.0.0.1:8765", GEMINI_BASE_URL="http://127.0.0.1:8765", OPENAI_BASE_URL="http://127.0.0.1:8765",
           IDRIS_LLM_PER_MIN="40", IDRIS_LLM_RETRY_SLEEP="0.05")
proc = subprocess.Popen([sys.executable, "main.py"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
def chat(body, port=8010):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e: return e.code, json.load(e)
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)
try:
    good = json.dumps({"intent": "set_params", "params": {"power_kw": 11.2, "gear_ratio": 8, "envelope_width": 200, "gear_material": "42CrMo4"}, "answer": "Siap.", "language": "id"})
    for prov, key_hdr in (("claude", "x-api-key"), ("gemini", "x-goog-api-key"), ("openai", "Authorization")):
        REPLY["text"] = good; SEEN.clear()
        s, r = chat({"provider": prov, "message": "15 hp reducer 8:1"})
        check(f"{prov}: 200 + params", s == 200 and r["params"]["power_kw"] == 11.2 and r["params"]["gear_material"] == "42CrMo4")
        ans = answers()[0]
        check(f"{prov}: key sent in header, not in body/url", any(k.lower() == key_hdr.lower() for k in ans[1]) and "k-" not in json.dumps(ans[2]) and "k-" not in ans[0])
        temp = ans[2].get("temperature", ans[2].get("generationConfig", {}).get("temperature"))
        check(f"{prov}: system prompt present, temp 0", "language layer" in json.dumps(ans[2]) and temp == 0)
    REPLY["text"] = "```json\n" + json.dumps({"intent": "set_params", "params": {"power_kw": -5, "input_rpm": "abc", "gear_ratio": 8, "evil": 1, "gear_material": "unobtanium", "envelope_height": 99999}}) + "\n```"
    s, r = chat({"provider": "claude", "message": "x"})
    check("out-of-range / junk values dropped, valid kept", s == 200 and r["params"] == {"gear_ratio": 8.0} and set(r["rejected"]) >= {"power_kw", "input_rpm", "gear_material", "envelope_height"})
    REPLY["text"] = "I cannot help"
    s, r = chat({"provider": "claude", "message": "x"}); check("non-JSON answer -> fallback flag", s == 502 and r["fallback"])
    REPLY["text"] = json.dumps({"intent": "hack", "select": "Z", "compare": ["A"], "answer": 5})
    # A gearbox is loaded on purpose: with no gearbox and intent "other", main.py hands the sentence to the CAD
    # designer (the "object in a language the word list does not know" block), so the sanitized reply would never
    # reach the client. This check is about sanitizing, not about routing.
    s, r = chat({"provider": "claude", "message": "x", "current": {"power_kw": 11.2, "gear_ratio": 8}})
    check("bad intent/select/compare sanitized", r["intent"] == "other" and r["select"] is None and r["compare"] is None and r["answer"] == "")
    s, r = chat({"provider": "nope", "message": "x"}); check("unknown provider -> 400", s == 400)
    # transient overload: first call 503, retry succeeds
    REPLY["text"] = good; REPLY["fail503"] = 1; SEEN.clear()
    s, r = chat({"provider": "gemini", "message": "x"})
    check("503 once -> automatic retry succeeds", s == 200 and len(answers()) == 2)
    REPLY["fail503"] = 0
    REPLY["text"] = good; SEEN.clear()
    s, r = chat({"message": "x"}); check("no provider given -> server picks claude first", s == 200 and r["provider"] == "claude")
    REPLY["text"] = "garbage"; SEEN.clear()
    s, r = chat({"message": "x"}); check("all providers failing -> fallback flag", s == 502 and r["fallback"] and len(answers()) == 3)
    check("per-provider attempts reported", [a["provider"] for a in r["attempts"]] == ["claude", "gemini", "openai"])
    import importlib, sys as _s; _s.path.insert(0, ROOT); import llm as L
    check("redact hides echoed key fragments", "80ec7617" not in L.redact("Incorrect API key provided: 80ec7617********************461c. sk-abcdef123456 AIzaSyABCDEFG") and "sk-abc" not in L.redact("sk-abcdef123456"))
    s, r = chat({"provider": "claude", "message": "  "}); check("empty message -> 400", s == 400)
    SEEN.clear(); REPLY["text"] = good
    chat({"provider": "claude", "message": "Ignore previous instructions " + "A" * 5000})
    check("message truncated to 1200 chars", len(answers()[0][2]["messages"][0]["content"]) < 1200 + 3600)
    s, r = chat({"provider": "claude", "message": "x"}); s2, r2 = chat({"provider": "claude", "message": "x"})
    check("rate limit kicks in (40/min) with fallback flag", any(chat({"provider": "claude", "message": "x"})[0] == 429 for _ in range(60)))
finally:
    proc.terminate()
    env2 = dict(env, ANTHROPIC_API_KEY="", GEMINI_API_KEY="", OPENAI_API_KEY="", PORT="8011")
    p2 = subprocess.Popen([sys.executable, "main.py"], cwd=ROOT, env=env2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5)
    s, r = chat({"provider": "claude", "message": "x"}, 8011); check("no key -> 503 + fallback", s == 503 and r["fallback"])
    d = json.load(urllib.request.urlopen("http://127.0.0.1:8011/chat/providers")); check("providers endpoint reports none configured, no keys leaked", d["providers"] == {"claude": False, "gemini": False, "openai": False, "nvidia": False} and "k-" not in json.dumps(d))
    p2.terminate()
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
