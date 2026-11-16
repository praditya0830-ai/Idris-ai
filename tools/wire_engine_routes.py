"""One-shot, idempotent: forward the engine routes from main.py to engine_routes.py.

Why a script instead of a hand edit: main.py is 53 KB and a single character of drift in a copy would
break the server. This applies three exact anchors, refuses to write if an anchor is missing, and
syntax-checks the result before saving. Running it twice changes nothing.
"""
from __future__ import annotations

import pathlib
import sys

MAIN = pathlib.Path(__file__).resolve().parents[1] / "main.py"

# 1. GET: hand /geometry/* to engine_routes before the ACE fallback.
GET_OLD = """        if self._ace_unavailable(self.path):
            return
        if ace_get is not None:"""
GET_NEW = """        if self.path.split("?")[0].startswith("/geometry/"):
            import engine_routes          # geometry contract, kernel status, engine selftest
            res = engine_routes.handle_get(self.path)
            if res is not None:
                self._send_json(*res)
                return

        if self._ace_unavailable(self.path):
            return
        if ace_get is not None:"""

# 2. POST: the allowlist must name the new routes (tests/test_routes.py enforces this).
POST_OLD = '                       "/catalog/bolt-check", "/cad/review", "/cad/explore", "/cad/rank")'
POST_NEW = ('                       "/catalog/bolt-check", "/cad/review", "/cad/explore", "/cad/rank",\n'
            '                       "/model/validate", "/assembly/manifest", "/exchange/step")')

# 3. POST: dispatch before the chat handler.
DISPATCH_OLD = """        if base == "/chat":
            try:"""
DISPATCH_NEW = """        if base.startswith(("/model/", "/assembly/", "/exchange/")):
            import engine_routes          # feature graph, assembly manifest, STEP in
            res = engine_routes.handle_post(base, params)
            if res is None:
                self._send_json(404, {"success": False, "error": "unknown endpoint"})
            else:
                self._send_json(*res)
            return

        if base == "/chat":
            try:"""

HUNKS = [(GET_OLD, GET_NEW), (POST_OLD, POST_NEW), (DISPATCH_OLD, DISPATCH_NEW)]
MARKERS = ('startswith("/geometry/")', '"/exchange/step"',
           'startswith(("/model/", "/assembly/", "/exchange/"))')


def patch(src: str) -> str:
    out = src
    for old, new in HUNKS:
        if new.strip() and new in out:          # already applied: nothing to do
            continue
        if old not in out:
            raise SystemExit(f"anchor not found, refusing to write: {old.splitlines()[0]!r}")
        out = out.replace(old, new, 1)
    return out


def main() -> int:
    src = MAIN.read_text(encoding="utf-8")
    out = patch(src)
    missing = [m for m in MARKERS if m not in out]
    if missing:
        raise SystemExit(f"patch did not land: {missing}")
    compile(out, str(MAIN), "exec")             # a patched server that cannot parse is worse than none
    if out == src:
        print("already wired; nothing changed")
        return 0
    MAIN.write_text(out, encoding="utf-8")
    print(f"wired engine routes into {MAIN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
