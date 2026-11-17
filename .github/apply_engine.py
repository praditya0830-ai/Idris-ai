"""Apply the engineering-bench wiring to the source files that already exist on main.

Every edit is an exact, unique replacement and the result is checked against the blob hashes of the
files that passed locally, so a mismatch fails the job before anything is committed. Workflow files are
not touched here: the App token cannot push them.
"""
import subprocess
import sys

EXPECT = {
    "engine_routes.py": "e5dd51129b580085579401775e9db0ba625af680",
    "main.py": "1db1c951ae1863924ff7a2e4c1a5e2523aeae341",
    "llm.py": "811972cab2263967ecf514605f19e61fb6ec3ff1",
}

EDITS = {
    "engine_routes.py": [
        (
            'GET_PATHS = ("/geometry/checks", "/geometry/kernel", "/geometry/selftest")\n'
            'POST_PATHS = ("/model/validate", "/model/build", "/assembly/manifest", "/assembly/node",\n'
            '              "/exchange/step")\n',
            'GET_PATHS = ("/geometry/checks", "/geometry/kernel", "/geometry/selftest", "/engineer/materials")\n'
            'POST_PATHS = ("/model/validate", "/model/build", "/assembly/manifest", "/assembly/node",\n'
            '              "/exchange/step", "/engineer/solve", "/engineer/verify")\n',
        ),
        (
            '    if p == "/geometry/selftest":\n'
            '        return _selftest()\n'
            '    return None\n',
            '    if p == "/geometry/selftest":\n'
            '        return _selftest()\n'
            '    if p == "/engineer/materials":\n'
            '        from ai import engineer as ENG\n'
            '        return 200, {"success": True, "materials": ENG.MATERIALS,\n'
            '                     "note": "typical handbook values; a real part needs the mill certificate or the "\n'
            '                             "standard\'s guaranteed minimum for its grade"}\n'
            '    return None\n',
        ),
        (
            '    if base == "/model/validate":\n',
            '    if base in ("/engineer/solve", "/engineer/verify"):\n'
            '        from ai import engineer as ENG\n'
            '        if base == "/engineer/verify":\n'
            '            eng = params.get("engineering") if isinstance(params.get("engineering"), dict) else None\n'
            '            return 200, {"success": True, **ENG.verify(str(params.get("answer") or ""), eng)}\n'
            '        given = params.get("given") if isinstance(params.get("given"), dict) else None\n'
            '        eng = ENG.solve(str(params.get("message") or ""), given)\n'
            '        if params.get("answer"):\n'
            '            eng["verification"] = ENG.verify(str(params["answer"]), eng)\n'
            '        return 200, {"success": True, **eng}\n'
            '\n'
            '    if base == "/model/validate":\n',
        ),
    ],
    "llm.py": [
        (
            "def want_english(reply_language) -> bool:\n",
            'def _engineering_block(context) -> str:\n'
            '    """Evidence computed by ai/engineer.py. When it is present the answer may only quote its numbers."""\n'
            '    if isinstance(context, dict) and isinstance(context.get("engineering"), str) and context["engineering"].strip():\n'
            '        return context["engineering"]\n'
            '    return ""\n'
            '\n'
            '\n'
            "def want_english(reply_language) -> bool:\n",
        ),
        (
            '    user = build_user_block(message, current, variants, selected) + (EN_RULE if want_english(reply_language) else "")\n'
            "    attempts, statuses = [], []\n",
            '    eng = _engineering_block(context)\n'
            '    user = ((eng + "\\n\\n") if eng else "") + build_user_block(message, current, variants, selected) \\\n'
            '        + (EN_RULE if want_english(reply_language) else "")\n'
            "    attempts, statuses = [], []\n",
        ),
        (
            '    return ("CONTEXT (data, from the engine):\\n" + head +\n'
            '            ("\\nVARIANTS (data):\\n" + vs[:room] if room > 200 and variants else "") +\n'
            '            "\\n\\nUSER MESSAGE (data):\\n" + message)\n',
            '    eng = _engineering_block(context)\n'
            '    return ((eng + "\\n\\n") if eng else "") + ("CONTEXT (data, from the engine):\\n" + head +\n'
            '                                               ("\\nVARIANTS (data):\\n" + vs[:room] if room > 200 and variants else "") +\n'
            '                                               "\\n\\nUSER MESSAGE (data):\\n" + message)\n',
        ),
    ],
    "main.py": [
        (
            '                       "/exchange/step")\n',
            '                       "/exchange/step", "/engineer/solve", "/engineer/verify")\n',
        ),
        (
            '                msg = str(params.get("message") or "")[:llm.MAX_MSG]\n',
            '                msg = str(params.get("message") or "")[:llm.MAX_MSG]\n'
            "                # the bench answers what can be computed; the language layer may only quote those numbers\n"
            "                eng, eng_verify = None, None\n"
            "                try:\n"
            "                    from ai import engineer as _ENG\n"
            '                    given = ctx.get("given") if isinstance(ctx, dict) and isinstance(ctx.get("given"), dict) else None\n'
            "                    eng = _ENG.solve(msg, given)\n"
            "                    eng_verify = _ENG.verify\n"
            '                    if eng.get("claims") or eng.get("missing"):\n'
            "                        ctx = dict(ctx or {})\n"
            '                        ctx["engineering"] = _ENG.chat_block(eng)\n'
            "                except Exception:\n"
            "                    traceback.print_exc()\n"
            "                    eng, eng_verify = None, None\n",
        ),
        (
            '                res = llm.interpret(params.get("provider"), params.get("message"), params.get("current"),\n'
            '                                    params.get("variants"), params.get("selected"),\n'
            '                                    params.get("context") if isinstance(params.get("context"), dict) else None, rl)\n',
            '                res = llm.interpret(params.get("provider"), params.get("message"), params.get("current"),\n'
            '                                    params.get("variants"), params.get("selected"), ctx, rl)\n'
            "                if eng and eng_verify:\n"
            '                    res["engineering"] = {"topic": eng.get("topic"), "claims": eng.get("claims"),\n'
            '                                          "missing": eng.get("missing"), "notes": eng.get("notes")}\n'
            '                    res["unverified"] = eng_verify(res.get("answer") or "", eng)\n',
        ),
    ],
}


def main() -> int:
    for path, edits in EDITS.items():
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for old, new in edits:
            n = text.count(old)
            if n != 1:
                print(f"FAIL {path}: anchor found {n} times, expected 1: {old[:70]!r}")
                return 1
            text = text.replace(old, new)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    bad = []
    for path, want in EXPECT.items():
        got = subprocess.run(["git", "hash-object", path], capture_output=True, text=True).stdout.strip()
        print(f"{path}: {got} {'ok' if got == want else 'MISMATCH (want ' + want + ')'}")
        if got != want:
            bad.append(path)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
