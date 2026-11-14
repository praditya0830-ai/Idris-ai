"""Every POST route that main.py handles must be in its allowlist. v0.25 to v0.31.2 shipped /cad/review and
/catalog/bolt-check behind an allowlist that did not name them: both answered 404 'unknown endpoint' on live."""
import os
import re


def test_every_handled_post_route_is_allowed():
    src = open(os.path.join(os.path.dirname(__file__), "..", "main.py"), encoding="utf-8").read()
    post = src[src.index("def do_POST"):]
    allowed = set(re.findall(r'"(/[a-z\-/]+)"', post[post.index("POST_ROUTES = ("):post.index(")", post.index("POST_ROUTES = ("))]))
    handled = set(re.findall(r'if base == "(/[a-z\-/]+)":', post))
    assert handled and handled <= allowed, sorted(handled - allowed)
    assert {"/cad/review", "/catalog/bolt-check", "/chat"} <= allowed
