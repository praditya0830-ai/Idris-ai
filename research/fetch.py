"""Fetch one page and reduce it to text the extractor can quote from."""
from __future__ import annotations

import html
import re
import urllib.request
from html.parser import HTMLParser

MAX_BYTES = 600_000          # a datasheet fits; a runaway download does not
MAX_CHARS = 20_000


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def fetch(url: str, timeout: float = 15.0, max_chars: int = MAX_CHARS) -> str:
    """-> visible text, whitespace collapsed. Empty string for anything that is not plain http(s) or fails."""
    if not re.match(r"https?://", url or "", re.I):
        return ""
    req = urllib.request.Request(url, headers={
        "User-Agent": "IdrisResearch/1.0 (engineering dimension lookup; contact: idris-core)",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(MAX_BYTES)
        charset = r.headers.get_content_charset() or "utf-8"
    text = raw.decode(charset, "replace")
    if "<" in text[:4000]:
        p = _Text()
        p.feed(text)
        text = " ".join(p.parts)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()[:max_chars]
