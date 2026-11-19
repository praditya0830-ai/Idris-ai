"""One counter for every paid AI call (review of PR #5-#8, item 7).

Why this exists: `llm.check_rate` counts HTTP *requests*, one per /chat call. One request can spend many AI
calls — a design job runs one per round (up to `IDRIS_DESIGN_ROUNDS`), an assembly job one per part — so the
cap on requests never bounded the money. This module counts the calls themselves, at the single door every
provider call goes through (`llm._post`), and it is the one place that says "no more AI today".

Limits (environment):
    IDRIS_AI_CALLS_DAILY_CAP     paid AI calls per UTC day, all routes and background jobs together (300)
    IDRIS_AI_MONTHLY_BUDGET_USD  estimated spend per UTC month (20 USD)

Prices are list prices per million tokens, read from each provider's public pricing page on 2026-11-19. The
model ids this deployment defaults to are newer than some of those pages, so every row is the price of the
same model tier and is marked assumed: the number is a budget guard, never a bill. `usage()` says so, and a
provider that is not in the table is counted with no price (its tokens still count, its cost does not).

The counters live in memory (the server is one process), roll over at 00:00 UTC, and a restart starts a new
day. Nothing here stores or returns a key, a prompt or a user: only counts, tokens and money.
"""
from __future__ import annotations

import os
import threading
import time

CALL_CAP_ENV = "IDRIS_AI_CALLS_DAILY_CAP"
BUDGET_ENV = "IDRIS_AI_MONTHLY_BUDGET_USD"
DEFAULT_CALLS_DAILY = 300
DEFAULT_MONTHLY_USD = 20.0

# USD per million tokens: (input, output). Source: the public pricing page of each provider, read 2026-11-19 —
# Anthropic (Haiku tier), Google AI (Flash tier), OpenAI (mini tier), DeepSeek, Alibaba Model Studio (Qwen),
# Moonshot (Kimi). NVIDIA build.nvidia.com is a free API, hence zero. `ASSUMED` lists every row whose model id
# is not the exact id on that page: the tier price is used instead of guessing a number.
PRICES = {
    "claude": (1.00, 5.00),
    "gemini": (0.30, 2.50),
    "openai": (0.15, 0.60),
    "deepseek": (0.28, 0.42),
    "qwen": (0.40, 1.20),
    "kimi": (0.60, 2.50),
    "nvidia": (0.00, 0.00),
}
ASSUMED = tuple(sorted(PRICES))
PRICE_NOTE = ("USD per 1M tokens (input, output) from each provider's public pricing page, read 2026-11-19. "
              "Every row is the price of the same model tier, not of the exact deployment model id, so the "
              "cost is an estimate for a budget guard and not a bill.")

_LOCK = threading.Lock()
_S = {"day": "", "month": "", "calls": 0, "calls_month": 0, "input": 0, "output": 0,
      "cost_day": 0.0, "cost_month": 0.0, "unpriced": []}


def _num(name: str, default: float) -> float:
    """A limit from the environment: a typo falls back to the default instead of stopping the server."""
    raw = os.environ.get(name)
    try:
        return float(raw) if raw not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)


def _roll(now: float) -> None:
    """Start a new day/month (UTC). The caller holds the lock."""
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    month = day[:7]
    if _S["day"] != day:
        _S.update(day=day, calls=0, input=0, output=0, cost_day=0.0, unpriced=[])
    if _S["month"] != month:
        _S.update(month=month, calls_month=0, cost_month=0.0)


def provider_of(url: str) -> str:
    """Which provider a call went to. The configured base URL wins over the default one."""
    u = (url or "").lower()
    for env_name, prov in (("ANTHROPIC_BASE_URL", "claude"), ("GEMINI_BASE_URL", "gemini"),
                           ("OPENAI_BASE_URL", "openai"), ("DEEPSEEK_BASE_URL", "deepseek"),
                           ("DASHSCOPE_BASE_URL", "qwen"), ("MOONSHOT_BASE_URL", "kimi"),
                           ("NVIDIA_BASE_URL", "nvidia")):
        base = (os.environ.get(env_name) or "").lower().rstrip("/")
        if base and u.startswith(base):
            return prov
    for needle, prov in (("anthropic", "claude"), ("generativelanguage", "gemini"), ("openai.com", "openai"),
                         ("deepseek", "deepseek"), ("dashscope", "qwen"), ("aliyuncs", "qwen"),
                         ("moonshot", "kimi"), ("nvidia", "nvidia")):
        if needle in u:
            return prov
    return "unknown"


def tokens(response) -> tuple:
    """(input, output) tokens the provider itself reported, or (0, 0) when the reply does not carry them."""
    if not isinstance(response, dict):
        return 0, 0
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    pin, pout = usage.get("prompt_tokens", usage.get("input_tokens")), usage.get("completion_tokens",
                                                                              usage.get("output_tokens"))
    if pin is None and pout is None:                     # Gemini reports a different shape
        meta = response.get("usageMetadata") if isinstance(response.get("usageMetadata"), dict) else {}
        pin, pout = meta.get("promptTokenCount"), meta.get("candidatesTokenCount")

    def as_int(x) -> int:
        try:
            return max(0, int(x))
        except (TypeError, ValueError):
            return 0

    return as_int(pin), as_int(pout)


def gate(now: float | None = None) -> str | None:
    """The reason no more AI may be called right now, or None. The caller turns it into a clear message."""
    now = now if now is not None else time.time()
    with _LOCK:
        _roll(now)
        calls, cost, unpriced = _S["calls"], _S["cost_month"], list(_S["unpriced"])
    cap, budget = _num(CALL_CAP_ENV, DEFAULT_CALLS_DAILY), _num(BUDGET_ENV, DEFAULT_MONTHLY_USD)
    if calls >= cap:
        return (f"Daily AI budget reached ({calls:g} of {cap:g} AI calls today, {CALL_CAP_ENV}). IDRIS makes no "
                "more AI calls today: /chat answers with the built-in parser and the Bagian F geometry ladder, "
                "and jobs stop. The count resets at 00:00 UTC.")
    if cost >= budget:
        extra = (" Some providers have no price in this build, so their cost is not in this number."
                 if unpriced else "")
        return (f"Monthly AI budget reached (about ${cost:.2f} of ${budget:.2f}, {BUDGET_ENV}). IDRIS makes no "
                f"more AI calls this month. The estimate comes from the token counts the providers report."
                + extra)
    return None


def record(url: str, body: dict | None = None, response=None) -> None:
    """Count one call that was really made (a failed call is still a call), with its tokens when reported."""
    prov = provider_of(url)
    pin, pout = tokens(response)
    price = PRICES.get(prov)
    cost = (pin * price[0] + pout * price[1]) / 1e6 if price else 0.0
    with _LOCK:
        _roll(time.time())
        _S["calls"] += 1
        _S["calls_month"] += 1
        _S["input"] += pin
        _S["output"] += pout
        _S["cost_day"] += cost
        _S["cost_month"] += cost
        if price is None and prov not in _S["unpriced"]:
            _S["unpriced"].append(prov)


def usage(now: float | None = None) -> dict:
    """GET /ai/usage: counts, tokens, the estimate and what is left. No key, no prompt, no user data."""
    now = now if now is not None else time.time()
    with _LOCK:
        _roll(now)
        snap = {k: (list(v) if isinstance(v, list) else v) for k, v in _S.items()}
    cap, budget = _num(CALL_CAP_ENV, DEFAULT_CALLS_DAILY), _num(BUDGET_ENV, DEFAULT_MONTHLY_USD)
    blocked = gate(now)
    return {"date_utc": snap["day"], "month_utc": snap["month"],
            "calls_today": snap["calls"], "calls_today_cap": cap,
            "calls_today_left": max(0, int(cap) - snap["calls"]),
            "calls_month": snap["calls_month"],
            "tokens_today": {"input": snap["input"], "output": snap["output"]},
            "estimated_cost_usd_today": round(snap["cost_day"], 4),
            "estimated_cost_usd_month": round(snap["cost_month"], 4),
            "monthly_budget_usd": budget,
            "monthly_budget_left_usd": round(max(0.0, budget - snap["cost_month"]), 4),
            "blocked": blocked, "blocked_reason": blocked,
            "price_source": PRICE_NOTE, "price_assumed": list(ASSUMED),
            "unpriced_providers": snap["unpriced"],
            "note": "Counted in memory and reset at 00:00 UTC (day) and on the 1st (month); a restart starts a "
                    "new count. Every paid provider call goes through this counter, including calls made by "
                    "background jobs."}


def reset() -> None:
    """For a test (or an operator after a budget change): start counting from zero again."""
    with _LOCK:
        _S.update(day="", month="", calls=0, calls_month=0, input=0, output=0, cost_day=0.0, cost_month=0.0,
                  unpriced=[])
