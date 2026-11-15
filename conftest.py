"""Pytest configuration.

`tests/test_chat.py` and `tests/test_validation.py` are standalone QA scripts, not
pytest modules: they execute at import time, hit live services, and call `sys.exit()`.
`tests/test_waitlist.py` belongs to the Idris-company-web repo and imports `waitlist`,
which this backend does not ship. Keep all three out of collection so `pytest` works;
they still run directly with `python3 tests/<file>.py`.
"""

collect_ignore = [
    "tests/test_chat.py",
    "tests/test_validation.py",
    "tests/test_waitlist.py",
]
