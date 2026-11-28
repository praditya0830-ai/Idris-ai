"""What this build can actually run, and whether its benchmarks passed.

Nothing in this file is hardcoded to `true`. A solver is `available` only when
its module exists in `fea_worker/solvers/` and its binaries are on PATH, and it is
`validated` only when that module's benchmark gate ran in this container and
passed. A worker built without CalculiX or gmsh therefore reports both as false
instead of pretending to solve.
"""
from __future__ import annotations

import importlib
import shutil
import subprocess

from .config import Config

VERSION = "0.1.0"

# Analysis names the Studio asks about, in the order it shows them.
ANALYSES = ("static_linear", "thermal_steady", "modal", "topology", "drop_test", "cfd_3d")

# Analyses that this worker will never do, whatever gets built here.
_NEVER = {
    "cfd_3d": "not CalculiX; needs OpenFOAM",
}
_NOT_BUILT = "not built yet"


def tool_version(exe, flag):
    """First line of `<exe> <flag>`, or None when the program is not installed."""
    path = shutil.which(exe)
    if not path:
        return None
    try:
        p = subprocess.run([path, flag], capture_output=True, text=True, timeout=25)
    except Exception:
        return None
    out = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    for line in out.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def binaries(cfg=None):
    cfg = cfg or Config()
    return {
        "ccx": shutil.which("ccx"),
        "gmsh": shutil.which("gmsh"),
    }


def solver_modules():
    """name -> module for every solver implemented in this build."""
    found = {}
    for name in ANALYSES:
        try:
            found[name] = importlib.import_module(f".solvers.{name}", __package__)
        except ModuleNotFoundError:
            continue
    return found


def build(cfg=None):
    cfg = cfg or Config()
    mods = solver_modules()
    have = binaries(cfg)
    analyses = {}
    for name in ANALYSES:
        if name in _NEVER:
            analyses[name] = {"available": False, "reason": _NEVER[name]}
            continue
        mod = mods.get(name)
        if mod is None:
            analyses[name] = {"available": False, "validated": False, "benchmarks": [],
                              "reason": _NOT_BUILT}
            continue
        try:
            probe = mod.probe(cfg)
        except Exception as ex:                      # a broken solver must not take the worker down
            analyses[name] = {"available": False, "validated": False, "benchmarks": [],
                              "reason": f"{type(ex).__name__}: {ex}"}
            continue
        analyses[name] = {
            "available": bool(probe.get("available")),
            "validated": bool(probe.get("validated")),
            "benchmarks": probe.get("benchmarks") or [],
            "reason": probe.get("reason") or "",
        }
    return {
        "worker": "idris-fea",
        "version": VERSION,
        "ccx_version": tool_version("ccx", "-v"),
        "gmsh_version": tool_version("gmsh", "--version"),
        "binaries": {k: bool(v) for k, v in have.items()},
        "analyses": analyses,
        "limits": cfg.describe_limits(),
    }
