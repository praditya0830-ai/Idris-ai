"""IDRIS FEA worker — CalculiX + gmsh behind one small HTTP service.

The worker is deployed to Google Cloud Run and called by the Idris solver
(`fea3d.py`). It is deliberately separate from the solver: CalculiX and gmsh are
GPL programs and are executed as subprocesses, never linked into IDRIS code
(see LICENSES.md).

Run it locally:
    FEA_WORKER_TOKEN=dev PORT=8080 python3 -m fea_worker.app
"""
