"""Background job runner for slow analyses (FEA, later modal/CFD).

One worker thread (Render free tier has little CPU/RAM, so jobs run one at a time);
others wait in the queue. Finished jobs are kept for JOB_TTL_S, then forgotten.
"""
import queue
import threading
import time
import traceback
import uuid

JOB_TTL_S = 3600
MAX_QUEUE = 8
_jobs = {}
_q = queue.Queue()
_lock = threading.Lock()
_worker = None


def _set(job_id, **kw):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kw, updated=time.time())


def _loop():
    while True:
        job_id, fn = _q.get()
        _set(job_id, status="running", stage="Meshing", progress=0.0)
        t0 = time.time()
        try:
            def progress(stage, frac=None):
                _set(job_id, stage=stage, progress=None if frac is None else round(float(frac), 3))
            result, error = fn(progress)
            if error:
                _set(job_id, status="failed", error=error, elapsed_s=round(time.time() - t0, 2))
            else:
                _set(job_id, status="done", stage="Post-processing", progress=1.0, result=result,
                     elapsed_s=round(time.time() - t0, 2))
        except Exception as ex:
            traceback.print_exc()
            _set(job_id, status="failed", error=f"Solver crashed: {type(ex).__name__}: {ex}")
        finally:
            _q.task_done()


def _cleanup():
    now = time.time()
    with _lock:
        for k in [k for k, j in _jobs.items()
                  if j["status"] in ("done", "failed") and now - j["updated"] > JOB_TTL_S]:
            del _jobs[k]


def submit(kind, fn):
    """fn(progress) -> (result_dict, error_message_or_None). Returns job_id or None if queue is full."""
    global _worker
    _cleanup()
    with _lock:
        waiting = sum(1 for j in _jobs.values() if j["status"] == "queued")
        if waiting >= MAX_QUEUE:
            return None
        job_id = uuid.uuid4().hex[:12]
        _jobs[job_id] = {"job_id": job_id, "kind": kind, "status": "queued", "stage": "Queued",
                         "progress": 0.0, "created": time.time(), "updated": time.time()}
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, daemon=True, name="ace-jobs")
            _worker.start()
    _q.put((job_id, fn))
    return job_id


def status(job_id):
    with _lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None
