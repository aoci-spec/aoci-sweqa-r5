#!/usr/bin/env python3
"""Shared plumbing for the evaluation harness.

- JSONL records keyed by question (last valid record wins on resume)
- thread-safe append + a bounded thread pool (`run_parallel`) so every stage
  runs N requests at once (default 7, 1..20)
- run manifests (what exactly produced a run directory)
- the degenerate-answer guard applied symmetrically to every arm
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_WORKERS = 7
MAX_WORKERS = 100  # DeepSeek gateway probed 300/300 OK at 100-way (2026-08-20); per-pass useful max = question count


def clamp_workers(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = DEFAULT_WORKERS
    return max(1, min(MAX_WORKERS, n))


# ---------------------------------------------------------------- ids / hashes
def qid(question):
    """Stable short id for a question (SWE-QA has none)."""
    return hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:12]


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(path, short=True, untracked=True):
    """HEAD sha with a -dirty suffix when tracked files are modified (and, if
    untracked=True, when untracked files exist — target repos carry untracked
    index volumes by design, so callers pass untracked=False for them)."""
    try:
        cmd = ["git", "-C", path, "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        st = ["git", "-C", path, "status", "--porcelain"] + ([] if untracked else ["-uno"])
        dirty = subprocess.run(st, capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return None


HARNESS_PATHS = ("harness", "tests", "config/profiles.json", "config/prices.json", "specs", "requirements.txt", "Makefile")


def harness_sha():
    """Short HEAD sha with -dirty when the harness CODE (not run data) has uncommitted tracked changes."""
    try:
        sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "-uno", "--"] + list(HARNESS_PATHS),
                            capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if st else "")
    except Exception:  # noqa: BLE001
        return None


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- jsonl
_locks = {}
_locks_guard = threading.Lock()


def _lock_for(path):
    with _locks_guard:
        return _locks.setdefault(os.path.abspath(path), threading.Lock())


def append_jsonl(path, rec):
    """Thread-safe single-line append with flush."""
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _lock_for(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l:
                try:
                    out.append(json.loads(l))
                except json.JSONDecodeError:
                    continue  # a torn line from an interrupted run is skipped, not fatal
    return out


def valid_answer(rec):
    return bool(rec.get("answer")) and not rec.get("error")


def valid_score(rec):
    return bool(rec.get("scores"))


def load_records(path, valid=None):
    """question -> record. Later records replace earlier ones; a later invalid record
    never replaces an earlier valid one (resume appends repairs)."""
    valid = valid or (lambda r: valid_answer(r) or valid_score(r))
    by = {}
    for r in read_jsonl(path):
        q = r.get("question")
        if q is None:
            continue
        prev = by.get(q)
        if prev is None or valid(r) or not valid(prev):
            by[q] = r
    return by


def check_param_drift(path, current, keys=("params", "model", "profile", "kind"), allow=False):
    """Refuse to resume onto records produced with different parameters (a silent
    reproducibility trap). `current` is the dict of the values in effect now."""
    drift = {}
    for r in read_jsonl(path):
        if not valid_answer(r):
            continue
        for k in keys:
            if k in r and r.get(k) != current.get(k):
                drift[k] = (r.get(k), current.get(k))
        if drift:
            break
    if drift and not allow:
        sys.exit(f"refusing to resume {os.path.basename(path)}: existing records were produced with different settings "
                 f"{ {k: {'existing': v[0], 'now': v[1]} for k, v in drift.items()} }; delete the file or pass --allow-param-drift")
    return drift


class Budget:
    """Running USD estimate across threads; stop launching work when a cap is hit."""

    def __init__(self, cap_usd, price):
        self.cap, self.price, self.spent, self.lock = cap_usd, price, 0.0, threading.Lock()

    def add(self, usage):
        if not self.price or not usage:
            return
        c = ((usage.get("input") or 0) * self.price[0] + (usage.get("output") or 0) * self.price[1]) / 1e6
        with self.lock:
            self.spent += c

    def exceeded(self):
        return self.cap is not None and self.spent >= self.cap


def load_questions(path, limit=0):
    qs = [r for r in read_jsonl(path) if r.get("question")]
    return qs[:limit] if limit else qs


def question_categories(path):
    """Explicit category axis of a question bank: ({question_text: category}, axis tuple in bank order).

    Banks may carry a `category` field per question (e.g. the xingyun bank: change/why/where/how/what —
    Chinese question text defeats the first-word heuristic). Banks without the field, or an unreadable
    path, return ({}, ()) so callers fall back to the SWE-QA first-word heuristic unchanged."""
    if path and not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(ROOT, path)
    qmap, axis = {}, []
    try:
        for q in load_questions(path):
            c = q.get("category")
            if c:
                qmap[q["question"]] = c
                if c not in axis:
                    axis.append(c)
    except Exception:  # noqa: BLE001 — categories are display/analysis sugar, never fatal
        return {}, ()
    return qmap, tuple(axis)


# ---------------------------------------------------------------- degenerate answers
DEGENERATE_MIN_CHARS = 40
_DEGENERATE_TAIL = re.compile(r"(:|\*\*answer:?\*\*|answer:)\s*$", re.I)
# a final line that is only a heading / bold label / empty bullet: the body never came
_HEADING_LINE = re.compile(r"^(#{1,6}\s.*|\*\*[^*\n]+\*\*:?|[-*•]\s*|\d+[.)]\s*)$")
# a first line that only announces the answer ("I have enough context to answer.")
_LEAD_IN = re.compile(r"^(I|Now I|Based on|Okay|OK)\b.*?(enough|answer|context)[^\n]*$", re.I)


def degenerate_reason(answer):
    """Why an answer should be retried once, or None.

    Symmetric across arms: an empty answer, a very short one, one that stops on a
    lead-in ('...**Answer:**'), one whose last line is a bare heading/bold label,
    or one that is only an announcement ('I have enough context now.') is a
    harness/gateway artefact (thinking consumed the budget), not a model verdict.
    """
    a = (answer or "").strip()
    if not a:
        return "empty"
    if _DEGENERATE_TAIL.search(a):
        return "cut_off"
    lines = [l.strip() for l in a.splitlines() if l.strip()]
    if lines and _HEADING_LINE.match(lines[-1]):
        return "cut_off"
    body = lines[1:] if len(lines) > 1 and _LEAD_IN.match(lines[0]) else lines
    if len(" ".join(body)) < DEGENERATE_MIN_CHARS:
        return "too_short"
    return None


NUDGE = ("Your previous reply was empty or cut off before the answer. "
         "Give the complete final answer now.")

# A REFUSAL is not an answer: the model says the message carried no question / was garbled / looked like an
# injection it will not follow. Observed as an EPISODE — identical bytes, temperature 0: 0/48 (k=1), then 6/48
# in one 8-minute window (k=1b), then 0/48 (k=1c) — i.e. gateway/model-side, like a dropped request. Treated
# like one: the question is re-run in a FRESH conversation (bounded), and the count is recorded per record so
# refusal rate is a reportable, symmetric secondary outcome.
_REFUSAL = re.compile(
    r"\b(no|not any|without|doesn'?t (?:contain|include|have)|does not (?:contain|include|have)|isn'?t|is not)\s+"
    r"(?:a |an |any |the )?(?:clear |actual |answerable |concrete |specific |real )?(?:question|task|request)\b"
    r"|\bgarbled\b|\bwon'?t follow\b|\bprompt[- ]injection\b"
    r"|\b(?:don'?t|do not|didn'?t|did not|cannot|can'?t) (?:see|find|identify) (?:a |an |any |the )?(?:clear |actual |answerable |concrete |specific |real )?(?:question|task|request)\b"
    r"|\bready to help\b.{0,80}\b(?:question|task|request)\b"
    r"|\bwhat (?:you'?d|you would) like me to (?:do|help)\b"
    r"|\bplease (?:provide|state|restate|share|clarify|let me know)\b.{0,60}\b(?:question|task)\b", re.I | re.S)
MAX_REFUSAL_RERUNS = 2


_GARBLE = re.compile(r"<[|｜][^>\n]{0,60}[|｜]>|<｜|｜>")   # special-token debris, e.g. '<｜DSML｜calling…' (5/2160 in R1)


def refusal_reason(answer):
    """'refusal' / 'garbled' if the answer is not an answer, else None. Both re-run in a fresh conversation."""
    a = (answer or "").strip()
    if a and _GARBLE.search(a):
        return "garbled"
    a = a[:600]
    return "refusal" if a and _REFUSAL.search(a) else None


# ---------------------------------------------------------------- parallel driver
def progress_path_for(out_path, stage=None):
    """runs/<id>/progress-<stage or arm>.json for an answers-/scores- file (what the dashboard reads for live concurrency)."""
    d = os.path.dirname(os.path.abspath(out_path))
    b = os.path.basename(out_path)
    name = b[:-len(".jsonl")] if b.endswith(".jsonl") else b
    for pfx in ("answers-", "scores-"):
        if name.startswith(pfx):
            name = name[len(pfx):]
    return os.path.join(d, f"progress-{stage + '-' if stage else ''}{name}.json")


def _write_progress(path, state):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except OSError:
        pass


def run_parallel(items, fn, workers=DEFAULT_WORKERS, on_result=None, label="", progress_path=None):
    """Run fn(item) for every item on a bounded pool; call on_result(item, result, i, n)
    as each finishes (already serialized by a lock). Exceptions from fn are returned
    as {"error": ...} results so one bad question never kills a run.
    progress_path: if given, a small JSON (label, workers, total, done, in_flight, started_at, updated_at) is
    rewritten atomically at every start/finish — the dashboard's live-concurrency counter."""
    workers = clamp_workers(workers)
    n = len(items)
    done = 0
    lock = threading.Lock()
    results = [None] * n
    if n == 0:
        return results
    state = {"label": label, "workers": workers, "total": n, "done": 0, "in_flight": 0,
             "started_at": time.time(), "updated_at": time.time()}

    def bump(d_flight, d_done):
        with lock:
            state["in_flight"] += d_flight
            state["done"] += d_done
            state["updated_at"] = time.time()
            if progress_path:
                _write_progress(progress_path, dict(state))

    def wrapped(it):
        bump(+1, 0)
        try:
            return fn(it)
        finally:
            bump(-1, 0)

    if progress_path:
        _write_progress(progress_path, dict(state))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=label or "w") as ex:
        futs = {ex.submit(wrapped, it): i for i, it in enumerate(items)}
        try:
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    res = {"error": f"{e.__class__.__name__}: {str(e)[:300]}"}
                results[i] = res
                bump(0, +1)
                with lock:
                    done += 1
                    if on_result:
                        on_result(items[i], res, done, n)
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
    return results


# ---------------------------------------------------------------- manifest
def deep_merge(dst, src):
    """Recursive dict merge: later writes add keys instead of replacing whole sub-dicts."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def write_manifest(run_dir, extra):
    """Merge `extra` into <run_dir>/manifest.json (created on first call), recursively:
    a second write for the same arm must not erase what the first one recorded.
    Guarded by an fcntl lock: with concurrent_passes > 1 two ADAPTER PROCESSES write the
    same manifest, and an unlocked read-modify-write loses one arm's record."""
    import fcntl
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "manifest.json")
    lockp = path + ".lock"
    with open(lockp, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            return _write_manifest_locked(path, extra)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _write_manifest_locked(path, extra):
    run_dir = os.path.dirname(path)
    m = {}
    if os.path.exists(path):
        try:
            m = json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            m = {}
    m.setdefault("created", now_iso())
    m.setdefault("harness_sha", harness_sha())
    m["updated"] = now_iso()
    # every pass that touches this run leaves a line: a resumed or repaired run is never one SHA
    prov = m.setdefault("provenance", [])
    stamp = {"sha": harness_sha(), "at": m["updated"]}
    if not prov or prov[-1].get("sha") != stamp["sha"]:
        prov.append(stamp)
    deep_merge(m, extra)
    os.makedirs(run_dir, exist_ok=True)
    tmp = path + ".tmp"
    with _lock_for(path):
        json.dump(m, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    return m


def read_manifest(run_dir):
    path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def index_fingerprint(repo_root):
    """Entry count + sha256 of the volumes actually injected/served."""
    vols = {}
    for name in ("aoci.txt", "aoci.meta.txt", "aoci.code.txt"):
        p = os.path.join(repo_root, name)
        if os.path.exists(p):
            vols[name] = {"sha256": sha256_file(p)[:16], "bytes": os.path.getsize(p)}
    code = os.path.join(repo_root, "aoci.code.txt")
    entries = 0
    if os.path.exists(code):
        # entry lines look like `name[CG8L]: F:... | R:... | A:... | S:...`
        entries = sum(1 for l in open(code, encoding="utf-8") if re.match(r"^\S+\[[A-Z]{1,3}\d[A-Z]?\]: ", l))
    return {"repo_commit": git_sha(repo_root, short=False, untracked=False), "entries": entries, "volumes": vols}


def load_artifact(path):
    """Read a built index artefact (build_index_artifact.py) and its sidecar; refuse if they disagree.

    Returns (text, fingerprint). The fingerprint goes into the manifest and every record, so
    'delivered chars == artefact chars' is an assertion analysis can make, and two arms (or two hosts)
    that claim to have injected 'the index' can be shown to have injected the same bytes.
    """
    side = path[:-4] + ".json" if path.endswith(".txt") else path + ".json"
    if not os.path.exists(path) or not os.path.exists(side):
        sys.exit(f"index artefact or sidecar missing: {path} / {side} — run harness/build_index_artifact.py")
    text = open(path, encoding="utf-8").read()
    meta = json.load(open(side, encoding="utf-8"))
    if sha256_text(text) != meta.get("artifact_sha256"):
        sys.exit(f"index artefact {path} does not match its sidecar sha256 — rebuild it")
    fp = {"artifact": os.path.relpath(path, ROOT), "artifact_sha256": meta["artifact_sha256"], "chars": meta["chars"],
          "entries": meta["entries"], "volumes": meta["volumes"], "repo_commit": meta.get("repo_commit"),
          "built_at": meta.get("built_at")}
    return text, fp


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)
