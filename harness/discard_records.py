#!/usr/bin/env python3
"""Discard specific answer records from a run so the resumable runner re-answers them (fresh draw).

The acceptance gate (accept.py) blocks judging while a pass carries a hard failure such as a record whose agent
modified its workspace (`workspace_dirty` with real changes). The protocol remedy is "fix or discard the offending
records, then re-run": this tool removes exactly the named records, keeps a byte-identical backup of the file, and
appends an audit line to <run>/discards.jsonl so the result card can disclose every discard. It never edits scores.

  python3 harness/discard_records.py runs/<id> --pass oh-aoci-3 --dirty          # every workspace_dirty record of that pass
  python3 harness/discard_records.py runs/<id> --pass oh-aoci-3 --index 17       # the 17th record (0-based) of that pass
  python3 harness/discard_records.py runs/<id> --pass oh-aoci-3 --error          # every record carrying an error
"""
import argparse
import json
import os
import shutil
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pass", dest="pass_name", required=True)
    ap.add_argument("--index", type=int, action="append", default=[])
    ap.add_argument("--dirty", action="store_true", help="discard records with workspace_dirty=true")
    ap.add_argument("--error", action="store_true", help="discard records with a non-empty error")
    ap.add_argument("--reason", default="acceptance gate hard check (workspace modified): discarded and re-answered")
    a = ap.parse_args()
    path = os.path.join(a.run_dir, f"answers-{a.pass_name}.jsonl")
    lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
    keep, dropped = [], []
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            keep.append(line)
            continue
        hit = (i in a.index) or (a.dirty and rec.get("workspace_dirty")) or (a.error and rec.get("error"))
        if hit:
            dropped.append({"index": i, "question": rec.get("question"), "qid": rec.get("qid"), "workspace_dirty": rec.get("workspace_dirty"),
                            "dirty_paths": rec.get("dirty_paths"), "error": rec.get("error"), "forced": rec.get("forced")})
        else:
            keep.append(line)
    if not dropped:
        print("nothing matched; no change")
        return
    ts = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy(path, f"{path}.discarded-{ts}")
    open(path, "w", encoding="utf-8").write("".join(keep))
    with open(os.path.join(a.run_dir, "discards.jsonl"), "a", encoding="utf-8") as f:
        for d in dropped:
            f.write(json.dumps({"ts": ts, "pass": a.pass_name, "reason": a.reason, **d}, ensure_ascii=False) + "\n")
    print(f"discarded {len(dropped)} record(s) from {path} (backup {path}.discarded-{ts}); re-run the spec to re-answer them")
    for d in dropped:
        print("  ", d["index"], (d["question"] or "")[:80], d["dirty_paths"] or d["error"])


if __name__ == "__main__":
    main()
