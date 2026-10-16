#!/usr/bin/env python3
"""Apply pre-authored AOCI entries to a governed repository through the product's own MCP write path.

The model (S3 pipeline, any profile) authors one entry line per file into an authoring clone's
aoci.code.txt. This driver is only the host harness: it drives `aoci mcp` over stdio exactly like a
coding-agent host would — aoci_maintain → author_complete_current_machine_batch → aoci_update_entry
(complete batch, code_batch_id + candidate_id + source_sha256 bound) — until remaining=0, then verify.
No entry text is derived from paths or templates here; a missing entry is an error, never a stub.

  python3 harness/aoci_apply_entries.py --repo <governed repo> --entries <authoring clone>/aoci.code.txt \
      --authoring-root <authoring clone> [--aoci-bin build/aoci] [--max-batches 400]

repair_required: the response names the failing candidates and fields (entry_field_budget_exceeded with
actual/max tokens; rc20 counts ≈ 3 chars per token). The named field is shortened at a word boundary to fit
and the SAME complete batch is resubmitted (AGENTS.md rule 8). Anything else stops the run and is printed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ENTRY_RE = re.compile(r"^(?P<name>[^\[\]]+)\[(?P<tag>[^\]]+)\]: F:(?P<F>.*?) \| R:(?P<R>.*?) \| A:(?P<A>.*?) \| S:(?P<S>.*)$")


def load_entries(volume_path, authoring_root):
    """Map repo-relative path → complete single-line entry, from an S3 code volume with absolute section headers."""
    out, sec = {}, None
    root = os.path.realpath(authoring_root)
    for line in open(volume_path, encoding="utf-8"):
        line = line.rstrip("\n")
        m = re.match(r"^===(.*)===\s*$", line)
        if m:
            sec = os.path.realpath(m.group(1)) if m.group(1).startswith("/") else os.path.realpath(os.path.join(root, m.group(1)))
            continue
        if not line or line.startswith("#"):
            continue
        em = ENTRY_RE.match(line)
        if not em:
            continue
        rel = os.path.relpath(os.path.join(sec or root, em.group("name")), root)
        out[rel] = line
    return out


class Mcp:
    def __init__(self, aoci_bin, repo):
        self.p = subprocess.Popen([aoci_bin, "--repo", repo, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        self.n = 0
        self.rpc("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                "clientInfo": {"name": "aoci_apply_entries", "version": "1"}})
        self.rpc("notifications/initialized", notify=True)

    def rpc(self, method, params=None, notify=False):
        self.n += 1
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            msg["id"] = self.n
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        if notify:
            return None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("aoci mcp exited: " + self.p.stderr.read()[:400])
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if d.get("id") == self.n:
                return d

    def call(self, name, args):
        r = self.rpc("tools/call", {"name": name, "arguments": args})
        res = r.get("result") or r.get("error") or {}
        txt = "".join(c.get("text", "") for c in (res.get("content") or [])) if isinstance(res, dict) else json.dumps(res)
        try:
            return json.loads(txt), txt
        except Exception:  # noqa: BLE001
            return {"_raw": txt}, txt

    def close(self):
        try:
            self.p.terminate()
        except Exception:  # noqa: BLE001
            pass


def shorten(text, max_chars):
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sp = cut.rfind(" ")
    if sp > max_chars // 2:
        cut = cut[:sp]
    return cut.rstrip(" ,;:")


def repair(entry_line, field, max_tokens):
    """Shorten one FRAS field of a complete entry line to fit max_tokens (≈3 chars/token, 8% margin)."""
    m = ENTRY_RE.match(entry_line)
    if not m:
        return entry_line
    parts = {k: m.group(k) for k in ("name", "tag", "F", "R", "A", "S")}
    budget = int(max_tokens * 3 * 0.92)
    if field == "R":
        refs = [x.strip() for x in parts["R"].split(",") if x.strip()]
        while refs and len(", ".join(refs)) > budget:
            refs.pop()
        parts["R"] = ", ".join(refs) or "-"
    else:
        parts[field] = shorten(parts[field], budget) or "-"
    return f"{parts['name']}[{parts['tag']}]: F:{parts['F']} | R:{parts['R']} | A:{parts['A']} | S:{parts['S']}"


def find_budget_failures(obj, acc):
    """Collect {path, field, max_tokens} from any nested repair detail.

    rc20 update_entry findings look like {"path": ..., "field": "S", "rule_code": "entry_field_budget_exceeded",
    "expected": "max_tokens=40", ...}; older shapes carry max_tokens as a number. Both are accepted."""
    if isinstance(obj, dict):
        if "field" in obj and ("path" in obj or "object_ref" in obj):
            exp = str(obj.get("expected", "")) + " " + str(obj.get("message", ""))
            mt = obj.get("max_tokens")
            if mt is None:
                m = re.search(r"max_tokens\s*=\s*(\d+)", exp)
                mt = int(m.group(1)) if m else None
            mr = re.search(r"max_runes\s*=\s*(\d+)", exp)          # meta S quota (runes): fras_s_too_long
            if mt is not None:
                acc.append({"path": obj.get("path") or obj.get("object_ref"), "field": obj["field"], "max_tokens": int(mt)})
            elif mr is not None:
                acc.append({"path": obj.get("path") or obj.get("object_ref"), "field": obj["field"], "max_tokens": int(mr.group(1)) / 3.0})
        for v in obj.values():
            find_budget_failures(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_budget_failures(v, acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--entries", required=True, help="authoring clone aoci.code.txt (model-authored)")
    ap.add_argument("--authoring-root", required=True)
    ap.add_argument("--aoci-bin", default=os.environ.get("AOCI_BIN", "aoci"))
    ap.add_argument("--max-batches", type=int, default=400)
    ap.add_argument("--max-repairs", type=int, default=3)
    a = ap.parse_args()
    entries = load_entries(a.entries, a.authoring_root)
    print(f"loaded {len(entries)} authored entries from {a.entries}", flush=True)
    mcp = Mcp(a.aoci_bin, a.repo)
    t0 = time.time()
    applied = 0
    for b in range(a.max_batches):
        d, raw = mcp.call("aoci_maintain", {})
        cands = d.get("candidates") or []
        cp = d.get("code_plan") or {}
        if d.get("aligned") is True or (not cands and d.get("status") not in ("repair_required",)):
            print(f"maintain: status={d.get('status')} result={d.get('result')} aligned={d.get('aligned')} — no candidates, stop", flush=True)
            break
        if not cands:
            print("maintain returned no candidates but status", d.get("status"), raw[:600], flush=True)
            break
        batch = []
        missing = []
        for c in cands:
            if c.get("domain") != "code":
                continue
            line = entries.get(c["path"])
            if not line:
                missing.append(c["path"])
                continue
            batch.append({"candidate_id": c["candidate_id"], "path": c["path"], "source_sha256": c["source_sha256"], "new_entry": line})
        if missing:
            print("STOP: no authored entry for", missing, flush=True)
            sys.exit(2)
        rem = cp.get("remaining")
        for attempt in range(a.max_repairs + 1):
            r, rraw = mcp.call("aoci_update_entry", {"code_batch_id": cp.get("batch_id"), "entries": batch})
            st = r.get("status")
            if st in ("applied", "ok", "aligned") or r.get("ok") is True and st not in ("repair_required", "stopped"):
                break
            if st == "repair_required":
                fails = []
                find_budget_failures(r, fails)
                if not fails or attempt == a.max_repairs:
                    print("repair_required without fixable detail:", rraw[:1500], flush=True)
                    sys.exit(3)
                fixed = 0
                for f in fails:
                    path = (f.get("path") or "").replace("code:", "", 1)
                    for e in batch:
                        if e["path"] == path:
                            e["new_entry"] = repair(e["new_entry"], f.get("field"), int(f.get("max_tokens", 40)))
                            fixed += 1
                print(f"  repair_required: shortened {fixed} field(s) ({sorted({f.get('field') for f in fails})}), resubmitting the same batch", flush=True)
                continue
            print("STOP: update_entry status", st, rraw[:1500], flush=True)
            sys.exit(4)
        applied += len(batch)
        print(f"batch {b + 1}: submitted {len(batch)} entries, status={r.get('status')} remaining={rem} applied_total={applied} "
              f"{time.time() - t0:.0f}s", flush=True)
        if not rem:
            d2, _ = mcp.call("aoci_maintain", {})
            if not (d2.get("candidates") or []):
                print("maintain after final batch: status", d2.get("status"), "aligned", d2.get("aligned"), flush=True)
                break
    mcp.close()
    v = subprocess.run([a.aoci_bin, "--repo", a.repo, "verify", "--json"], capture_output=True, text=True).stdout
    try:
        vj = json.loads(v)
        g = vj.get("governance") or {}
        print(f"verify: aligned={vj.get('governance_aligned')} result={g.get('result')} entries={g.get('code_entry_count')} "
              f"findings={[(f.get('code'), f.get('target')) for f in (g.get('findings') or [])[:5]]}", flush=True)
        sys.exit(0 if vj.get("governance_aligned") else 1)
    except Exception:  # noqa: BLE001
        print("verify raw:", v[:600])
        sys.exit(1)


if __name__ == "__main__":
    main()
