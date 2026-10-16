#!/usr/bin/env python3
"""S2: rewrite the S (constraints) field of every code-volume entry from the COMPLETE source file.

  .venv/bin/python3 harness/s2_author.py --repo kcp-go-56b1fffe [--profile gptnb-opus5] [--workers 6]
                                          [--byte-budget 160000] [--limit N] [--paths a.go,b.go] [--dry-run]
                                          [--feedback feedback.json]

Why: the original authoring pass fed non-Python files as a 40-line head (outline.py HEAD_LINES), so every
DeepSWE S was written without ever seeing a function body. S2 feeds the whole file, keeps F/R/A/tag untouched,
applies the S2 quota bands (C8-9<=600, C4-7<=400, C1-3<=200 runes; token band = runes/3) and a fill floor
(>=80% of quota for C>=4, soft for C1-3). Output goes to deepswe-inject-s2/<repo>/ (frozen artefacts untouched);
the meta header declares the bands (#S配额:) — the pinned aoci binary's hard gate does not yet honour declared
quotas (R50 pending), a disclosed deviation. Inputs are repository files only: no task/benchmark/test-suite info.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import fras_lint  # noqa: E402
import llm  # noqa: E402

ROOT = os.path.expanduser("~/aoci-eval")
REPOS = os.path.join(ROOT, "repos-deepswe")
OUT = os.path.join(ROOT, "deepswe-inject-s2")
FILL_FLOOR = 0.80
def _bands():
    prof = os.environ.get("AOCI_S_QUOTA_PROFILE", "eval")
    if prof == "rc20":
        return "C9≤600 C8≤420 C7-5≤200 C4≤120 C3-1≤50"
    hi, mid, lo = _QUOTA_PROFILES[prof]
    return f"C9-8≤{hi} C7-4≤{mid} C3-1≤{lo}"


class _Bands(str):
    """惰性配额声明:随 AOCI_S_QUOTA_PROFILE 环境变量取值,保持旧代码的字符串用法不变。"""
    def __str__(self):
        return _bands()
    def __format__(self, spec):
        return format(_bands(), spec)


S2_BANDS = _Bands()


# S 配额档:eval=评测默认 600/400/200;mega=大仓压缩档 600/200/50(=产品机器契约,>1,000 文件仓用,2026-08-30 拍板)
_QUOTA_PROFILES = {"eval": (600, 400, 200), "mega": (600, 200, 50)}


# rc20 档(2026-10-14):aoci 0.1.0-rc20 的 cognition-budget-policy 按 token 计(≈3 字符/token,实测中位 3.02):
# S max_tokens C1-4=40 / C5-7=80 / C8=140 / C9=200;R max_tokens C1-4=50 / C5-7=90 / C8=140 / C9=180。字符上限 = token×3。
# 叠加 meta 卷自带的 rune 配额(#S quota: C9-8≤600 C7-4≤200 C3-1≤50)后取两者较小值
_RC20_S = {1: 50, 2: 50, 3: 50, 4: 120, 5: 200, 6: 200, 7: 200, 8: 420, 9: 600}
_RC20_R = {1: 150, 2: 150, 3: 150, 4: 150, 5: 270, 6: 270, 7: 270, 8: 420, 9: 540}


def rc20_r_limit(c):
    return _RC20_R[max(1, min(9, int(c)))]


def s2_quota(c):
    prof = os.environ.get("AOCI_S_QUOTA_PROFILE", "eval")
    if prof == "rc20":
        return _RC20_S[max(1, min(9, int(c)))]
    hi, mid, lo = _QUOTA_PROFILES[prof]
    return hi if c >= 8 else mid if c >= 4 else lo


def s2_token_band(c):
    return s2_quota(c) // 3


AUTHOR_SYSTEM = (
    "You are the index author for an AOCI repository-cognition index: one line per file, machine-checked. "
    "You see the COMPLETE source file. Write only what the file supports; never invent APIs, behaviours or relations. "
    "Answer with JSON only."
)

S2_RULES = """You are rewriting ONLY the S field of existing index entries. F, R, A and the tag are fixed context.

WHAT S IS. S holds the constraints, invariants and gotchas of this file that satisfy ALL of:
  (1) you only know them after reading the file body — they are NOT inferable from F, R, A, the file name or the signatures;
  (2) not knowing them would lead an engineer working on or around this file to make a concrete mistake;
  (3) they are facts of the code as it is, stated precisely (names, values, conditions), not advice.
Self-test for every clause: "which mistake, at which step, would an engineer make without this clause?" If you cannot name the mistake, drop the clause.

WHERE TO LOOK (the kinds of facts that usually qualify): preconditions and invariants callers must keep; hidden ordering and lifecycle rules (must call X before Y, init/close semantics, one-shot vs reusable); boundary and zero-value semantics (nil/empty/overflow, inclusive vs exclusive ranges); error contracts (what is swallowed, wrapped, panics, retried, partially applied); units, encodings and index bases (ms vs s, UTC, bytes vs runes, 0- vs 1-based); concurrency and mutability (locks, ownership, aliasing of returned pointers/slices, mutation during iteration, reentrancy); performance cliffs (quadratic paths, hidden copies, cache assumptions); compatibility and version traps; implicit couplings (global state, environment variables, side-effect registration, magic names).

WHAT S IS NOT: no restatement of F; no listing of interfaces (that is A); no relations (that is R); no generic advice ("be careful", "ensure correctness"); nothing deducible from names or signatures alone; no description of test cases; nothing about any external task or benchmark. Plain ASCII punctuation; no line breaks; never use the character '|'.

QUOTA (Unicode characters): C8-9 <= 600, C4-7 <= 400, C1-3 <= 200. The quota is a ceiling, not a target: fill it with qualifying facts ranked by how costly the mistake is. For C>=4 files aim to use at least 80% of the quota when the file supports it; for C1-3 write what qualifies and use "-" if nothing does.

The OLD S was written from only the first 40 lines of the file. Treat it as untrusted: keep an old fact only if the full file confirms it and it still qualifies; otherwise replace it."""


BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp", ".woff", ".woff2", ".ttf", ".otf", ".eot",
              ".pdf", ".zip", ".gz", ".tgz", ".tar", ".jar", ".wasm", ".so", ".dylib", ".dll", ".exe", ".bin", ".mp3", ".mp4"}


def is_text_file(path):
    if os.path.splitext(path)[1].lower() in BINARY_EXT:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(8192)
    except Exception:
        return False
    return b"\x00" not in head


def load_entries(repo_root):
    """Entries with repo-relative paths; section headers normalised to ===./=== / ===dir/=== form."""
    root = os.path.realpath(repo_root)
    entries = []
    cur = ""
    for line in open(os.path.join(repo_root, "aoci.code.txt"), encoding="utf-8"):
        raw = line.rstrip("\n")
        if raw.startswith("===") and raw.endswith("==="):
            d = raw.strip("=").strip()
            if d.startswith("/"):
                rel = os.path.relpath(os.path.realpath(d), root)
            else:
                rel = d[2:] if d.startswith("./") else d
            rel = rel.strip("/")
            cur = "" if rel in ("", ".") else rel + "/"
            entries.append({"raw": "===./===" if not cur else f"==={cur}===", "kind": "section"})
            continue
        m = fras_lint.ENTRY_RE.match(raw)
        if m:
            e = {"raw": raw, "kind": "entry", "path": cur + m.group("name"), "name": m.group("name"),
                 "tag": m.group("tag"), "F": m.group("F"), "R": m.group("R"), "A": m.group("A"), "S": m.group("S")}
            mc = re.search(r"[1-9]", e["tag"])
            e["C"] = int(mc.group(0)) if mc else 5
            entries.append(e)
        else:
            entries.append({"raw": raw, "kind": "other"})
    return entries


def clean_s(s):
    s = str(s or "").replace("|", "/").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or "-"


def file_block(e, text):
    c = e["C"]
    return (f"### FILE: {e['path']}  (C={c}, S quota {s2_quota(c)} chars"
            + (f", fill target >= {int(FILL_FLOOR * s2_quota(c))} chars" if c >= 4 else ", C1-3: write what qualifies")
            + f")\ntag: {e['tag']}\nF: {e['F']}\nR: {e['R']}\nA: {e['A']}\nOLD S: {e['S']}\n"
            f"<<<BEGIN FILE ({len(text)} bytes)>>>\n{text}\n<<<END FILE>>>")


def author_prompt(blocks, feedback=None):
    fb = ""
    if feedback:
        fb = ("\n\nAUDIT FEEDBACK on the previous S of some files (clauses judged inferable from F/A/signatures alone, i.e. "
              "low-entropy — replace them with body-level facts or drop them):\n"
              + "\n".join(f"- {p}: {'; '.join(cl)}" for p, cl in feedback.items()) + "\n")
    return (S2_RULES + fb + "\n\nReturn a JSON array: [{\"path\": \"<exactly as given>\", \"S\": \"<new S>\"}]\n\n"
            + "\n\n".join(blocks))


def extend_prompt(items):
    body = "\n\n".join(f"path: {p}\ncurrent S ({len(s)} chars of {q} quota): {s}" for p, s, q in items)
    return (S2_RULES + "\n\nThe S below uses less than 80% of its quota. Re-read the file text you were given for these "
            "files (repeated below) and, ONLY if the file supports further facts that pass the self-test, extend S up to the "
            "quota. If nothing more qualifies, return the same S unchanged and set \"exhausted\": true.\n"
            "Return a JSON array: [{\"path\": ..., \"S\": ..., \"exhausted\": bool}]\n\n" + body)


def shorten_prompt(items):
    body = "\n\n".join(f"path: {p}\nS ({len(s)} chars, quota {q}): {s}" for p, s, q in items)
    return ("Some S fields exceed their quota. Shorten each to fit within its quota (Unicode characters), keeping the "
            "facts whose omission would cause the costliest mistakes; do not add facts. "
            "Return a JSON array: [{\"path\": ..., \"S\": ...}]\n\n" + body)


def parse_json_array(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\[.*\]", t, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return arr if isinstance(arr, list) else None


def call(prof, system, user, max_tokens=16000):
    r = llm.complete(prof, [{"type": "text", "text": system}], user, max_tokens=max_tokens, temperature=0)
    return r["text"], dict(r.get("usage") or {})


def hard_truncate(s, quota):
    if len(s) <= quota:
        return s
    cut = s[:quota]
    for sep in (". ", "; "):
        i = cut.rfind(sep)
        if i > quota * 0.5:
            return cut[:i + 1].strip()
    return cut.rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--profile", default="gptnb-opus5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--byte-budget", type=int, default=160_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--paths", default="")
    ap.add_argument("--feedback", default="", help="json {path: [clauses]} from s2_audit to drive a rewrite round")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    repo_root = os.path.join(REPOS, a.repo)
    out_dir = os.path.join(OUT, a.repo)
    os.makedirs(out_dir, exist_ok=True)
    entries = load_entries(repo_root)
    targets = [e for e in entries if e["kind"] == "entry"]
    feedback = json.load(open(a.feedback)) if a.feedback else None
    if a.paths:
        want = set(a.paths.split(","))
        targets = [e for e in targets if e["path"] in want]
    elif feedback:
        targets = [e for e in targets if e["path"] in feedback]
    if a.limit:
        targets = targets[:a.limit]

    # attach file text; skip unreadable
    work = []
    for e in targets:
        p = os.path.join(repo_root, e["path"])
        if not is_text_file(p):
            e["skip"] = "binary-or-missing"
            continue
        try:
            e["text"] = open(p, encoding="utf-8", errors="replace").read()
            work.append(e)
        except Exception:
            e["skip"] = "unreadable"
    # batch by byte budget (big files alone)
    batches, cur, size = [], [], 0
    for e in sorted(work, key=lambda x: x["path"]):
        b = len(e["text"].encode("utf-8"))
        if cur and (size + b > a.byte_budget or len(cur) >= 12):
            batches.append(cur)
            cur, size = [], 0
        cur.append(e)
        size += b
    if cur:
        batches.append(cur)
    total_bytes = sum(len(e["text"].encode("utf-8")) for e in work)
    skipped = [e["path"] for e in targets if e.get("skip")]
    print(f"s2 · {a.repo} · entries={len(work)} skipped(binary/missing)={len(skipped)} batches={len(batches)} src={total_bytes // 1024}KB "
          f"model={a.profile} workers={a.workers}" + (" · DRY-RUN" if a.dry_run else ""), flush=True)
    if a.dry_run:
        for i, b in enumerate(batches[:3]):
            print(f"  batch {i}: {[e['path'] for e in b]} ({sum(len(e['text']) for e in b) // 1024}KB)")
        sys.exit(0)

    llm.load_secrets()
    prof = llm.load_profile(a.profile)
    ledger_path = os.path.join(out_dir, "s2-ledger.jsonl")
    usage_tot = {"in": 0, "out": 0}

    def do_batch(batch):
        t0 = time.time()
        blocks = [file_block(e, e["text"]) for e in batch]
        fb = {e["path"]: feedback[e["path"]] for e in batch if feedback and e["path"] in feedback} or None
        text, u = call(prof, AUTHOR_SYSTEM, author_prompt(blocks, fb))
        usage_tot["in"] += u.get("input", 0) or 0
        usage_tot["out"] += u.get("output", 0) or 0
        objs = {o.get("path"): clean_s(o.get("S")) for o in (parse_json_array(text) or []) if isinstance(o, dict)}
        res = {}
        for e in batch:
            res[e["path"]] = {"S": objs.get(e["path"], None), "attempts": 1}
        # extend round for C>=4 under floor
        under = [(e["path"], res[e["path"]]["S"], s2_quota(e["C"])) for e in batch
                 if e["C"] >= 4 and res[e["path"]]["S"] and res[e["path"]]["S"] != "-"
                 and len(res[e["path"]]["S"]) < FILL_FLOOR * s2_quota(e["C"])]
        if under:
            blk = "\n\n".join(file_block(e, e["text"]) for e in batch if e["path"] in {p for p, _, _ in under})
            text2, u2 = call(prof, AUTHOR_SYSTEM, extend_prompt(under) + "\n\n" + blk)
            usage_tot["in"] += u2.get("input", 0) or 0
            usage_tot["out"] += u2.get("output", 0) or 0
            for o in parse_json_array(text2) or []:
                if isinstance(o, dict) and o.get("path") in res and o.get("S"):
                    res[o["path"]]["S"] = clean_s(o["S"])
                    res[o["path"]]["exhausted"] = bool(o.get("exhausted"))
                    res[o["path"]]["attempts"] += 1
        # shorten round for over-quota
        over = [(e["path"], res[e["path"]]["S"], s2_quota(e["C"])) for e in batch
                if res[e["path"]]["S"] and (len(res[e["path"]]["S"]) > s2_quota(e["C"])
                                            or fras_lint.tokens(res[e["path"]]["S"]) > s2_token_band(e["C"]))]
        if over:
            text3, u3 = call(prof, AUTHOR_SYSTEM, shorten_prompt(over), max_tokens=8000)
            usage_tot["in"] += u3.get("input", 0) or 0
            usage_tot["out"] += u3.get("output", 0) or 0
            for o in parse_json_array(text3) or []:
                if isinstance(o, dict) and o.get("path") in res and o.get("S"):
                    res[o["path"]]["S"] = clean_s(o["S"])
                    res[o["path"]]["attempts"] += 1
        for e in batch:
            r = res[e["path"]]
            q = s2_quota(e["C"])
            if r["S"] and len(r["S"]) > q:
                r["S"] = hard_truncate(r["S"], q)
                r["truncated"] = True
            r["fill"] = round(len(r["S"]) / q, 2) if r["S"] else 0
            r["fill_short"] = bool(e["C"] >= 4 and r["S"] and r["S"] != "-" and r["fill"] < FILL_FLOOR)
        return {"res": res, "wall": round(time.time() - t0, 1)}

    results = {}

    def on_result(batch, r, i, n):
        if isinstance(r, dict) and "res" in r:
            results.update(r["res"])
            print(f"[batch {i}/{n}] files={len(batch)} wall={r['wall']}s", flush=True)
        else:
            print(f"[batch {i}/{n}] ERROR {str(r)[:200]}", flush=True)

    common.run_parallel(batches, do_batch, workers=a.workers, on_result=on_result, label="s2")

    # write outputs: code volume with S replaced; meta with declared bands; root copied
    by_path = {e["path"]: e for e in work}
    out_lines = []
    with open(ledger_path, "a", encoding="utf-8") as lf:
        for e in entries:
            if e["kind"] == "entry" and e["path"] in results and results[e["path"]]["S"]:
                r = results[e["path"]]
                new = f"{e['name']}[{e['tag']}]: F:{e['F']} | R:{e['R']} | A:{e['A']} | S:{r['S']}"
                out_lines.append(new)
                import hashlib
                lf.write(json.dumps({"path": e["path"], "C": e["C"], "old_S": e["S"], "new_S": r["S"],
                                     "src_sha": hashlib.sha256(by_path[e["path"]]["text"].encode("utf-8")).hexdigest()[:16],
                                     "src_lines": by_path[e["path"]]["text"].count("\n"),
                                     "quota": s2_quota(e["C"]), "fill": r["fill"], "fill_short": r.get("fill_short"),
                                     "exhausted": r.get("exhausted"), "truncated": r.get("truncated"),
                                     "attempts": r["attempts"], "bytes": len(by_path[e["path"]]["text"].encode()),
                                     "ts": time.strftime("%F %T")}, ensure_ascii=False) + "\n")
            else:
                out_lines.append(e["raw"])
    # merge with a previous output if present (partial rounds): key by full path, keep earlier S2 results
    prev = os.path.join(out_dir, "aoci.code.txt")
    if os.path.exists(prev) and (a.paths or feedback or a.limit):
        prev_map, cur = {}, ""
        for raw in open(prev, encoding="utf-8").read().splitlines():
            if raw.startswith("===") and raw.endswith("==="):
                d = raw.strip("=").strip()
                cur = "" if d in ("", ".") else d.rstrip("/") + "/"
                continue
            m = fras_lint.ENTRY_RE.match(raw)
            if m:
                prev_map[cur + m.group("name")] = raw
        merged = []
        for e, l in zip(entries, out_lines):
            if e["kind"] == "entry" and not (e["path"] in results and results[e["path"]]["S"]) and e["path"] in prev_map:
                merged.append(prev_map[e["path"]])
            else:
                merged.append(l)
        out_lines = merged
    open(prev, "w", encoding="utf-8").write("\n".join(out_lines) + "\n")
    meta = open(os.path.join(repo_root, "aoci.meta.txt"), encoding="utf-8").read().splitlines()
    meta2 = []
    for l in meta:
        if l.startswith("#S quota:"):
            # S2_BANDS 是惰性配额对象:必须走 str()/f-string 取值;"+" 拼接在 mega 档下得到空串(2026-09-04 comet 封卷时二进制拒收)
            meta2.append(f"#S quota: {S2_BANDS}")
            meta2.append(f"#S配额: {S2_BANDS}")
        elif l.startswith("#S配额:"):
            continue
        else:
            meta2.append(l)
    open(os.path.join(out_dir, "aoci.meta.txt"), "w", encoding="utf-8").write("\n".join(meta2) + "\n")
    open(os.path.join(out_dir, "aoci.txt"), "w", encoding="utf-8").write(
        open(os.path.join(repo_root, "aoci.txt"), encoding="utf-8").read())

    done = [r for r in results.values() if r["S"]]
    fills = [r["fill"] for r in done]
    print(f"\ns2 done · {len(done)}/{len(work)} entries · mean fill {sum(fills) / max(1, len(fills)):.2f} · "
          f"fill_short {sum(1 for r in done if r.get('fill_short'))} · truncated {sum(1 for r in done if r.get('truncated'))} · "
          f"tokens in={usage_tot['in']} out={usage_tot['out']} · est ${usage_tot['in'] / 1e6 * 5 + usage_tot['out'] / 1e6 * 25:.2f}")
    print(f"→ {out_dir}/aoci.code.txt (+meta with declared bands, ledger)")


if __name__ == "__main__":
    main()
