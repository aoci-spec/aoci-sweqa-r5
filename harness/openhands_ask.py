#!/usr/bin/env python3
"""OpenHands host adapter: answer repository questions with the OpenHands SDK
default agent — the SWE-QA paper's agent configuration — with or without AOCI.

Runs under the separate OpenHands venv (.venv-oh, see requirements-openhands.txt):
openhands-sdk 1.1.0 `get_default_agent(cli_mode=True)` = terminal + file_editor +
task_tracker tools + built-in finish/think, `max_iteration_per_run` 10 (paper).
The LLM is our endpoint profile through LiteLLM (`openai/<model>` → the
OpenAI-compatible gateway), temperature 0.

EVERY arm's workspace is a pristine git worktree at the pinned commit — byte-identical
across arms. A map never sits in the agent's working directory; it is served from a
per-worker MIRROR worktree of the same commit that the agent is never given a reason to
visit. So the arms differ only in which MCP server is attached and one structurally
identical hint line naming its tools:
  oh          no map
  oh-aoci     AOCI served from <mirror> (aoci --repo <mirror> mcp), 9 tools
              ('oh-mcp' is an alias). The mirror carries the volumes + AGENTS.md + .aoci
              because AOCI's own governance requires the indexed files to exist
              (verified: governance_aligned=True, guide complete=True).
  oh-cg       CodeGraph served from <mirror> (codegraph serve --mcp -p <mirror> --no-watch),
              1 tool (its shipped default; CODEGRAPH_MCP_TOOLS is deliberately not set)
  oh-inject   no server; the volumes' text is prepended to the first message

No arm can run a map CLI from the terminal: $HOME/.local/bin is stripped from PATH for
the whole adapter process (CodeGraph's own benchmark blocks its CLI in both arms for the
same reason), and the MCP servers are launched by absolute path. Every record carries a
`contamination` list — references in the agent's own commands to a map artefact or to a
path outside its workspace — so the isolation is a reported number, not an assumption.

Each worker owns one git worktree (pool = --workers); before every question the
worktree is reset (`git checkout -- . && git clean -fdq`, then index files are
re-copied for the index arms) so the terminal tool can never leak state between
questions. Every question gets a fresh Conversation.

Records use the same schema as agent_ask.py: answer, steps (LLM iterations that
issued tool calls; one iteration may issue several), tool_calls (ActionEvents), files_read (file_editor view + cat/sed/head/tail
targets, repo-relative), trace, rounds (per-LLM-call tokens), usage, retried,
forced, wall_s. Absolute workspace paths in the answer are rewritten to
repo-relative so the judge sees the same path style as the other arms.

Usage:
  .venv-oh/bin/python3 harness/openhands_ask.py --repo-name xarray --repo-root repos/xarray \
      --questions benchmarks/swe-qa/xarray.jsonl --condition oh --out runs/x/answers-oh.jsonl [--workers 7] [--max-iterations 10]
"""
import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import llm  # noqa: E402
from ask import UPSTREAM_SYSTEM, VOLUME_FILES  # noqa: E402

AOCI_BIN = os.environ.get("AOCI_BIN", "aoci")
CG_BIN = os.environ.get("CODEGRAPH_BIN", os.path.expanduser("~/.local/bin/codegraph"))
# Worktrees live OUTSIDE the harness repository. When they sat under <ROOT>/.worktrees, an agent that lost its
# question walked `cd ../..` into the harness itself — MEMO, specs, harness/*.py, and runs/ (whose records carry
# the reference answers) were all two levels up (k=1 preflight, 2026-08-19). Now `..` from a workspace reaches
# an otherwise empty directory. Override with AOCI_EVAL_WORKTREES for a different disk.
WORKTREES = os.environ.get("AOCI_EVAL_WORKTREES") or os.path.join(os.path.expanduser("~"), ".cache", "eval-workspaces")   # neutral name: no 'aoci' breadcrumb in the agent's own pwd
MIRROR_ARMS = {"oh-aoci", "oh-cg", "oh-inject-cg", "oh-aoci-cg"}      # arms whose map is served from a mirror worktree
MAP_TOOL_PREFIXES = ("aoci_", "codegraph_")
INIT_LOCK = threading.Lock()
MAX_OUTPUT_TOKENS = [12288]   # set from --max-output-tokens in main()
REPO_ROOT_HINT = [None]   # set once in main(); lets rel() normalise reads that went to the authoring tree  # SDK 1.1.0 tool registration / agent init is not safe under concurrent construction

# Delivered as the agent's SYSTEM-prompt suffix (AgentContext.system_message_suffix), not inside the user turn.
# With the SWE-QA protocol text ("You are a direct answer generator…") sitting in the USER message, DeepSeek in
# OpenHands answered "I don't see a specific task or question in your message" on 0/48, then 10/48, then 7/48
# baseline questions across three k=1 preflights — the same bytes, temperature 0 — while the minimal agent, which
# frames the identical text as "Final-answer format:" in its system prompt, had 0/144. So the framing is now the
# same on both hosts, and the user turn is exactly SWE-QA's `Repository: X\n Question: Y`. This is CLOSER to
# the protocol (system message = SWE-QA's text; user message = repository + question), not further from it.
TASK_INSTRUCTION = (
    "You are answering a question about a specific code repository. The repository is checked out in the current "
    "working directory; inspect it with your tools as needed, but do NOT modify, create or delete any file. Read the "
    "relevant source before answering; do not guess when a file can settle the point. When you know the answer, reply "
    "with the final answer as a plain message. Final-answer format: " + UPSTREAM_SYSTEM
)
# Every arm gets ONE sentence of the same shape naming the tools it should navigate with, so the
# contrast is "which tools" and not "an extra instruction vs none" (the baseline used to get nothing).
BASE_HINT = ("\n\nThis repository is a plain checkout: a map of it is not provided. Your terminal and "
             "file_editor tools are how you navigate (grep, find, view); use them to decide which files "
             "to read, then read them.")
AOCI_HINT = ("\n\nThis repository has an AOCI cognition index: a map of the repository (one line per file — function, "
             "relations, interfaces, constraints). The aoci_* tools give you that map (aoci_overview, aoci_search, "
             "aoci_get_entries); use it to decide which files to read, then read them.")
CG_HINT = ("\n\nThis repository has a CodeGraph index: a map of the repository (symbols, call graph, imports, "
           "dependents). The codegraph_explore tool gives you that map; use it to decide which files to read, "
           "then read them.")
# The index arrives as a BUILT ARTEFACT (harness/build_index_artifact.py): the three volumes with repo-relative
# headers and a one-line legend stating the entry grammar. The legend lives in the artefact so every host injects
# identical bytes; only the same-shape behavioural sentence stays here, mirroring BASE_HINT / CG_HINT.
#
# ORDER MATTERS. The first shape was [question+instruction][index part 1..n]: with ~105k characters of index
# after it, DeepSeek V4 Flash lost the question on 2/48 questions ("no benchmark question was present…") and
# wandered out of its worktree looking for one (k=1 preflight, 2026-08-19); Sonnet 5 never did (0/144). The
# minimal agent, which keeps the index in the system prompt and the question LAST, had 0/48. So the inject
# message is now [intro][index part 1..n][question + instruction + hint]: the question is the most recent thing
# the model reads. Same-shape hint sentence, same bytes of index; only the position of the task block changed.
OH_COMBO_HINT = ("\n\nThis repository has TWO maps: the AOCI cognition index (the user message carries it in full, "
                 "one line per file) and a CodeGraph index (the codegraph_explore tool: symbols, call graph, imports). "
                 "Use either or both to decide which files to read, then read them with your tools and answer from the source.")
OH_FILE_HINT = ("\n\nThis repository carries its AOCI cognition index as files at the repo root (aoci.txt, "
                "aoci.meta.txt, aoci.code.txt — one line per source file). Grep or read them to decide which "
                "files to read, then read those files.")
OH_AOCI_CG_HINT = ("\n\nThis repository has TWO maps, both served as tools: the AOCI cognition index (aoci_overview, "
                   "aoci_search, aoci_get_entries — one line per file: function, relations, interfaces, constraints) and a "
                   "CodeGraph index (codegraph_explore — symbols, call graph, imports, dependents). Use them to decide which "
                   "files to read, then read them.")
# Cognition turn (2026-10-14, r4 batch): the product contract (AGENTS.md written by `aoci init`) makes every new Agent Run
# establish complete repository cognition BEFORE the task — aoci_rules, then aoci_overview through every next_cursor to
# completed=true, then the Attestation. Under SWE-QA's 10-iteration task budget that chain cannot fit (3-4 chunks at
# chunk_tokens=24000 + rules + attestation), so it runs as its own conversation turn with its own budget; the question
# then follows as the paper's 10-iteration turn with the delivered Whole-Index already in context. Recorded per record
# under "cognition" (calls, chunks, completed, attestation outcome) so it is a reported number, not an assumption.
COGNITION_PROMPT = (
    "Before any question, establish complete cognition of this repository through the AOCI tools, exactly as the "
    "repository contract requires: (1) call aoci_rules; (2) call aoci_overview with no arguments, then keep calling "
    "aoci_overview with the returned next_cursor until a response says completed=true, reading every delivered chunk; "
    "(3) then call aoci_overview once more with BOTH objects and no other arguments: host_delivery_confirmation = "
    "{version: 'overview-delivery-receipt/v1', body_bytes, body_sha256 (copied from the delivery metadata), "
    "end_marker_observed: true} and model_cognition_attestation = {version (the attestation_contract_version of the "
    "final chunk), index_sha256, entry_sequence_sha256, entry_count (copy the challenge_* values exactly), "
    "challenge_digest, reported_entry_count, reported_estimated_tokens, coverage_percent, confidence_percent, "
    "system_mastery_percent, truncation_detected, unseen_sections (array), uncertainty_reasons (array), "
    "challenge_answers: one object per challenge ordinal with exactly the keys ordinal, object_identity (the exact "
    "repository-relative path), tag, core_f}. Use no other property names. Do not read repository files and do not "
    "answer anything in this turn. When cognition is established, reply with the single line: COGNITION ESTABLISHED.")
COGNITION_ITERS = int(os.environ.get("AOCI_COGNITION_ITERS", "12"))
COGNITION_ARMS = {"oh-aoci", "oh-aoci-cg"}
SDK_TEXT_LIMIT = int(os.environ.get("AOCI_SDK_TEXT_LIMIT", "400000"))
INJECT_INTRO = "Below is the AOCI cognition index of this repository; the question follows it.\n\n"
INJECT_HINT = ("\n\nThis repository has an AOCI cognition index: the user message carries it in full (one line per file). "
               "It is a map, not the code: use it to decide which files to read, then read them with your tools and "
               "answer from the source.")
FORCED_ANSWER = ("Your tool budget is exhausted. Give your final answer now, from what you have already read; "
                 "do not use any more tools.")

# openhands-sdk 1.1.0 truncates any single TextContent above 50,000 characters, middle-out, every time
# the message is serialised (message.py:176; the list serializer is always chosen because our agent has
# tools, so function_calling_enabled is true). A 106,376-character index injected as ONE part therefore
# reached the model as 47% of itself with the middle deleted — silently, on every round. Splitting the
# same text across several parts of the SAME message is lossless and touches no SDK behaviour.
TEXT_PART_LIMIT = 45_000   # 10% headroom under the SDK's 50,000


def index_parts(text):
    """Split on entry boundaries so no line is cut in half; label each part for the model."""
    if len(text) <= TEXT_PART_LIMIT:
        return [text]
    parts, cur = [], []
    n = 0
    for line in text.splitlines(keepends=True):
        if n + len(line) > TEXT_PART_LIMIT and cur:
            parts.append("".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line)
    if cur:
        parts.append("".join(cur))
    total = len(parts)
    return [f"[AOCI index part {i+1}/{total}]\n{p}" for i, p in enumerate(parts)]
# Files the AOCI mirror needs: the volumes, plus AGENTS.md because the index has an entry for it
# and AOCI's governance blocks on an entry whose file is missing (code_orphan). None of this is ever
# placed in the agent's workspace.
MIRROR_FILES = ("AGENTS.md",)
# .aoci subtrees no MCP tool reads: 86 MB of transaction bodies + verify history per copy.
AOCI_SKIP_SUBDIRS = ("transactions", "verify_history")


# ---------------------------------------------------------------- worktrees
def sh(cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


CG_ENV = {"CODEGRAPH_TELEMETRY": "0", "NO_COLOR": "1"}


def codegraph_stats(ws):
    """Node/edge counts of the built graph — proof the index is real, not just a file named .db."""
    p = subprocess.run([CG_BIN, "status", "."], cwd=ws, capture_output=True, text=True,
                       env=dict(os.environ, **CG_ENV))
    out = (p.stdout or "") + (p.stderr or "")

    def num(label):
        m = re.search(rf"{label}:\s+([\d,]+)", out)
        return int(m.group(1).replace(",", "")) if m else None

    db = os.path.join(ws, ".codegraph", "codegraph.db")
    return {"files": num("Files"), "nodes": num("Nodes"), "edges": num("Edges"),
            "db_bytes": os.path.getsize(db) if os.path.exists(db) else 0,
            "up_to_date": "up to date" in out}


def codegraph_init(ws):
    """Build the CodeGraph index inside a worktree (local SQLite, no key; ~2 s warm on xarray)."""
    if not os.path.exists(CG_BIN):
        sys.exit(f"codegraph binary not found at {CG_BIN}; install it or set CODEGRAPH_BIN")
    t0 = time.time()
    p = subprocess.run([CG_BIN, "init", "."], cwd=ws, capture_output=True, text=True,
                       env=dict(os.environ, **CG_ENV))
    st = codegraph_stats(ws)
    if p.returncode != 0 or not st["nodes"] or not st["edges"] or st["db_bytes"] < 1_000_000:
        raise RuntimeError(f"codegraph init did not produce a usable index in {ws}: rc={p.returncode} "
                           f"stats={st} out={(p.stdout + p.stderr)[-300:]}")
    return round(time.time() - t0, 1), st


def _add_worktree(repo_root, path, commit):
    if os.path.exists(path):
        sh(["git", "-C", repo_root, "worktree", "remove", "--force", path], check=False)
        shutil.rmtree(path, ignore_errors=True)
    sh(["git", "-C", repo_root, "worktree", "add", "--detach", path, commit])
    return path


def export_tree(repo_root, commit, dest):
    """The agent's workspace: a PLAIN EXPORT of <commit> — the tracked files and nothing else.

    Not a worktree: a worktree's `.git` file says `gitdir: <repo_root>/.git/worktrees/…`, which points the
    agent straight at the authoring tree (where the AOCI volumes live); DeepSeek did `cat .git` while probing
    (k=1b, 2026-08-19). An export has no VCS metadata, no map, and nothing that names another directory.
    """
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    p1 = subprocess.Popen(["git", "-C", repo_root, "archive", "--format=tar", commit], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", dest], stdin=p1.stdout, check=True)
    p1.wait()
    if p1.returncode:
        raise RuntimeError(f"git archive {commit} failed in {repo_root}")
    return dest


def tree_fpmap(root):
    """relpath -> (size, mtime_ns) over the whole tree: the per-path view of tree_fingerprint, kept per slot so a dirty
    workspace can name WHAT changed (r5: GLM-4.6 answers came back dirty with only view/grep/find in the trace)."""
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            fp = os.path.join(dp, f)
            try:
                st_ = os.stat(fp)
            except OSError:
                continue
            out[os.path.relpath(fp, root)] = (st_.st_size, st_.st_mtime_ns)
    return out


def dirty_paths(slot, limit=50):
    """Paths added / removed / modified in the agent's tree since the slot fingerprint was taken (sorted, capped)."""
    before = slot.get("fpmap") or {}
    after = tree_fpmap(slot["ws"])
    changed = sorted({p for p in set(before) | set(after) if before.get(p) != after.get(p)})
    return changed[:limit]


def tree_fingerprint(root):
    """path|size|mtime over the whole tree — cheap, and enough to tell 'the agent edited/created/deleted a file'."""
    h = hashlib.sha1()
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            fp = os.path.join(dp, f)
            try:
                st_ = os.stat(fp)
            except OSError:
                continue
            h.update(f"{os.path.relpath(fp, root)}|{st_.st_size}|{st_.st_mtime_ns}\n".encode())
    return h.hexdigest()


def fill_aoci_mirror(mirror, repo_root):
    """Give the mirror everything AOCI needs to serve an aligned index — and nothing to the agent."""
    for name in VOLUME_FILES + MIRROR_FILES:
        src = os.path.join(repo_root, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(mirror, name))
    src = os.path.join(repo_root, ".aoci")
    if os.path.isdir(src):
        shutil.rmtree(os.path.join(mirror, ".aoci"), ignore_errors=True)   # .aoci/.gitignore is `*`: git clean cannot
        shutil.copytree(src, os.path.join(mirror, ".aoci"), ignore=shutil.ignore_patterns(*AOCI_SKIP_SUBDIRS))


def aoci_aligned(mirror):
    p = sh([AOCI_BIN, "--repo", mirror, "verify", "--json"], check=False)
    try:
        g = json.loads(p.stdout)["governance"]
        return {"governance_aligned": g["governance_aligned"], "result": g["result"],
                "entries": g["code_entry_count"], "findings": g["findings"][:3]}
    except Exception:  # noqa: BLE001
        return {"governance_aligned": None, "result": (p.stdout + p.stderr)[-200:]}


def place_volumes(ws, repo_root):
    """oh-file: the three volumes at the workspace root, directory headers rewritten repo-relative.

    This is the treatment, declared as such — unlike the accidental delivery the adversarial review killed
    (index + a 14 KB manual copied into the AOCI arm). The volumes on disk carry ABSOLUTE authoring-machine
    headers (the product writes them; memo §9.41), which misdirect an agent out of its workspace, so the
    placed copies get the same repo-relative rewrite the artefact gets. Returns {name: sha256} of the placed files."""
    from build_index_artifact import relativize
    shas = {}
    for name_ in VOLUME_FILES:
        src = os.path.join(repo_root, name_)
        if not os.path.exists(src):
            sys.exit(f"oh-file: missing volume {src}")
        text, _real = relativize(open(src, encoding="utf-8").read(), repo_root)
        with open(os.path.join(ws, name_), "w", encoding="utf-8") as f:
            f.write(text)
        shas[name_] = common.sha256_text(text)[:16]
    return shas


def make_slots(repo_root, name, n, commit, condition):
    """One slot per worker: the agent's pristine EXPORT (<WORKTREES>/<name>-<i>/repo — `..` from inside it is a
    private directory holding only `repo`) + (for map arms) a mirror WORKTREE under <WORKTREES>/mirrors/ that
    holds the map and that the agent is never pointed at."""
    os.makedirs(os.path.join(WORKTREES, "mirrors"), exist_ok=True)
    sh(["git", "-C", repo_root, "worktree", "prune"], check=False)
    slots, prep = [], {}
    try:
        cg_stats, cg_secs = [], []
        shared_cg = None
        if condition in ("oh-cg", "oh-inject-cg", "oh-aoci-cg"):
            # ONE mirror for the whole pass: codegraph serves it read-only (--no-watch), and per-conversation MCP
            # servers all read the same SQLite db (the minimal host's CgPool proved this). 48 per-worker mirrors
            # cost 3-4 minutes of sequential init per pass — the tail-collapse the user saw (2026-08-20).
            shared_cg = _add_worktree(repo_root, os.path.join(WORKTREES, "mirrors", f"{name}-shared"), commit)
            secs, st = codegraph_init(shared_cg)
            cg_secs.append(secs)
            cg_stats.append(st)
        for i in range(n):
            ws = export_tree(repo_root, commit, os.path.join(WORKTREES, f"{name}-{i}", "repo"))
            vol_shas = place_volumes(ws, repo_root) if condition == "oh-file" else None
            mirror = None
            if condition in ("oh-cg", "oh-inject-cg"):
                mirror = shared_cg
            elif condition in MIRROR_ARMS:
                mirror = _add_worktree(repo_root, os.path.join(WORKTREES, "mirrors", f"{name}-{i}"), commit)
                fill_aoci_mirror(mirror, repo_root)
            slots.append({"ws": ws, "mirror": mirror, "fp": tree_fingerprint(ws), "fpmap": tree_fpmap(ws), "commit": commit, "repo_root": repo_root,
                          "condition": condition, "cg_mirror": shared_cg if condition == "oh-aoci-cg" else None})
        if condition == "oh-file":
            prep["volumes_placed"] = vol_shas
        if cg_stats:
            if len({(x["nodes"], x["edges"]) for x in cg_stats}) > 1:
                raise RuntimeError(f"codegraph indexes disagree across mirrors: "
                                   f"{[(x['nodes'], x['edges']) for x in cg_stats]}")
            prep = {"map_build": {"seconds": cg_secs, "graph": cg_stats[0]}}
            print(f"codegraph init: {len(cg_secs)} mirrors, {min(cg_secs)}-{max(cg_secs)} s each, "
                  f"{cg_stats[0]['nodes']} nodes / {cg_stats[0]['edges']} edges / {cg_stats[0]['db_bytes']} B", flush=True)
        if condition in ("oh-aoci", "oh-aoci-cg"):
            checks = [aoci_aligned(s["mirror"]) for s in slots]
            bad = [c for c in checks if not c.get("governance_aligned")]
            if bad:
                raise RuntimeError(f"AOCI mirror is not governance-aligned: {bad[0]}")
            prep = dict(prep, map_state=checks[0])
            print(f"aoci mirrors: {len(slots)} aligned, {checks[0]['entries']} entries", flush=True)
        return slots, prep
    except BaseException:            # never leave half-built worktrees behind
        remove_slots(repo_root, slots)
        raise


def remove_slots(repo_root, slots):
    seen = set()
    for s in slots:
        ws, mirror = s.get("ws"), s.get("mirror")
        if ws:
            kill_mcp_servers(ws)
            shutil.rmtree(os.path.dirname(ws), ignore_errors=True)     # <name>-<i>/ (the private parent)
        if mirror and mirror not in seen:                              # cg arms share one mirror across all slots
            seen.add(mirror)
            kill_mcp_servers(mirror)
            sh(["git", "-C", repo_root, "worktree", "remove", "--force", mirror], check=False)
            shutil.rmtree(mirror, ignore_errors=True)
        cgm = s.get("cg_mirror")
        if cgm and cgm not in seen:                                    # oh-aoci-cg: the pass-shared CodeGraph mirror
            seen.add(cgm)
            kill_mcp_servers(cgm)
            sh(["git", "-C", repo_root, "worktree", "remove", "--force", cgm], check=False)
            shutil.rmtree(cgm, ignore_errors=True)
    sh(["git", "-C", repo_root, "worktree", "prune"], check=False)


def reset_slot(slot, repo_root, condition):
    """Undo anything the previous question left behind, in the agent's tree and in the mirror."""
    ws = slot["ws"]
    if tree_fingerprint(ws) != slot["fp"]:                     # the agent touched its tree: re-export it
        export_tree(slot["repo_root"], slot["commit"], ws)
        if slot.get("condition") == "oh-file":
            place_volumes(ws, slot["repo_root"])
        slot["fp"] = tree_fingerprint(ws)
        slot["fpmap"] = tree_fpmap(ws)
    mirror = slot.get("mirror")
    if not mirror:
        return
    if slot.get("condition") in ("oh-cg", "oh-inject-cg") or condition in ("oh-cg", "oh-inject-cg"):
        # SHARED read-only mirror: 48 workers must not git-clean it concurrently; re-init only if the db vanished
        if not os.path.exists(os.path.join(mirror, ".codegraph", "codegraph.db")):
            with INIT_LOCK:
                if not os.path.exists(os.path.join(mirror, ".codegraph", "codegraph.db")):
                    codegraph_init(mirror)
        return
    sh(["git", "-C", mirror, "checkout", "--", "."], check=False)
    sh(["git", "-C", mirror, "clean", "-fdq", "-e", ".codegraph"], check=False)
    if condition in ("oh-aoci", "oh-aoci-cg"):
        fill_aoci_mirror(mirror, repo_root)                   # AOCI has write tools: rebuild every time
    cgm = slot.get("cg_mirror")
    if cgm and not os.path.exists(os.path.join(cgm, ".codegraph", "codegraph.db")):
        with INIT_LOCK:
            if not os.path.exists(os.path.join(cgm, ".codegraph", "codegraph.db")):
                codegraph_init(cgm)


def kill_mcp_servers(ws):
    """Reap the MCP servers a conversation spawned for this workspace.

    openhands-sdk 1.1.0 starts a stdio MCP server per conversation and does NOT stop it on
    close(); a 48-question run would otherwise leave 48 orphaned servers holding the index open
    (observed in the oh-cg smoke). Only processes whose command line names THIS worktree and an
    MCP flag are touched, so the session's own MCP servers are never affected.
    """
    ws = os.path.realpath(ws)
    me = os.getpid()
    victims = []
    try:
        out = subprocess.run(["pgrep", "-f", ws], capture_output=True, text=True).stdout.split()
    except FileNotFoundError:
        return 0
    for pid in out:
        try:
            pid = int(pid)
        except ValueError:
            continue
        if pid == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        # 2026-10-14: match the path as a whole argument — `mirrors/x-oh-aoci-1-1` is a prefix of `...-1-10`..`-19`, and
        # the substring test reaped slots 10-19's live servers every time slot 1 finished a question (McpError: Connection closed)
        if ((ws + " ") in cmd or cmd.rstrip().endswith(ws)) and ("--mcp" in cmd or cmd.rstrip().endswith(" mcp")):
            victims.append(pid)
    for pid in victims:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    if victims:
        time.sleep(0.3)
        for pid in victims:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return len(victims)


def workspace_dirty(slot):
    """Did the agent edit / create / delete anything in its export? (it is told not to)"""
    return tree_fingerprint(slot["ws"]) != slot["fp"]


# ---------------------------------------------------------------- terminal isolation
DEAD_PROXY = "http://127.0.0.1:9"        # nothing listens; every proxied request fails fast
SECRET_RE = re.compile(r"(api[_-]?key|secret|token|passw|credential|auth)", re.I)
LEAKY_VARS = {"virtual_env", "pythonpath", "pythonhome", "conda_prefix", "conda_default_env", "aoci_bin",
              "codegraph_bin", "openhands_py", "aoci_eval_worktrees", "oldpwd"}


def sanitize_terminal_env(env, cwd=None):
    """The environment the AGENT'S bash gets. The SDK copies os.environ into the terminal (subprocess_terminal.py:76),
    which handed the agent: our API key (llm.load_profile exports it), the real proxy (DeepSeek fetched GitHub
    issue-search results with curl — 8 calls on one question, k=1b), and paths naming the harness/venv/authoring
    tree. So: no secrets, a dead proxy in both letter cases with NO bypass (localhost included: the dashboard and
    the MCP mirrors are local), no leaky variables, and a PATH stripped of anything under the harness or ~/.local."""
    out = {}
    root = os.path.realpath(common.ROOT)
    for k, v in env.items():
        kl = k.lower()
        if SECRET_RE.search(k) or kl in LEAKY_VARS or kl.endswith("_proxy"):
            continue
        if kl != "path" and isinstance(v, str) and (root in v or "/.venv" in v):
            continue
        out[k] = v
    for k in ("http_proxy", "https_proxy", "all_proxy", "ftp_proxy"):
        out[k] = out[k.upper()] = DEAD_PROXY
    out["no_proxy"] = out["NO_PROXY"] = ""
    keep = []
    for d in (out.get("PATH") or "").split(os.pathsep):
        rd = os.path.realpath(d or ".")
        if rd.startswith(root) or "/.venv" in rd or rd == os.path.realpath(os.path.expanduser("~/.local/bin")):
            continue
        keep.append(d)
    out["PATH"] = os.pathsep.join(keep)
    if cwd:
        # `~` for the agent is the private slot directory (holds only `repo`): `ls ~` no longer lists the user's
        # home — where the harness checkout, and with it artifacts/<repo>/aoci-index.txt, is one guess away
        # (a k=1b baseline record read exactly that file).
        out["HOME"] = os.path.dirname(os.path.realpath(cwd))
    return out


BWRAP = shutil.which("bwrap")


def jail_command(cmd, cwd):
    """Wrap the SDK's `/bin/bash -i` in a bubblewrap jail rooted at the agent's workspace.

    Inside: read-only /usr and /etc (merged-usr symlinks for /bin /sbin /lib /lib64), /proc /dev, tmpfs /tmp and
    /run, the workspace bind-mounted read-write AT ITS OWN PATH, its parent as an empty dir (HOME), and
    --unshare-all: no network at all, a private PID namespace (no other process is visible — no `ps aux |
    grep openhands`, no /proc/<pid>/environ of the adapter), no /home. Sanitised env alone was not enough:
    a confused agent ran `find /home/<user> -newermt …` and then read files it found (k=1d). The jail is
    what makes 'the agent sees its checkout and nothing else' true rather than merely audited."""
    ws = os.path.realpath(cwd)
    slot = os.path.dirname(ws)
    args = [BWRAP, "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run"]
    for p_ in ("/bin", "/sbin", "/lib", "/lib64"):
        if os.path.islink(p_):
            args += ["--symlink", os.readlink(p_), p_]
        elif os.path.isdir(p_):
            args += ["--ro-bind", p_, p_]
    args += ["--dir", slot, "--bind", ws, ws, "--chdir", ws, "--setenv", "HOME", slot, "--"]
    return args + list(cmd)


def patch_terminal_isolation():
    """Force the SDK's subprocess terminal (its env construction is one visible line) and wrap that module's
    Popen so the bash it launches (a) gets sanitize_terminal_env() and (b) runs inside jail_command() when
    bubblewrap is available. Nothing else in the process is touched: our own git/tar/MCP subprocesses and the
    LLM's httpx client keep the real environment. Returns a description for the manifest."""
    from openhands.tools.terminal.terminal import factory, subprocess_terminal
    factory._is_tmux_available = lambda: False        # tmux is present on this box; the SDK would pick it
    real = subprocess_terminal.subprocess
    if getattr(real, "_aoci_sanitized", False):
        return "already applied"

    class _Shim(types.ModuleType):
        _aoci_sanitized = True

        def __getattr__(self, name):
            return getattr(real, name)

        @staticmethod
        def Popen(*a, **kw):
            if kw.get("env") is not None:
                kw["env"] = sanitize_terminal_env(kw["env"], cwd=kw.get("cwd"))
            if BWRAP and kw.get("cwd") and a and isinstance(a[0], (list, tuple)) and os.path.realpath(kw["cwd"]).startswith(os.path.realpath(WORKTREES) + os.sep):
                a = (jail_command(a[0], kw["cwd"]),) + tuple(a[1:])
            return real.Popen(*a, **kw)

    subprocess_terminal.subprocess = _Shim("subprocess")
    return ("subprocess terminal (tmux disabled) + sanitized env (no secrets, dead proxy, no venv/harness paths)"
            + (" + bubblewrap jail (unshare-all: no net/pid/home; ro /usr /etc; rw workspace only)" if BWRAP else " (bwrap NOT found: no jail)"))


def patch_file_editor_confinement():
    """The SDK's file_editor runs IN THE ADAPTER PROCESS (not in the jailed bash) and only uses workspace_root as
    a cwd for suggestions — `view /anything/absolute` succeeds. Confine it: any path that does not resolve
    under the conversation's workspace is rejected with the tool's own parameter error."""
    from openhands.tools.file_editor import editor as _ed
    if getattr(_ed.FileEditor, "_aoci_confined", False):
        return "already applied"
    orig = _ed.FileEditor.validate_path

    def validate_path(self, command, path):
        orig(self, command, path)
        root = os.path.realpath(str(self._cwd)) if self._cwd else None
        if root:
            rp = os.path.realpath(str(path))
            if rp != root and not rp.startswith(root + os.sep):
                raise _ed.EditorToolParameterInvalidError("path", str(path), f"Access denied: only paths under {root} may be opened.")

    _ed.FileEditor.validate_path = validate_path
    _ed.FileEditor._aoci_confined = True
    return "file_editor confined to the workspace"


# ---------------------------------------------------------------- OpenHands
def patch_force_chat_completions():
    """gpt-5.x ids match the SDK's RESPONSES_API_PATTERNS (model_features.py), so the SDK routes
    them to the OpenAI Responses API, whose multi-turn round-trips ENCRYPTED reasoning items
    (rs_...). A reseller gateway forwards under its own upstream credentials, so the encrypted
    items cannot be decrypted on the next turn: 400 'Encrypted content could not be verified'
    (25 fatal records in oh-luna k=1, 2026-08-21). Every profile this harness runs speaks plain
    chat.completions — the minimal host and the 200-way bursts prove the same model works fine
    there — so pin the whole adapter to the chat path. No-op for models that never matched."""
    from openhands.sdk.llm.llm import LLM
    LLM.uses_responses_api = lambda self: False
    return "uses_responses_api pinned False (chat.completions only; reseller gateway cannot round-trip encrypted reasoning items)"


def patch_sdk_union_types():
    """openhands-sdk 1.1.0 crashes on a legal MCP schema that declares an optional property as a
    union (`"type": ["null", "array"]`): tool/schema.py does `if t in _map` with a list. Upstream
    fixed this later (1.42.1's py_type normalizes unions, comment: "MCP schemas often mark optional
    fields this way"); we backport exactly that normalization so the paper-pinned 1.1.0 can load
    such servers. It is a no-op for schemas without unions (CodeGraph's), so it cannot favour an arm.
    Returns a description of what was patched, for the manifest."""
    from openhands.sdk.tool import schema as _schema
    if getattr(_schema, "_aoci_union_patch", False):
        return "already applied"
    try:
        _schema.py_type({"type": ["null", "array"], "items": {"type": "string"}})
        return None                      # this SDK already handles unions
    except TypeError:
        pass
    original = _schema.py_type

    def py_type(spec):
        t = spec.get("type")
        if isinstance(t, (list, tuple, set)):
            non_null = [x for x in t if x != "null"]
            if len(non_null) != 1:
                from typing import Any as _Any
                return _Any
            spec = dict(spec, type=non_null[0])
        return original(spec)

    _schema.py_type = py_type
    _schema._aoci_union_patch = True
    return "backported union-type normalisation from openhands-sdk 1.42.1 into py_type (1.1.0 raises TypeError on MCP union types)"


def build_llm(prof, temperature, max_output_tokens=12288):
    from pydantic import SecretStr
    from openhands.sdk import LLM
    key = os.environ.get(prof["api_key_env"])
    # max_output_tokens explicit: the SDK derives it from LiteLLM's model table, which does not know this
    # gateway's model ids ("model isn't mapped yet"); reasoning models spend part of it on thinking.
    extra = {}
    if prof.get("litellm_extra_body"):
        # r5 parity profiles: per-profile request body extras (GPT-5.1 reasoning_effort=none; OpenRouter provider pin)
        extra["litellm_extra_body"] = dict(prof["litellm_extra_body"])
    if "api.deepseek.com" in str(prof.get("base_url") or ""):
        # DeepSeek 官方直连: flash 偶发进入思考模式后, SDK 多轮回放不带 reasoning_content 会被 400
        # ("must be passed back"); 关闭思考模式 = 与全部成功记录同 regime(成功记录本就无思考输出)。
        extra["litellm_extra_body"] = {"thinking": {"type": "disabled"}}   # enable_thinking 被官方接受但无视, 探针实证唯 thinking.type=disabled 生效
    return LLM(model="openai/" + prof["model"], base_url=prof.get("base_url"), api_key=SecretStr(key),
               temperature=temperature, reasoning_effort=None, drop_params=True, usage_id="agent",
               max_output_tokens=max_output_tokens, **extra)


def mcp_config_for(condition, ws, cg_ws=None):
    aoci = {"aoci": {"command": AOCI_BIN, "args": ["--repo", ws, "mcp"]}}

    def cg(path):
        return {"codegraph": {"command": CG_BIN, "args": ["serve", "--mcp", "-p", path, "--no-watch"], "env": dict(CG_ENV)}}
    if condition == "oh-aoci":
        return {"mcpServers": aoci}
    if condition in ("oh-cg", "oh-inject-cg"):
        return {"mcpServers": cg(ws)}
    if condition == "oh-aoci-cg":                     # AOCI from the private mirror, CodeGraph from the pass-shared mirror
        return {"mcpServers": {**aoci, **cg(cg_ws or ws)}}
    return None


def system_suffix_for(condition):
    """TASK_INSTRUCTION + the arm's one same-shape hint sentence — appended to the agent's system prompt."""
    hint = {"oh-aoci": AOCI_HINT, "oh-cg": CG_HINT, "oh-inject": INJECT_HINT, "oh-file": OH_FILE_HINT,
            "oh-inject-cg": OH_COMBO_HINT, "oh-aoci-cg": OH_AOCI_CG_HINT}.get(condition, BASE_HINT)
    return TASK_INSTRUCTION + hint


def build_agent(model, condition, ws, cg_ws=None):
    from openhands.tools.preset.default import get_default_agent
    from openhands.sdk.context.agent_context import AgentContext
    patch_sdk_union_types()
    patch_sdk_text_limit()
    agent = get_default_agent(model, cli_mode=True)
    upd = {"agent_context": AgentContext(system_message_suffix=system_suffix_for(condition))}
    cfg = mcp_config_for(condition, ws, cg_ws)
    if cfg:
        upd["mcp_config"] = cfg
    return agent.model_copy(update=upd)


# Terminal verbs that open a file. grep is by far the most common one (102/157 baseline commands),
# so leaving it out undercounted files_read by ~1/3 — and unequally across arms.
_FILE_EXT = r"py|pyi|pyx|rst|md|txt|toml|cfg|ini|yml|yaml|json"
PATH_IN_CMD = re.compile(
    r"(?:cat|sed|head|tail|less|more|nl|wc|awk|grep|rg|egrep|fgrep|find|view|python3? -c)\b[^|;&]*?\s"
    r"([\w./-]+\.(?:" + _FILE_EXT + r"))\b")
# Paths a map tool handed back (codegraph_explore returns verbatim source; aoci_* return entry lines).
PATH_IN_MAP = re.compile(r"\b((?:[\w.-]+/)+[\w.-]+\.(?:" + _FILE_EXT + r"))\b")


def llm_call_count(conv):
    stats = conv.state.stats
    m = stats.usage_to_metrics.get("agent") or stats.get_combined_metrics()
    return len(m.token_usages)


def patch_sdk_text_limit():
    """openhands-sdk 1.1.0 middle-out truncates any single TextContent above 50,000 characters when a message is
    serialised (llm/message.py, DEFAULT_TEXT_CONTENT_LIMIT). An aoci_overview chunk at chunk_tokens=24000 is ~72k
    characters, so the paper-pinned SDK would silently amputate every chunk of the cognition turn. The limit is raised
    uniformly for EVERY arm (symmetric; it only matters for outputs above 50k characters) and recorded in the manifest."""
    import openhands.sdk.llm.message as _msg
    import openhands.sdk.utils.truncate as _tr
    if getattr(_msg, "DEFAULT_TEXT_CONTENT_LIMIT", 0) >= SDK_TEXT_LIMIT:
        return "already applied"
    _tr.DEFAULT_TEXT_CONTENT_LIMIT = SDK_TEXT_LIMIT
    _msg.DEFAULT_TEXT_CONTENT_LIMIT = SDK_TEXT_LIMIT
    try:
        import openhands.sdk.utils as _u
        _u.DEFAULT_TEXT_CONTENT_LIMIT = SDK_TEXT_LIMIT
    except Exception:  # noqa: BLE001
        pass
    return f"DEFAULT_TEXT_CONTENT_LIMIT 50000 -> {SDK_TEXT_LIMIT}"


def parse_overview_text(txt):
    """aoci_overview observations reach the model either as a JSON head line or, through the SDK's MCP wrapper, as
    '[Tool ... executed.]AOCI Overview Metadata:' followed by `key: value` lines. Both become one flat dict."""
    body = re.sub(r"^\[[^\]]*\]", "", txt.strip())
    body = re.sub(r"^\[[^\]]*\]", "", body.strip())
    if body.lstrip().startswith("{"):
        try:
            return json.loads(body.lstrip().split("\n", 1)[0])
        except Exception:  # noqa: BLE001
            return {"error": txt[:200]}
    if "AOCI Overview Metadata:" in body:
        out = {}
        for line in body.split("AOCI Overview Metadata:", 1)[1].splitlines():
            m = re.match(r"^([A-Za-z_][\w.]*):\s?(.*)$", line.strip())
            if not m:
                if out and not line.strip():
                    break
                continue
            k, v = m.group(1), m.group(2).strip()
            out[k] = True if v == "true" else False if v == "false" else (int(v) if re.fullmatch(r"-?\d+", v) else v)
        return out
    if "error occurred" in txt[:80].lower() or "validating" in txt[:200]:
        return {"error": txt[:200]}
    return {}


def canon_tool(name):
    """With two MCP servers attached the SDK namespaces tool names by server (aoci_aoci_overview,
    codegraph_codegraph_explore); with one server it does not. Strip the server prefix so arms compare alike."""
    return re.sub(r"^(aoci|codegraph)_(?=\1_)", "", name or "")


def cognition_summary(conv):
    """What the cognition turn actually did: aoci_rules / aoci_overview calls, chunks seen, completed=true observed,
    attestation submitted and the server's verdict on it (from the observation JSON head)."""
    from openhands.sdk.event.llm_convertible import ActionEvent, MessageEvent, ObservationEvent
    out = {"rules_calls": 0, "overview_calls": 0, "chunks_seen": [], "completed_seen": False, "attestation_submitted": False,
           "attestation": None, "final_state": None, "other_tools": 0, "last_message": None, "calls": []}
    pending, by_id = {}, {}
    for e in conv.state.events:
        if isinstance(e, ActionEvent):
            try:
                full = e.action.model_dump()
            except Exception:  # noqa: BLE001
                full = {}
            tool = canon_tool(e.tool_name)
            if tool.startswith("aoci_") and len(out["calls"]) < 24:      # bounded audit trail of the cognition turn
                rec = {"tool": tool, "args": str(full)[:240]}
                out["calls"].append(rec)
                by_id[getattr(e, "id", None)] = rec
            if tool == "aoci_rules":
                out["rules_calls"] += 1
            elif tool == "aoci_overview":
                out["overview_calls"] += 1
                if "model_cognition_attestation" in str(full):
                    out["attestation_submitted"] = True
                    pending[getattr(e, "id", None)] = "attestation"
                else:
                    pending[getattr(e, "id", None)] = "overview"
            elif tool not in ("think", "finish"):
                out["other_tools"] += 1
        elif isinstance(e, ObservationEvent):
            kind = pending.get(getattr(e, "action_id", None))
            if not kind:
                continue
            try:
                txt = "".join(getattr(c, "text", "") for c in e.observation.to_llm_content)
            except Exception:  # noqa: BLE001
                txt = ""
            meta = parse_overview_text(txt)
            rec = by_id.get(getattr(e, "action_id", None))
            if rec is not None:
                rec["obs_chars"], rec["obs_head"] = len(txt), txt[:1200 if kind == "attestation" else 400]
            if kind == "overview":
                if meta.get("chunk_index"):
                    out["chunks_seen"].append([meta.get("chunk_index"), meta.get("chunk_count")])
                if meta.get("completed") is True:
                    out["completed_seen"] = True
            else:
                out["attestation"] = {k: v for k, v in meta.items()
                                      if any(x in k for x in ("attest", "cognition", "verified", "reliable", "level", "mastery", "coverage", "delivery_mode", "error", "message"))}
            if meta.get("cognition_level_state"):
                out["final_state"] = meta["cognition_level_state"]
        elif isinstance(e, MessageEvent) and e.source == "agent":
            txt = "".join(getattr(c, "text", "") for c in e.llm_message.content)
            if txt.strip():
                out["last_message"] = txt.strip()[:120]
    return out


def collect(conv, ws, start=0):
    """Walk the event log (from `start`, i.e. after the cognition turn): actions, files read (by the agent's own
    tools and via the map), map usage, final answer."""
    from openhands.sdk.event.llm_convertible import ActionEvent, MessageEvent, ObservationEvent
    actions, obs, final = [], {}, None
    resp_ids = []  # one LLM iteration can emit several tool calls; steps = distinct LLM responses
    for e in conv.state.events[start:]:
        if isinstance(e, ActionEvent):
            rid = getattr(e, "llm_response_id", None)
            if rid not in resp_ids:
                resp_ids.append(rid)
            act = getattr(e, "action", None)
            full = {}
            if act is not None:
                try:
                    full = {k: v for k, v in act.model_dump().items() if k != "kind"}
                except Exception:  # noqa: BLE001
                    full = {"repr": str(act)}
            # `args` is the trace rendering (bounded); parsing and the final answer must use the FULL
            # values — truncating them silently amputated every answer given through the finish tool.
            args = {k: str(v)[:160] for k, v in full.items()}
            actions.append({"id": getattr(e, "id", None), "step": len(resp_ids), "tool": e.tool_name,
                            "args": args, "full": full})
            if e.tool_name == "finish":
                final = (str(full.get("message") or "").strip() or final)
        elif isinstance(e, ObservationEvent):
            try:
                txt = "".join(getattr(c, "text", "") for c in e.observation.to_llm_content)
            except Exception:  # noqa: BLE001
                txt = ""
            obs[getattr(e, "action_id", None)] = txt
        elif isinstance(e, MessageEvent) and e.source == "agent":
            txt = "".join(getattr(c, "text", "") for c in e.llm_message.content)
            if txt.strip():
                final = txt

    def rel(paths):
        """Repo-relative form. A path under the authoring repository names the SAME file (same pinned
        commit), so it is normalised rather than dropped — otherwise a read outside the workspace would
        silently vanish from files_read. The command itself is still reported by contamination()."""
        roots = [ws] + ([os.path.realpath(REPO_ROOT_HINT[0])] if REPO_ROOT_HINT[0] else [])
        out = []
        for f in paths:
            f = f.strip("'\"")
            for root in roots:
                if root and f.startswith(root + "/"):
                    f = f[len(root) + 1:]
                    break
            if f.startswith("./"):
                f = f[2:]
            if not f.startswith("/") and not f.startswith(".."):
                out.append(f)
        return sorted(set(out))

    trace, read, via_map = [], [], []
    for a in actions:
        text = obs.get(a["id"], "")
        is_map = a["tool"].startswith(MAP_TOOL_PREFIXES)
        full = a["full"]
        trace.append({"step": a["step"], "tool": a["tool"], "args": a["args"], "chars": len(text)})
        if a["tool"] == "file_editor" and str(full.get("command")) == "view" and full.get("path"):
            read.append(str(full["path"]))
        if a["tool"] == "terminal" and full.get("command"):
            read += [m.group(1) for m in PATH_IN_CMD.finditer(str(full["command"]))]   # full command, not the trace copy
        if is_map and text:
            # paths the map NAMED in its output (codegraph_explore mixes snippets with blast-radius
            # mentions; aoci_* return entry lines). An upper bound on "the map showed me this file" —
            # never a read, and never to be summed with files_read.
            via_map += PATH_IN_MAP.findall(text)
    return trace, rel(read), rel(via_map), (final or "").strip(), actions


MAP_ARTEFACT = re.compile(r"aoci\.(?:txt|meta\.txt|code\.txt)|\.aoci\b|\.codegraph\b|AGENTS\.md")


SYSTEM_PREFIXES = ("/usr/", "/bin/", "/sbin/", "/lib", "/etc/", "/dev/", "/proc/", "/sys/", "/opt/", "/run/", "/var/", "/snap/",
                   "/tmp/", "/tmp")   # /tmp is the jail's private tmpfs: scratch there is not an escape
NETWORK_RE = re.compile(r"\b(curl|wget|nc|ncat|ssh|scp|rsync|telnet|pip3?\s+(install|download)|python3?\s+-m\s+pip\b|git\s+(clone|fetch|pull|ls-remote))\b|https?://", re.I)
ENV_PROBE_RE = re.compile(r"(^|[\s;&|])(env|printenv|set)\s*(\||;|$|>)|/proc/\d|\bls\s+(-\w+\s+)?/(tmp|root|home)/?(?=[\s;|&\"']|$)"
                          r"|\bfind\s+/(?:home|root|tmp)?\b(?!\S)|\.bash_history|\bhistory\b", re.M)
ABS_PATH_RE = re.compile(r"(?<![\w@:.-])/(?:[\w.@+~-]+/)*[\w.@+~-]*")
PARENT_ESCAPE_RE = re.compile(r"(?:^|[\s/=\"'])\.\.(?:/|[\s\"';&|]|$)")


def contamination(actions, ws, mirror, repo_root, allow_map=False):
    """Commands in which the agent reached for a map artefact or for a tree that is not its own.

    Reads the FULL action values — the 160-char trace copy would miss anything late in a command.
    'outside-workspace' = the command names the mirror, the authoring tree, the harness root, or ANY absolute
    path that does not resolve under the workspace (system directories excepted); 'parent-escape' = a `..`
    path component. The first version only knew two specific outside trees and let `cd ../.. && cat MEMO…`
    through as clean (k=1 preflight, 2026-08-19)."""
    ws_real = os.path.realpath(ws)
    hits = []
    for a in actions:
        if a["tool"].startswith(MAP_TOOL_PREFIXES):
            continue
        cmd = " ".join(str(v) for v in (a.get("full") or {}).values())
        why = []
        if MAP_ARTEFACT.search(cmd) and not allow_map:
            why.append("map-artefact")     # oh-file: reading the placed volumes IS the treatment, not a leak
        # every absolute path token must resolve INSIDE the workspace (mirror, authoring tree and harness root
        # are all outside it, so a substring list is unnecessary — and a substring check misfires when the
        # workspace itself lives under one of those trees)
        for tok in ABS_PATH_RE.findall(cmd):
            if len(tok) < 2 or tok.startswith(SYSTEM_PREFIXES) or tok in ("/", "/dev/null"):
                continue
            # sed/awk addresses (`/class Foo/,/def bar/p`) and glob fragments (`*/node_modules/*`) look like paths:
            # a token counts only if its first component is a real top-level directory on this host, and only an
            # EXISTING path outside the workspace is an escape (a typo like /home/alkor2002000/… is not)
            top = "/" + tok.strip("/").split("/", 1)[0]
            if not os.path.isdir(top):
                continue
            rp = os.path.realpath(tok)
            if rp != ws_real and not rp.startswith(ws_real + os.sep) and os.path.exists(rp):
                why.append("outside-workspace")
                break
        if PARENT_ESCAPE_RE.search(cmd):
            why.append("parent-escape")
        if NETWORK_RE.search(cmd):
            why.append("network")            # blocked by the dead proxy, but the ATTEMPT is the finding
        if ENV_PROBE_RE.search(cmd):
            why.append("env-probe")          # looking for a hidden task file / secrets / the harness
        if why:
            hits.append({"step": a["step"], "tool": a["tool"], "why": why, "cmd": cmd[:300]})
    return hits


def normalize_paths(text, ws):
    return (text or "").replace(ws + "/", "").replace(ws, ".")


def run_one(prof, condition, slot, repo_root, question, repo_name, max_iter, temperature, index_text):
    from openhands.sdk import Conversation
    ws, mirror = slot["ws"], slot.get("mirror")
    reset_slot(slot, repo_root, condition)
    with INIT_LOCK:  # construct serially; run() below is concurrent
        model = build_llm(prof, temperature, MAX_OUTPUT_TOKENS[0])
        agent = build_agent(model, condition, mirror or ws, slot.get("cg_mirror"))
        conv = Conversation(agent=agent, workspace=ws, max_iteration_per_run=max_iter, visualizer=None, stuck_detection=True)
    t0 = time.time()
    forced, retried, err, delivered = 0, None, None, 0
    cog, cog_boundary, cog_calls = None, 0, 0
    prompt = f"Repository: {repo_name}\n Question: {question}"     # SWE-QA's user turn, verbatim; task text is in the system prompt
    try:
        if condition in COGNITION_ARMS and os.environ.get("AOCI_COGNITION_TURN", "1") != "0":
            # turn 1: establish complete cognition (own budget); turn 2 below is the paper's 10-iteration question turn
            conv.max_iteration_per_run = COGNITION_ITERS
            conv.send_message(COGNITION_PROMPT)
            conv.run()
            cog_boundary = len(conv.state.events)
            cog = cognition_summary(conv)
            cog["wall_s"] = round(time.time() - t0, 1)
            cog_calls = llm_call_count(conv)
            cog["llm_calls"] = cog_calls
            conv.max_iteration_per_run = max_iter
        if condition in ("oh-inject", "oh-inject-cg") and index_text:
            from openhands.sdk.llm.message import Message, TextContent
            # [intro][index part 1..n][Repository/Question] — question LAST (see INJECT_INTRO / TASK_INSTRUCTION notes)
            parts = [INJECT_INTRO] + index_parts(index_text) + [prompt]
            conv.send_message(Message(role="user", content=[TextContent(text=p) for p in parts]))
            delivered = sum(len(p) for p in parts)
        else:
            conv.send_message(prompt)
            delivered = len(prompt)
        conv.run()
        trace, files, via_map, final, acts = collect(conv, ws, cog_boundary)
        if not final:
            # Iteration cap hit without an answer. The paper protocol simply stops here; we allow ONE more
            # LLM step to state the answer (max_iteration_per_run=1 → a tool call would end the round with no
            # text, which is then recorded as such). Recorded as forced=1 so it can be excluded.
            forced = 1
            conv.max_iteration_per_run = 1
            conv.send_message(FORCED_ANSWER)
            conv.run()
            trace, files, via_map, final, acts = collect(conv, ws, cog_boundary)
        reason = common.degenerate_reason(final)
        if reason:
            retried = reason
            conv.max_iteration_per_run = 1  # the symmetric guard is one answer step, never more tool budget
            conv.send_message(common.NUDGE)
            conv.run()
            trace, files, via_map, final2, acts = collect(conv, ws, cog_boundary)
            if final2:
                final = final2
    except Exception as e:  # noqa: BLE001
        err = f"{e.__class__.__name__}: {str(e)[:300]}"
        trace, files, via_map, final, acts = collect(conv, ws, cog_boundary) if conv.state is not None else ([], [], [], "")
    finally:
        try:
            conv.close()
        except Exception:  # noqa: BLE001
            pass
        # Shared-mirror arms (oh-cg / oh-inject-cg): ONE mirror serves the whole pass, so reaping by
        # mirror path here would kill the other 47 workers' live servers mid-call (2026-08-21: 11
        # Connection-closed bursts, 8 fatal records in oh-cg-2). Their servers are reaped once, at
        # pass teardown (remove_slots). Private-mirror arms (oh-aoci) still reap per question.
        killed = kill_mcp_servers(ws if condition in ("oh-cg", "oh-inject-cg") else (mirror or ws))
    # SDK 1.1.0: the default agent's condenser LLM is a model_copy of the agent LLM and SHARES its Metrics
    # object, so get_combined_metrics() counts every call twice. Take the agent's own metrics; fall back to
    # the combined view only if that usage id is missing, and de-duplicate an exactly repeated second half.
    stats = conv.state.stats
    m = stats.usage_to_metrics.get("agent") or stats.get_combined_metrics()
    tu = [(t.prompt_tokens, t.completion_tokens, t.cache_read_tokens) for t in m.token_usages]
    if len(tu) % 2 == 0 and tu and tu[: len(tu) // 2] == tu[len(tu) // 2:]:
        tu = tu[: len(tu) // 2]
    rounds = [{"round": i + 1, "input": a, "output": b, "cache_read": c} for i, (a, b, c) in enumerate(tu)]
    usage = {"input": sum(a for a, _, _ in tu), "output": sum(b for _, b, _ in tu),
             "cache_read": sum(c for _, _, c in tu), "cache_write": 0}
    dirty = workspace_dirty(slot)
    dpaths = dirty_paths(slot) if dirty else []
    map_calls = [t for t in trace if t["tool"].startswith(MAP_TOOL_PREFIXES)]
    contam = contamination(acts, ws, mirror, repo_root, allow_map=(condition == "oh-file"))
    _ = killed  # reaped MCP servers; count is not part of the record
    return {"answer": normalize_paths(final, ws), "stop_reason": str(conv.state.execution_status).split(".")[-1].lower(),
            "prompt_chars_delivered": delivered, "prompt_parts": (len(index_parts(index_text)) + 1) if (condition in ("oh-inject", "oh-inject-cg") and index_text) else 1,
            "index_chars": len(index_text) if index_text else 0,   # assert: delivered >= head + index_chars for inject arms
            "files_map_named": via_map,
            "contamination": contam,
            "map_calls": len(map_calls), "map_chars": sum(t.get("chars", 0) for t in map_calls),
            "map_tools": sorted({t["tool"] for t in map_calls}), "obs_chars": sum(t.get("chars", 0) for t in trace),
            "steps": len({t["step"] for t in trace}), "tool_calls": len(trace),
            "files_read": [f for f in files if not MAP_ARTEFACT.search(f)],           # source files only
            "index_file_reads": [f for f in files if MAP_ARTEFACT.search(f)],         # oh-file: the treatment being used
            "trace": trace, "rounds": rounds, "cognition": cog, "cognition_llm_calls": cog_calls,
            "usage": usage, "retried": retried, "forced": forced, "hit_cap": bool(forced) or len({t["step"] for t in trace}) >= max_iter, "workspace_dirty": bool(dirty), "dirty_paths": dpaths,
            "wall_s": round(time.time() - t0, 1), "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-name", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--condition", choices=["oh", "oh-aoci", "oh-cg", "oh-inject", "oh-mcp", "oh-file", "oh-inject-cg", "oh-aoci-cg"], required=True,
                    help="'oh-mcp' is accepted as an alias of 'oh-aoci'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--max-iterations", type=int, default=10, help="paper: 10 reasoning-action iterations")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--index-artifact", default=None,
                    help="oh-inject: path to the built index artefact (harness/build_index_artifact.py). Required for oh-inject; "
                         "the artefact's sha256 is recorded in the manifest and every record.")
    ap.add_argument("--max-output-tokens", type=int, default=12288, help="LLM max output tokens per call (reasoning included)")
    ap.add_argument("--workers", type=int, default=common.DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--allow-param-drift", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    condition = "oh-aoci" if a.condition == "oh-mcp" else a.condition
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    # symmetry: no arm may call a map CLI from the terminal (MCP servers are launched by absolute path)
    os.environ["PATH"] = os.pathsep.join(
        d for d in os.environ.get("PATH", "").split(os.pathsep)
        if os.path.realpath(d or ".") != os.path.realpath(os.path.expanduser("~/.local/bin")))
    REPO_ROOT_HINT[0] = a.repo_root
    prof = llm.load_profile(a.profile)
    if prof.get("protocol") != "openai":
        sys.exit("openhands_ask drives LiteLLM's OpenAI-compatible route; profile protocol must be 'openai'")
    workers = common.clamp_workers(a.workers)
    try:
        import openhands.sdk as _sdk  # noqa: F401
        from importlib.metadata import version as _v
        sdk_version = _v("openhands-sdk")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"openhands-sdk not importable in this interpreter ({sys.executable}): {e}; use .venv-oh (requirements-openhands.txt)")
    mcp_names = {"oh-aoci": ["mcp:aoci"], "oh-cg": ["mcp:codegraph"], "oh-inject-cg": ["mcp:codegraph"],
                 "oh-aoci-cg": ["mcp:aoci", "mcp:codegraph"]}.get(condition, [])
    MAX_OUTPUT_TOKENS[0] = a.max_output_tokens
    params = {"max_iterations": a.max_iterations, "temperature": a.temperature, "max_output_tokens": a.max_output_tokens,
              "sdk": sdk_version, "workspace": "plain export of the pinned commit (no VCS metadata)",
              "terminal": patch_terminal_isolation(), "file_editor": patch_file_editor_confinement(),
              "tools": ["terminal", "file_editor", "task_tracker", "finish", "think"] + mcp_names}
    # NOT in params: params feed check_param_drift, and a new bookkeeping key refuses every resume
    # against pre-patch records (7 arms refused, 2026-08-21 — the R3 wall-clock-in-params lesson again).
    # The patch is applied here unconditionally; its description goes into the arm record top-level.
    params["sdk_text_limit"] = SDK_TEXT_LIMIT
    params["cognition_turn"] = {"arms": sorted(COGNITION_ARMS), "iters": COGNITION_ITERS,
                                "enabled": os.environ.get("AOCI_COGNITION_TURN", "1") != "0"} if condition in COGNITION_ARMS else None
    llm_path_note = patch_force_chat_completions()
    patch = patch_sdk_union_types() if condition in MIRROR_ARMS else None
    if patch:
        params["sdk_patch"] = patch
        print(f"sdk patch: {patch}", flush=True)
    if condition in ("oh-cg", "oh-inject-cg"):
        if not os.path.exists(CG_BIN):
            sys.exit(f"codegraph binary not found at {CG_BIN}; install it or set CODEGRAPH_BIN")
        params["codegraph_version"] = sh([CG_BIN, "--version"], check=False).stdout.strip() or None
    commit = common.git_sha(a.repo_root, short=False, untracked=False)
    if commit and commit.endswith("-dirty"):
        commit = commit[:-6]
    questions = common.load_questions(a.questions, a.limit)
    done = {q for q, r in common.load_records(a.out, common.valid_answer).items() if common.valid_answer(r)}
    todo = [q for q in questions if q["question"] not in done]
    if todo:
        # A complete file needs no resume, so parameter drift is moot for it — checking anyway made
        # every finished pass refuse (rc=1) after a bookkeeping key changed shape (2026-08-21).
        common.check_param_drift(a.out, {"params": params, "model": prof["model"], "profile": prof["name"], "kind": condition},
                                 allow=a.allow_param_drift)
    index_text, index_fp = "", None
    if condition in ("oh-inject", "oh-inject-cg"):
        # The index is a BUILT artefact: volumes concatenated, absolute directory headers rewritten to
        # repo-relative (pasted verbatim they sent 13/48 agents out of their own worktree), legend prepended,
        # sha256 recorded. Same bytes for every host; see harness/build_index_artifact.py.
        if not a.index_artifact:
            sys.exit(f"{condition} requires --index-artifact (build it with harness/build_index_artifact.py)")
        index_text, index_fp = common.load_artifact(a.index_artifact)
    print(f"condition={condition} profile={prof['name']} model={prof['model']} sdk={sdk_version} workers={workers} "
          f"max_iterations={a.max_iterations} questions={len(questions)} done={len(done)} todo={len(todo)} commit={commit[:12] if commit else None}", flush=True)
    if a.dry_run:
        print("dry-run: prompt =", repr((f"Repository: {a.repo_name}\n Question: <q>\n\n" + TASK_INSTRUCTION)[:200]))
        return
    if not prof["_has_key"]:
        sys.exit(f"key missing for profile {prof['name']}")
    if not todo:
        return
    run_dir = os.path.dirname(os.path.abspath(a.out))
    base = os.path.basename(a.out)
    arm = base[len("answers-"):-len(".jsonl")] if base.startswith("answers-") else condition
    hint = {"oh-aoci": AOCI_HINT, "oh-cg": CG_HINT, "oh-inject": INJECT_HINT, "oh-file": OH_FILE_HINT,
            "oh-inject-cg": OH_COMBO_HINT, "oh-aoci-cg": OH_AOCI_CG_HINT}.get(condition, BASE_HINT)
    common.write_manifest(run_dir, {"arms": {arm: {
        "kind": condition, "host": "openhands", "profile": prof["name"], "model": prof["model"], "protocol": "openai",
        "params": params, "workers": workers, "questions_sha256": common.sha256_file(a.questions)[:16],
        "prompt_sha256": common.sha256_text(system_suffix_for(condition) + "|Repository: X\n Question: Y")[:16], "hint_chars": len(hint),
        "task_delivery": "system_message_suffix (SWE-QA text in the system prompt; user turn = Repository/Question)",
        "index_chars": len(index_text), "index_parts": len(index_parts(index_text)) if index_text else 0,
        "text_part_limit": TEXT_PART_LIMIT, "sdk_text_content_limit": 50000,
        "workspace_extras": [],   # every arm's workspace is a pristine checkout; maps live in a mirror
        "llm_path": llm_path_note,
        "map_served_from": "mirror worktree" if condition in MIRROR_ARMS else None,
        "index": index_fp if condition in ("oh-inject", "oh-inject-cg") else (common.index_fingerprint(a.repo_root) if condition == "oh-aoci" else None)}}})

    import analyze
    price = analyze.price_for(analyze.load_prices(), prof["model"])
    budget = common.Budget(a.max_usd, (price["in"], price["out"]) if price else None)
    n_ws = min(workers, len(todo))
    # 2026-10-14: namespace every workspace/mirror by repo so concurrent runs of different repos cannot collide
    # (r4 first launch: xarray/reflex/streamlink passes shared <WORKTREES>/oh-aoci-1-<i> and tore each other down)
    slots, prep = make_slots(a.repo_root, f"{a.repo_name}-{arm}", n_ws, commit, condition)
    if prep:
        common.write_manifest(run_dir, {"arms": {arm: prep}})   # deep-merged: keeps the arm record
    pool = queue.Queue()
    for sl in slots:
        pool.put(sl)

    def work(q):
        rec = {"qid": common.qid(q["question"]), "repo": a.repo_name, "condition": arm, "kind": condition, "host": "openhands",
               "profile": prof["name"], "model": prof["model"], "protocol": "openai", "params": params,
               "question": q["question"], "reference": q.get("answer"),
               "index_sha256": index_fp["artifact_sha256"] if index_fp else None}
        if budget.exceeded():
            rec.update(answer="", error=f"budget: est ${budget.spent:.2f} >= cap ${a.max_usd}", usage=None, wall_s=0)
            return rec
        sl = pool.get()
        try:
            refusals, spent = 0, {"input": 0, "output": 0, "cache_read": 0}
            for _attempt in range(1 + common.MAX_REFUSAL_RERUNS):
                out = run_one(prof, condition, sl, a.repo_root, q["question"], a.repo_name, a.max_iterations, a.temperature, index_text)
                if not common.refusal_reason(out.get("answer")):
                    break
                refusals += 1          # fresh conversation next round; the last attempt's answer is kept either way
                for k in spent:        # the discarded attempt cost real tokens: they go into the record's usage
                    spent[k] += (out.get("usage") or {}).get(k, 0) or 0
            out["refusals"] = refusals
            if refusals and out.get("usage"):
                out["refusal_usage"] = dict(spent)
                for k in spent:
                    out["usage"][k] = (out["usage"].get(k, 0) or 0) + spent[k]
            rec.update(out)
        except Exception as e:  # noqa: BLE001
            rec.update(answer="", error=f"{e.__class__.__name__}: {str(e)[:300]}")
        finally:
            pool.put(sl)
        budget.add(rec.get("usage"))
        return rec

    def on_result(q, rec, i, n):
        if "question" not in rec:
            rec = {"qid": common.qid(q["question"]), "question": q["question"], "condition": arm, "answer": "", **rec}
        common.append_jsonl(a.out, rec)
        u = rec.get("usage") or {}
        print(f"[{i}/{n}] {rec.get('wall_s')}s steps={rec.get('steps')} calls={rec.get('tool_calls')} "
              f"files={len(rec.get('files_read') or [])}+{len(rec.get('files_via_map') or [])}m "
              f"map={rec.get('map_calls')}/{rec.get('map_chars')}c "
              f"in={u.get('input')} out={u.get('output')} status={rec.get('stop_reason')} "
              f"{'forced ' if rec.get('forced') else ''}{('retried=' + rec['retried']) if rec.get('retried') else ''} "
              f"{'DIRTY ' if rec.get('workspace_dirty') else ''}"
              f"{('CONTAM=' + str(len(rec['contamination']))) if rec.get('contamination') else ''} "
              f"{rec.get('error') or ''}", flush=True)

    try:
        common.run_parallel(todo, work, workers=n_ws, on_result=on_result, label="oh", progress_path=common.progress_path_for(a.out))
    finally:
        remove_slots(a.repo_root, slots)
    # Every record is on disk and every slot is torn down. openhands-sdk 1.1.0 leaves non-daemon threads behind
    # (MCP client readers, ~30 after 48 conversations) that keep the interpreter from exiting — the k=1e oh-cg
    # arm sat idle for 13+ minutes after its last question, and one such hang would stall a 70-pass run. Do
    # not wait for them.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
