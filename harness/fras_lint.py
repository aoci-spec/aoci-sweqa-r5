#!/usr/bin/env python3
"""Pre-submission linter for AOCI FRAS v2 entries.

Measures every limit the machine will enforce so a batch is not bounced for
arithmetic. Limits are the FRAS v2 machine contract as published by aoci-code
(F<=160 runes; R<=360 runes/8 items; A<=400 runes/6 items; S rune quota by C
band 9-8<=600, 7-4<=200, 3-1<=50) plus the S token band (bytes/3) which is
usually the binding constraint on C5-C9: C9=200, C8=140, C5-7=80, C3=40.

Usage:
  fras_lint.py ENTRIES.jsonl        # each line {"path":..., "entry":...}
  echo 'name[CG7S]: F:... | R:- | A:- | S:...' | fras_lint.py -
Exit 1 if any entry violates a limit.
"""
import json
import re
import sys

ENTRY_RE = re.compile(r"^(?P<name>[^\[\]]+)\[(?P<tag>[A-Z0-9]+)\]: F:(?P<F>.*?) \| R:(?P<R>.*?) \| A:(?P<A>.*?) \| S:(?P<S>.*)$")


def s_rune_quota(c):
    return 600 if c >= 8 else 200 if c >= 4 else 50


def s_token_band(c):
    return 200 if c == 9 else 140 if c == 8 else 80 if c >= 5 else 40


def tokens(s):
    return (len(s.encode("utf-8")) + 2) // 3


TAG_RE = re.compile(r"^(?P<A>[A-Z])(?P<B>[A-Z])(?P<C>[1-9])(?P<D>[A-Z]?)(?P<E>[LMST])$")


def lint_tag(tag, tagdict=None):
    """Compact tag = A layer + B module + C importance + [D] + E scale; letters must be in the
    repository's own dictionary (aoci.meta.txt) when one is given."""
    m = TAG_RE.match(tag)
    if not m:
        return [f"tag {tag!r} is not A+B+C+[D]+E (e.g. DB7M)"]
    errs = []
    if tagdict:
        if m.group("A") not in tagdict.get("A", {}):
            errs.append(f"tag layer {m.group('A')!r} not in dictionary {sorted(tagdict['A'])}")
        if m.group("B") not in tagdict.get("B", {}):
            errs.append(f"tag module {m.group('B')!r} not in dictionary {sorted(tagdict['B'])}")
    return errs


def lint(entry, tagdict=None):
    m = ENTRY_RE.match(entry)
    if not m:
        return ["entry does not match single-line FRAS form"]
    tag = m.group("tag")
    digits = re.findall(r"\d", tag)
    if not digits:
        return [f"tag {tag!r} has no C digit"]
    c = int(digits[0])
    errs = lint_tag(tag, tagdict)
    F, R, A, S = (m.group(k) for k in "FRAS")
    if len(F) > 160:
        errs.append(f"F {len(F)} runes > 160")
    if R != "-":
        items = R.split(",")
        if len(R) > 360:
            errs.append(f"R {len(R)} runes > 360")
        if len(items) > 8:
            errs.append(f"R {len(items)} items > 8")
        bad = [i for i in items if not (i.startswith("code:") or i.startswith("database://"))]
        if bad:
            errs.append(f"R has non-canonical refs: {bad[:3]}")
    if A != "-":
        if len(A) > 400:
            errs.append(f"A {len(A)} runes > 400")
        if len(A.split(",")) > 6:
            errs.append(f"A {len(A.split(','))} items > 6")
    if S != "-":
        q, t = s_rune_quota(c), s_token_band(c)
        if len(S) > q:
            errs.append(f"S {len(S)} runes > C{c} quota {q}")
        if tokens(S) > t:
            errs.append(f"S {tokens(S)} tok > C{c} band {t}")
    return errs


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "-"
    lines = sys.stdin if src == "-" else open(src, encoding="utf-8")
    bad = 0
    for n, line in enumerate(lines, 1):
        line = line.rstrip("\n")
        if not line:
            continue
        entry = json.loads(line)["entry"] if line.startswith("{") else line
        errs = lint(entry)
        m = ENTRY_RE.match(entry)
        label = m.group("name") if m else f"line {n}"
        if errs:
            bad += 1
            print(f"FAIL {label}: " + "; ".join(errs))
        else:
            S = m.group("S")
            c = int(re.findall(r"\d", m.group("tag"))[0])
            print(f"ok   {label}: F={len(m.group('F'))} S={len(S)}r/{tokens(S)}t (C{c} band {s_rune_quota(c)}r/{s_token_band(c)}t)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
