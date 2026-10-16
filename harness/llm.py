#!/usr/bin/env python3
"""Model endpoint profiles for the evaluation harness.

Experiments switch endpoints and model ids often, so both live in named
profiles rather than in code or on the command line:

  config/profiles.json   committed  — name → {protocol, base_url, model, api_key_env}
  config/secrets.env     gitignored — KEY_NAME=value lines the profiles refer to

A profile never holds a key, only the *name* of the environment variable that
holds it. secrets.env is loaded into the environment if the variable is not
already set, so the same profile works from a shell that exported the key and
from one that did not. Nothing here prints a key.

Two wire protocols are supported, because gateways differ:
  anthropic  Anthropic Messages API (native SDK, prompt caching, effort/thinking)
  openai     OpenAI-compatible chat completions (OneAPI, OpenRouter, vLLM, ...)
  auto       probe /v1/messages first, then /v1/chat/completions; `test` records the winner

CLI:
  llm.py list
  llm.py add NAME --base-url URL --model ID --api-key-env VAR [--protocol auto|anthropic|openai]
  llm.py use NAME                 # set default profile
  llm.py test [NAME]              # one tiny real request; reports protocol, latency, usage
  llm.py remove NAME

Library:
  p = load_profile(name_or_None)          # resolves default; loads secrets.env
  r = complete(p, system_blocks, user_text, max_tokens=..., effort=..., thinking=..., temperature=...)
  # system_blocks may be [] (no system message); user_text may be a str or a full messages list
  r -> {"text", "stop_reason", "usage": {input, output, cache_read, cache_write}, "model", "protocol"}
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.environ.get("AOCI_EVAL_CONFIG_DIR") or os.path.join(ROOT, "config")  # tests point this at a temp dir
PROFILES = os.path.join(CONFIG_DIR, "profiles.json")
SECRETS = os.path.join(CONFIG_DIR, "secrets.env")


# ---------------------------------------------------------------- storage
def _read():
    if not os.path.exists(PROFILES):
        return {"default": None, "profiles": {}}
    return json.load(open(PROFILES, encoding="utf-8"))


def _write(cfg):
    os.makedirs(os.path.dirname(PROFILES), exist_ok=True)
    json.dump(cfg, open(PROFILES, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def load_secrets():
    """KEY=value lines; does not override variables already in the environment."""
    if not os.path.exists(SECRETS):
        return
    for line in open(SECRETS, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def load_profile(name=None):
    cfg = _read()
    name = name or cfg.get("default")
    if not name or name not in cfg["profiles"]:
        sys.exit(f"llm: no profile {name!r}; run `llm.py list` / `llm.py add`")
    p = dict(cfg["profiles"][name])
    p["name"] = name
    load_secrets()
    key = os.environ.get(p.get("api_key_env") or "")
    p["_has_key"] = bool(key)
    return p


# ---------------------------------------------------------------- clients (one per profile, shared by threads)
_clients = {}


def _anthropic_client(p):
    import anthropic
    k = ("anthropic", p["name"], p.get("base_url"))
    if k not in _clients:
        _clients[k] = anthropic.Anthropic(api_key=os.environ.get(p["api_key_env"]), base_url=p.get("base_url") or None,
                                          max_retries=2, timeout=600)
    return _clients[k]


def _openai_client(p):
    import openai
    k = ("openai", p["name"], p.get("base_url"))
    if k not in _clients:
        _clients[k] = openai.OpenAI(api_key=os.environ.get(p["api_key_env"]), base_url=p.get("base_url") or None,
                                    max_retries=2, timeout=600)
    _clients[k]._aoci_extra_body = p.get("extra_body") or None   # profile-level request extras(如 DashScope enable_thinking=false),agent_ask 同读
    return _clients[k]


def _blocks_to_text(system_blocks):
    return "\n\n".join(b["text"] for b in system_blocks)


def _messages(user_text):
    """user_text may be a string (single user turn) or a full messages list."""
    return user_text if isinstance(user_text, list) else [{"role": "user", "content": user_text}]


def complete_anthropic(p, system_blocks, user_text, max_tokens=4096, effort=None, thinking=None, temperature=None, extra=None):
    client = _anthropic_client(p)
    kwargs = dict(model=p["model"], max_tokens=max_tokens, messages=_messages(user_text))
    if system_blocks:
        kwargs["system"] = system_blocks
    if temperature is not None:
        kwargs["temperature"] = temperature
    if effort:
        kwargs["output_config"] = {"effort": effort}
    if thinking:
        kwargs["thinking"] = thinking
    if extra:
        kwargs.update(extra)
    try:
        resp = client.messages.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        # 2026-10-13: Claude 5.5 系(sonnet-5-5 等)拒绝 temperature("`temperature` is deprecated");去掉后重发一次,其它 400 原样抛出
        if "temperature" in kwargs and "temperature" in str(e):
            kwargs.pop("temperature")
            resp = client.messages.create(**kwargs)
        else:
            raise
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    u = resp.usage
    return {"text": text.strip(), "stop_reason": resp.stop_reason, "model": resp.model, "protocol": "anthropic",
            "usage": {"input": u.input_tokens, "output": u.output_tokens,
                      "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                      "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}}


def complete_openai(p, system_blocks, user_text, max_tokens=4096, effort=None, thinking=None, temperature=None, extra=None):
    client = _openai_client(p)
    msgs = ([{"role": "system", "content": _blocks_to_text(system_blocks)}] if system_blocks else []) + _messages(user_text)
    kwargs = dict(model=p["model"], messages=msgs, max_tokens=max_tokens)
    if str(p.get("model", "")).startswith(("gpt-5", "o1", "o3", "o4")) or p.get("max_completion_tokens"):
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")   # 2026-10-16: GPT-5 family rejects max_tokens (r5 parity preflight)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if "api.deepseek.com" in (p.get("base_url") or ""):
        # 官方 flash 偶发思考模式:全部 token 进 reasoning_content、正文空(finish=length)。
        # enable_thinking 被官方接受但无视;唯 thinking.disabled 实证生效(2026-08-26 四参数探针,agent_ask 同款)。
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    if p.get("extra_body"):                       # profiles.json 里的 extra_body(第三方托管的钉版模型需要各自的关思考参数)
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **p["extra_body"]}
    if extra:
        kwargs.update(extra)
    resp = client.chat.completions.create(**kwargs)
    if not getattr(resp, "choices", None):
        # 网关偶发返回 HTTP 200 但没有 choices 的响应体(2026-09-22 A′ 批 sphinx-b2 两次:上游错误被网关包成 200)。
        # 当成连接故障走 complete() 的退避重试,而不是 TypeError 记成本条作答的错误。
        raise ConnectionError("gateway returned a completion without choices: %s" % str(getattr(resp, "error", None) or resp)[:200])
    choice = resp.choices[0]
    u = resp.usage
    details = getattr(u, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details else 0
    return {"text": (choice.message.content or "").strip(), "stop_reason": choice.finish_reason,
            "model": resp.model, "protocol": "openai",
            "usage": {"input": (u.prompt_tokens or 0) if u else 0, "output": (u.completion_tokens or 0) if u else 0,
                      "cache_read": cached or 0, "cache_write": 0}}


def _retryable(e):
    name = e.__class__.__name__
    msg = str(e).lower()
    return ("connection" in name.lower() or "timeout" in name.lower() or "ratelimit" in name.lower()
            or "internalserver" in name.lower() or "429" in msg or "overloaded" in msg or "502" in msg or "503" in msg or "504" in msg)


def complete(p, system_blocks, user_text, retries=4, **kw):
    """complete_once with exponential backoff on transient failures (5s, 15s, 45s, 135s)."""
    delay = 5
    for attempt in range(retries + 1):
        try:
            return complete_once(p, system_blocks, user_text, **kw)
        except Exception as e:  # noqa: BLE001
            if attempt >= retries or not _retryable(e):
                raise
            time.sleep(delay)
            delay *= 3


def complete_once(p, system_blocks, user_text, **kw):
    proto = p.get("protocol", "auto")
    if proto == "anthropic":
        return complete_anthropic(p, system_blocks, user_text, **kw)
    if proto == "openai":
        return complete_openai(p, system_blocks, user_text, **kw)
    # auto: try native first, fall back to chat completions
    try:
        return complete_anthropic(p, system_blocks, user_text, **kw)
    except Exception as e_a:  # noqa: BLE001 — probing; the second attempt reports if both fail
        try:
            return complete_openai(p, system_blocks, user_text, **kw)
        except Exception as e_o:  # noqa: BLE001
            raise RuntimeError(f"both protocols failed: anthropic: {e_a.__class__.__name__}: {str(e_a)[:160]} | "
                               f"openai: {e_o.__class__.__name__}: {str(e_o)[:160]}") from e_o


# ---------------------------------------------------------------- CLI
def cmd_list(_):
    cfg = _read()
    load_secrets()
    if not cfg["profiles"]:
        print("(no profiles)")
    for name, p in cfg["profiles"].items():
        mark = "*" if name == cfg.get("default") else " "
        has = "key:set" if os.environ.get(p.get("api_key_env") or "") else f"key:MISSING ({p.get('api_key_env')})"
        print(f"{mark} {name:22} {p.get('protocol','auto'):9} {p['model']:32} {p.get('base_url') or '(anthropic default)'}  {has}")


def cmd_add(a):
    cfg = _read()
    cfg["profiles"][a.name] = {"protocol": a.protocol, "base_url": a.base_url, "model": a.model,
                               "api_key_env": a.api_key_env}
    if not cfg.get("default"):
        cfg["default"] = a.name
    _write(cfg)
    print(f"saved profile {a.name}" + (" (default)" if cfg["default"] == a.name else ""))
    if not os.path.exists(SECRETS):
        with open(SECRETS, "w", encoding="utf-8") as f:
            f.write("# KEY_NAME=value — one per line; this file is gitignored\n")
        os.chmod(SECRETS, 0o600)
    load_secrets()
    if not os.environ.get(a.api_key_env):
        print(f"key not set: add a line `{a.api_key_env}=...` to {SECRETS} (or export it)")


def cmd_use(a):
    cfg = _read()
    if a.name not in cfg["profiles"]:
        sys.exit(f"no profile {a.name!r}")
    cfg["default"] = a.name
    _write(cfg)
    print(f"default profile = {a.name}")


def cmd_remove(a):
    cfg = _read()
    cfg["profiles"].pop(a.name, None)
    if cfg.get("default") == a.name:
        cfg["default"] = next(iter(cfg["profiles"]), None)
    _write(cfg)
    print(f"removed {a.name}")


def cmd_test(a):
    p = load_profile(a.name)
    print(f"profile={p['name']} protocol={p.get('protocol','auto')} model={p['model']} base_url={p.get('base_url')}")
    if not p["_has_key"]:
        sys.exit(f"key missing: set {p['api_key_env']} in {SECRETS} or export it")
    system = [{"type": "text", "text": "You are a terse assistant."}]
    t0 = time.time()
    try:
        r = complete(p, system, "Reply with exactly the single word: OK", max_tokens=16)
    except Exception as e:  # noqa: BLE001 — surface the gateway's own words
        sys.exit(f"FAIL after {time.time()-t0:.1f}s: {e.__class__.__name__}: {str(e)[:400]}")
    dt = time.time() - t0
    print(f"OK  {dt:.1f}s  protocol={r['protocol']}  served_model={r['model']}  stop={r['stop_reason']}  "
          f"usage={r['usage']}  text={r['text']!r}")
    # remember the protocol that worked so later calls do not probe twice
    cfg = _read()
    if cfg["profiles"][p["name"]].get("protocol", "auto") == "auto":
        cfg["profiles"][p["name"]]["protocol"] = r["protocol"]
        _write(cfg)
        print(f"recorded protocol={r['protocol']} into profile {p['name']}")
    if r["protocol"] == "openai":
        print("note: OpenAI-compatible route — no Anthropic prompt caching, no effort/thinking controls; "
              "the index is re-sent in full on every question")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    s = sub.add_parser("add"); s.add_argument("name"); s.add_argument("--base-url", default=None)
    s.add_argument("--model", required=True); s.add_argument("--api-key-env", required=True)
    s.add_argument("--protocol", default="auto", choices=["auto", "anthropic", "openai"]); s.set_defaults(fn=cmd_add)
    s = sub.add_parser("use"); s.add_argument("name"); s.set_defaults(fn=cmd_use)
    s = sub.add_parser("remove"); s.add_argument("name"); s.set_defaults(fn=cmd_remove)
    s = sub.add_parser("test"); s.add_argument("name", nargs="?"); s.set_defaults(fn=cmd_test)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
