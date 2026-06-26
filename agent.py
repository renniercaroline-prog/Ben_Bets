#!/usr/bin/env python3
"""
agent.py — the AI layer. It does NOT compute any probability.

Given two teams and their squads, it uses an LLM (with web search) to read the
latest team news and return, as strict JSON:
  - projected starters and their projected minutes for each side
  - any players ruled out
  - a one-line note

model.py turns those minutes into player-prop probabilities. The LLM's job is
judgment about availability and roles, nothing more.

No ANTHROPIC_API_KEY set -> returns a transparent mock projection so the rest of
the pipeline runs end to end.
"""
import os, json, urllib.request

KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
URL = "https://api.anthropic.com/v1/messages"

SYSTEM = (
  "You assist a sports STATISTICAL model. You never estimate odds or probabilities. "
  "Use web search to find the latest confirmed or projected lineup and injury news "
  "for the match. Return ONLY strict JSON, no prose, in this exact shape:\n"
  '{"home":{"starters":[{"name":str,"min":int}],"out":[str]},'
  '"away":{"starters":[{"name":str,"min":int}],"out":[str]},"note":str}\n'
  "min is projected minutes (90 for a likely full start, less for rotation risk, "
  "0 only if you would not list them). Keep note under 20 words."
)

def _call_llm(home, away, kickoff):
    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1500,
        "system": SYSTEM,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [{"role": "user",
            "content": f"Match: {home} vs {away}, kickoff {kickoff}. "
                       f"Project both starting XIs with minutes and list anyone ruled out."}],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "x-api-key": KEY, "anthropic-version": "2023-06-01",
        "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)

def project(home, away, kickoff):
    if not KEY:
        return _mock(home, away)
    try:
        return _call_llm(home, away, kickoff)
    except Exception as e:
        m = _mock(home, away)
        m["note"] = f"(agent fell back to defaults: {e})"
        return m

def _mock(home, away):
    return {
        "home": {"starters": [], "out": []},
        "away": {"starters": [], "out": []},
        "note": "Sample mode — add ANTHROPIC_API_KEY for live lineup projection.",
    }

def apply_minutes(player_pool, projection_side):
    """Merge agent-projected minutes into the per-90 player rate pool.
    Players not projected to start get reduced minutes; ruled-out get 0."""
    proj = {s["name"]: s["min"] for s in projection_side.get("starters", [])}
    out  = set(projection_side.get("out", []))
    merged = []
    for p in player_pool:
        if p["name"] in out:
            continue
        p = dict(p)
        p["min"] = proj.get(p["name"], 25 if p["name"] not in proj else p.get("min", 0))
        if p["min"] > 0:
            merged.append(p)
    return merged
