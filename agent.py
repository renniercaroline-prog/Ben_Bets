#!/usr/bin/env python3
"""
agent.py — the AI layer (OpenAI Responses API + web search).

It does NOT compute any probability. Given two teams, it uses GPT-5.5 with the
hosted web_search tool to read the latest team news and return, as strict JSON,
each side's projected starters + minutes and anyone ruled out. model.py turns
those minutes into player-prop probabilities.

No OPENAI_API_KEY set -> returns a transparent mock projection so the rest of
the pipeline still runs end to end.
"""
import os, json, urllib.request

KEY   = os.environ.get("OPENAI_API_KEY", "").strip()
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")   # swap to "gpt-5" if you prefer
URL   = "https://api.openai.com/v1/responses"

INSTRUCTIONS = (
  "You assist a sports STATISTICAL model. You never estimate odds or probabilities. "
  "Use web search to find the latest confirmed or projected lineup and injury news "
  "for the match. Return ONLY strict JSON, no prose, no code fences, in this exact shape:\n"
  '{"home":{"starters":[{"name":str,"min":int}],"out":[str]},'
  '"away":{"starters":[{"name":str,"min":int}],"out":[str]},"note":str}\n'
  "min is projected minutes (90 for a likely full start, less for rotation risk, "
  "omit a player rather than list 0). Keep note under 20 words."
)

def _call_llm(home, away, kickoff):
    body = json.dumps({
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "reasoning": {"effort": "low"},
        "input": f"Match: {home} vs {away}, kickoff {kickoff}. "
                 f"Project both starting XIs with minutes and list anyone ruled out.",
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.load(r)
    # Responses API: pull text out of the typed output array
    text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text += c.get("text", "")
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
    return {"home": {"starters": [], "out": []},
            "away": {"starters": [], "out": []},
            "note": "Sample mode — add OPENAI_API_KEY for live lineup projection."}

def apply_minutes(player_pool, projection_side):
    """Merge agent-projected minutes into the per-90 player pool.
    Ruled-out players are dropped; unlisted players get reduced minutes."""
    proj = {s["name"]: s["min"] for s in projection_side.get("starters", [])}
    out  = set(projection_side.get("out", []))
    merged = []
    for p in player_pool:
        if p["name"] in out:
            continue
        p = dict(p)
        p["min"] = proj.get(p["name"], 25)
        if p["min"] > 0:
            merged.append(p)
    return merged
