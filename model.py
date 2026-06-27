#!/usr/bin/env python3
"""
model.py — every probability the site shows. Pure math, no AI, no network.

Team markets are derived from a goals model (Poisson) and a corners model
(Negative Binomial). Player props are per-90 rates scaled by projected minutes
(the minutes come from agent.py, which reads team news).
"""
import os
from scipy.stats import poisson, nbinom

# These four are the model's calibration constants. They are env-overridable so
# backtest.py can A/B re-baselined values (Phase 1.3) without editing code, per the
# "read config from env, never hardcode" rule. Defaults are international-football
# sanity values; re-fit them on real data with `recalibrate()` in backtest.py.
MU_GOALS           = float(os.environ.get("MU_GOALS", "1.35"))   # avg goals per team
LEAGUE_AVG_CORNERS = float(os.environ.get("LEAGUE_AVG_CORNERS", "5.1"))
CORNER_R           = float(os.environ.get("CORNER_R", "7.0"))    # NB dispersion for corners
H1_SHARE           = float(os.environ.get("H1_SHARE", "0.45"))   # ~45% of goals fall in the first half
LEAGUE_AVG_SOT     = float(os.environ.get("LEAGUE_AVG_SOT", "4.5"))  # shots on target / team / game
# Dixon-Coles low-score dependence. In theory a small negative rho corrects the
# four lowest-score cells (independent Poisson under-predicts draws and 0-0/1-1).
# BUT: backtested on real Premier League data (exact-scoreline likelihood, see
# backtest.py --recalibrate) the data did NOT support a nonzero rho — NLL was
# flat-to-worse as rho went negative — so we DEFAULT TO 0 (plain Poisson) rather
# than ship an un-evidenced edge. The machinery stays; re-fit per competition and
# set RHO_DC via env if a league/tournament actually shows the dependence.
RHO_DC             = float(os.environ.get("RHO_DC", "0.0"))
GMAX = 8                             # goal grid cap

# ---------------------------------------------------------------- goals
def goals(home, away):
    lh = MU_GOALS * home["atk"] * away["def"]
    la = MU_GOALS * away["atk"] * home["def"]
    return lh, la

def _pmf(lmbda, n=GMAX):
    return [poisson.pmf(i, lmbda) for i in range(n + 1)]

def _dc_tau(i, j, lh, la, rho):
    """Dixon-Coles adjustment for the four lowest-score cells (1 elsewhere)."""
    if   i == 0 and j == 0: return 1 - lh * la * rho
    elif i == 0 and j == 1: return 1 + lh * rho
    elif i == 1 and j == 0: return 1 + la * rho
    elif i == 1 and j == 1: return 1 - rho
    return 1.0

def score_matrix(lh, la, rho=None):
    """Full-time scoreline grid with the Dixon-Coles low-score correction applied
    and renormalised to sum to 1. Every matrix-derived market (result, over/under,
    BTTS, halves, HT/FT) reads off this one grid, so prices stay mutually consistent."""
    if rho is None:
        rho = RHO_DC
    ph, pa = _pmf(lh), _pmf(la)
    M = [[ph[i] * pa[j] * _dc_tau(i, j, lh, la, rho)
          for j in range(GMAX + 1)] for i in range(GMAX + 1)]
    s = sum(M[i][j] for i in range(GMAX+1) for j in range(GMAX+1))
    return [[M[i][j] / s for j in range(GMAX + 1)] for i in range(GMAX + 1)]

# ---- markets off the full-time score matrix ----
def match_result(M):
    h = sum(M[i][j] for i in range(GMAX+1) for j in range(GMAX+1) if i > j)
    d = sum(M[i][i] for i in range(GMAX+1))
    a = 1 - h - d
    return h, d, a

def over_under_m(M, line):
    """P(total goals > line) read off the score matrix (DC-consistent)."""
    return float(sum(M[i][j] for i in range(GMAX+1) for j in range(GMAX+1) if i+j > line))

def btts_m(M):
    """P(both teams score) read off the score matrix (DC-consistent)."""
    return float(sum(M[i][j] for i in range(1, GMAX+1) for j in range(1, GMAX+1)))

# closed-form wrappers (build their own matrix) so external callers stay simple
def over_under(lh, la, line):
    return over_under_m(score_matrix(lh, la), line)

def btts(lh, la):
    return btts_m(score_matrix(lh, la))

# ---------------------------------------------------------------- halves
def _half_lambdas(lh, la):
    return (lh*H1_SHARE, la*H1_SHARE, lh*(1-H1_SHARE), la*(1-H1_SHARE))

def half_result(lh, la):
    l1h, l1a, _, _ = _half_lambdas(lh, la)
    M = score_matrix(l1h, l1a)
    return match_result(M)

def htft(lh, la):
    """9 half-time/full-time combinations, as {'H/H':p, 'H/D':p, ...}."""
    l1h, l1a, l2h, l2a = _half_lambdas(lh, la)
    p1h, p1a = _pmf(l1h, 6), _pmf(l1a, 6)
    p2h, p2a = _pmf(l2h, 6), _pmf(l2a, 6)
    res = {f"{x}/{y}": 0.0 for x in "HDA" for y in "HDA"}
    sign = lambda d: "H" if d > 0 else ("A" if d < 0 else "D")
    for a in range(7):
        for b in range(7):
            ht = sign(a - b)
            for c in range(7):
                for d in range(7):
                    ft = sign((a + c) - (b + d))
                    res[f"{ht}/{ft}"] += p1h[a]*p1a[b]*p2h[c]*p2a[d]
    return res

# ---------------------------------------------------------------- corners
def _gstate(supremacy):
    return (1 + 0.10*(1 if supremacy > 0 else -1)*min(1, abs(supremacy))) \
         * (1 + 0.18*max(0.0, -supremacy))

def team_corners(att, dfn, supremacy):
    return LEAGUE_AVG_CORNERS * att["catk"] * dfn["ccon"] * _gstate(supremacy)

def _nb(lmbda, r=CORNER_R):
    return [nbinom.pmf(k, r, r/(r+lmbda)) for k in range(31)]

def team_corner_tail(lmbda, k):
    return float(1 - nbinom.cdf(k-1, CORNER_R, CORNER_R/(CORNER_R+lmbda)))

def total_corners_over(lh_c, la_c, line):
    """P(total match corners > line), convolving the two NB distributions."""
    a, b = _nb(lh_c), _nb(la_c)
    conv = [0.0]*(len(a)+len(b)-1)
    for i, pa in enumerate(a):
        for j, pb in enumerate(b):
            conv[i+j] += pa*pb
    k = int(line)
    return float(sum(conv[k+1:]))

# ---------------------------------------------------------------- players
def _p_at_least_one(rate90, minutes, ctx=1.0):
    lam = rate90 * (minutes/90.0) * ctx
    return 1 - poisson.pmf(0, lam)

def player_markets(players, team_xg, opp_sot_mult=1.0):
    """players: list of dicts with per-90 rates + projected minutes.
    opp_sot_mult (§2.2 opponent adjustment): scales shot-on-target volume by how
    many shots the opponent concedes vs league average (>1 = leaky defence -> more
    SoT chances). Defaults to 1.0, so callers that don't supply it are unchanged."""
    ctx = team_xg / MU_GOALS                       # attacking environment
    out = []
    for p in players:
        m = p.get("min", 0)
        if m <= 0:
            continue
        out.append({"player": p["name"], "label": f"{p['name']} to score",
                    "p": round(_p_at_least_one(p["g90"], m, ctx), 3)})
        out.append({"player": p["name"], "label": f"{p['name']} to score or assist",
                    "p": round(_p_at_least_one(p["g90"]+p["a90"], m, ctx), 3)})
        out.append({"player": p["name"], "label": f"{p['name']} 1+ shot on target",
                    "p": round(_p_at_least_one(p["sot90"], m, opp_sot_mult), 3)})
        out.append({"player": p["name"], "label": f"{p['name']} to commit a foul",
                    "p": round(_p_at_least_one(p["fc90"], m), 3)})
        out.append({"player": p["name"], "label": f"{p['name']} to be fouled",
                    "p": round(_p_at_least_one(p["fd90"], m), 3)})
    return out

# ---------------------------------------------------------------- lineup -> team strength (§5.4)
# A rested or absent key player should weaken the TEAM's attack/defence rating,
# not just zero out his own player props — so a benched Yamal ripples through the
# result, over/under and corners markets too. We approximate each player's share
# of the team's attacking output from his goal-involvement per-90s. That input is
# WC form for now; §2.2 will feed the *same* function richer club-season weights.
#
# Only players the agent explicitly rules OUT or projects on reduced minutes count
# as evidence of weakness. Unlisted players are assumed to play normally, so an
# empty/mock projection (sample mode) returns (1.0, 1.0) and leaves ratings — and
# therefore the sample data.json — untouched.
ATK_LINEUP_SENS = 1.0       # how hard missing attackers dock the attack rating
DEF_LINEUP_SENS = 0.4       # losing creators also tends to concede a little more
ATK_FLOOR, DEF_CEIL = 0.6, 1.4   # clamp so one projection can't swing a rating wildly

def lineup_strength(pool, side):
    """Return (atk_mult, def_mult) to scale a team rating given the agent's lineup
    projection. `pool` is the full per-90 player pool; `side` is one half of the
    agent projection: {"starters":[{"name","min"}], "out":[name]}."""
    involvement = {p["name"]: p["g90"] + p["a90"] for p in pool}
    total = sum(involvement.values())
    if total <= 0:
        return 1.0, 1.0
    out = set(side.get("out", []))
    starters = {s["name"]: s.get("min", 90) for s in side.get("starters", [])}
    penalty = 0.0                                   # fraction of attacking output unavailable
    for name, w in involvement.items():
        share = w / total
        if name in out:
            penalty += share                        # ruled out -> full share lost
        elif name in starters:
            penalty += share * max(0.0, 1 - starters[name] / 90.0)   # rested -> partial
    atk = max(ATK_FLOOR, 1 - ATK_LINEUP_SENS * penalty)
    dfn = min(DEF_CEIL, 1 + DEF_LINEUP_SENS * penalty)
    return atk, dfn

# ---- more markets off the score matrix (all model-priced, bookmaker-matched) ----
def _marginals(M):
    rs = [sum(M[i][j] for j in range(GMAX+1)) for i in range(GMAX+1)]   # P(home scores i)
    cs = [sum(M[i][j] for i in range(GMAX+1)) for j in range(GMAX+1)]   # P(away scores j)
    return rs, cs

def _over_marg(marg, line):
    return float(sum(marg[k] for k in range(GMAX+1) if k > line))

def player_groups(hp, ap, lh, la, home, away):
    """Player props split into one group per prop type (instead of one big list),
    each sorted most-likely-first. SoT scales with the opponent's shots-conceded rate."""
    PROPS = [  # (group title, label suffix, rate keys, context kind)
        ("Players — to score",         "to score",           ("g90",),       "atk"),
        ("Players — shots on target",  "1+ shot on target",  ("sot90",),     "sot"),
        ("Players — to score/assist",  "to score or assist", ("g90","a90"),  "atk"),
        ("Players — to commit a foul", "to commit a foul",   ("fc90",),      "one"),
        ("Players — to be fouled",     "to be fouled",       ("fd90",),      "one"),
    ]
    sides = [(hp, lh, away.get("sot_con", 1.0)), (ap, la, home.get("sot_con", 1.0))]
    groups = []
    for title, suffix, keys, kind in PROPS:
        rows = []
        for players, team_xg, opp_sot in sides:
            ctx = {"atk": team_xg / MU_GOALS, "sot": opp_sot, "one": 1.0}[kind]
            for p in players:
                m = p.get("min", 0)
                if m <= 0:
                    continue
                rate = sum(p[k] for k in keys)
                rows.append({"player": p["name"], "label": f"{p['name']} {suffix}",
                             "p": round(_p_at_least_one(rate, m, ctx), 3)})
        if rows:
            groups.append({"name": title, "markets": sorted(rows, key=lambda x: -x["p"])})
    return groups

# ---------------------------------------------------------------- assemble
def build_fixture(home, away, hp, ap, agent_note=""):
    """home/away: team ratings. hp/ap: player lists w/ projected minutes."""
    H, A = home["name"], away["name"]
    lh, la = goals(home, away)
    ch = team_corners(home, away,  la-lh)
    ca = team_corners(away, home,  lh-la)
    M  = score_matrix(lh, la)                      # DC-corrected; all FT markets read off it
    M1 = score_matrix(lh*H1_SHARE, la*H1_SHARE)    # first-half scoreline grid
    hP, dP, aP = match_result(M)
    bts = btts_m(M)
    hf = htft(lh, la)
    h1h, h1d, h1a = half_result(lh, la)
    rs, cs = _marginals(M)
    csh = sum(M[i][0] for i in range(GMAX+1))      # home clean sheet (away scores 0)
    csa = sum(M[0][j] for j in range(GMAX+1))      # away clean sheet (home scores 0)
    wnh = sum(M[i][0] for i in range(1, GMAX+1))   # home win to nil
    wna = sum(M[0][j] for j in range(1, GMAX+1))   # away win to nil
    scores = sorted(((M[i][j], i, j) for i in range(GMAX+1) for j in range(GMAX+1)), reverse=True)[:7]

    def mk(label, p): return {"label": label, "p": round(p, 3)}
    def ou(prefix, line, p_over):                  # both sides of an over/under line
        return [mk(f"{prefix}Over {line}", p_over), mk(f"{prefix}Under {line}", 1 - p_over)]

    groups = [
        {"name": "Match result", "markets": [
            mk(f"{H} win", hP), mk("Draw", dP), mk(f"{A} win", aP)]},
        {"name": "Double chance", "markets": [
            mk(f"{H} or draw", hP+dP), mk("Either team (no draw)", hP+aP), mk(f"{A} or draw", dP+aP)]},
        {"name": "Total goals", "markets":
            [m for l in (0.5,1.5,2.5,3.5,4.5) for m in ou("", l, over_under_m(M, l))]},
        {"name": f"{H} total goals", "markets":
            [m for l in (0.5,1.5,2.5) for m in ou("", l, _over_marg(rs, l))]},
        {"name": f"{A} total goals", "markets":
            [m for l in (0.5,1.5,2.5) for m in ou("", l, _over_marg(cs, l))]},
        {"name": "Both teams to score", "markets": [mk("Yes", bts), mk("No", 1-bts)]},
        {"name": "Clean sheet", "markets": [mk(f"{H} yes", csh), mk(f"{A} yes", csa)]},
        {"name": "Win to nil", "markets": [mk(f"{H} yes", wnh), mk(f"{A} yes", wna)]},
        {"name": "Correct score", "markets": [mk(f"{i}:{j}", p) for p, i, j in scores]},
        {"name": "First-half result", "markets": [
            mk(f"{H} lead", h1h), mk("Level", h1d), mk(f"{A} lead", h1a)]},
        {"name": "First-half goals", "markets":
            [m for l in (0.5,1.5,2.5) for m in ou("", l, over_under_m(M1, l))]},
        {"name": "Half-time / Full-time", "markets": [
            mk(k.replace("H", H[:3]).replace("A", A[:3]).replace("D","Draw"), v)
            for k, v in sorted(hf.items(), key=lambda x:-x[1])[:6]]},
        {"name": "Corners", "markets": (
            [m for l in (7.5,8.5,9.5,10.5) for m in ou("Total ", l, total_corners_over(ch, ca, l))] +
            [mk(f"{A} {k}+", team_corner_tail(ca, k)) for k in (3,4,5)] +
            [mk(f"{H} {k}+", team_corner_tail(ch, k)) for k in (4,5,6)])},
    ] + player_groups(hp, ap, lh, la, home, away)
    return {"home": H, "away": A,
            "xg": [round(lh,2), round(la,2)], "corners": [round(ch,1), round(ca,1)],
            "agent_note": agent_note, "groups": groups}
