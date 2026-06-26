#!/usr/bin/env python3
"""
model.py — every probability the site shows. Pure math, no AI, no network.

Team markets are derived from a goals model (Poisson) and a corners model
(Negative Binomial). Player props are per-90 rates scaled by projected minutes
(the minutes come from agent.py, which reads team news).
"""
from scipy.stats import poisson, nbinom

MU_GOALS, LEAGUE_AVG_CORNERS, CORNER_R = 1.35, 5.1, 7.0
H1_SHARE = 0.45                      # ~45% of goals fall in the first half
GMAX = 8                             # goal grid cap

# ---------------------------------------------------------------- goals
def goals(home, away):
    lh = MU_GOALS * home["atk"] * away["def"]
    la = MU_GOALS * away["atk"] * home["def"]
    return lh, la

def _pmf(lmbda, n=GMAX):
    return [poisson.pmf(i, lmbda) for i in range(n + 1)]

def score_matrix(lh, la):
    ph, pa = _pmf(lh), _pmf(la)
    return [[ph[i] * pa[j] for j in range(GMAX + 1)] for i in range(GMAX + 1)]

# ---- markets off the full-time score matrix ----
def match_result(M):
    h = sum(M[i][j] for i in range(GMAX+1) for j in range(GMAX+1) if i > j)
    d = sum(M[i][i] for i in range(GMAX+1))
    a = 1 - h - d
    return h, d, a

def over_under(lh, la, line):
    return 1 - poisson.cdf(int(line), lh + la)        # over `line` goals

def btts(lh, la):
    return (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))

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

def player_markets(players, team_xg, opp=False):
    """players: list of dicts with per-90 rates + projected minutes."""
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
                    "p": round(_p_at_least_one(p["sot90"], m, ctx), 3)})
        out.append({"player": p["name"], "label": f"{p['name']} to commit a foul",
                    "p": round(_p_at_least_one(p["fc90"], m), 3)})
        out.append({"player": p["name"], "label": f"{p['name']} to be fouled",
                    "p": round(_p_at_least_one(p["fd90"], m), 3)})
    return out

# ---------------------------------------------------------------- assemble
def build_fixture(home, away, hp, ap, agent_note=""):
    """home/away: team ratings. hp/ap: player lists w/ projected minutes."""
    lh, la = goals(home, away)
    ch = team_corners(home, away,  la-lh)
    ca = team_corners(away, home,  lh-la)
    M  = score_matrix(lh, la)
    hP, dP, aP = match_result(M)
    hf = htft(lh, la)
    h1h, h1d, h1a = half_result(lh, la)

    def mk(label, p): return {"label": label, "p": round(p, 3)}

    groups = [
        {"name": "Match result", "markets": [
            mk(f"{home['name']} win", hP), mk("Draw", dP), mk(f"{away['name']} win", aP)]},
        {"name": "Total goals", "markets": [
            mk(f"Over {l}", over_under(lh, la, l)) for l in (0.5,1.5,2.5,3.5,4.5)]},
        {"name": "Both teams to score", "markets": [
            mk("Yes", btts(lh, la)), mk("No", 1-btts(lh, la))]},
        {"name": "First-half result", "markets": [
            mk(f"{home['name']} lead", h1h), mk("Level", h1d), mk(f"{away['name']} lead", h1a)]},
        {"name": "Half-time / Full-time", "markets": [
            mk(k.replace("H", home['name'][:3]).replace("A", away['name'][:3]).replace("D","Draw"), v)
            for k, v in sorted(hf.items(), key=lambda x:-x[1])[:6]]},   # top 6 most likely
        {"name": "Corners", "markets": (
            [mk(f"Total over {l}", total_corners_over(ch, ca, l)) for l in (7.5,8.5,9.5,10.5)] +
            [mk(f"{away['name']} {k}+", team_corner_tail(ca, k)) for k in (3,4,5)] +
            [mk(f"{home['name']} {k}+", team_corner_tail(ch, k)) for k in (4,5,6)])},
        {"name": "Players", "markets":
            player_markets(hp, lh) + player_markets(ap, la)},
    ]
    return {"home": home["name"], "away": away["name"],
            "xg": [round(lh,2), round(la,2)], "corners": [round(ch,1), round(ca,1)],
            "agent_note": agent_note, "groups": groups}
