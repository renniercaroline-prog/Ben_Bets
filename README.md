# World Cup Betting Model

A static, self-updating web app that prices betting markets for the 2026 FIFA World Cup,
compares its probabilities against live bookmaker odds, and **measures whether it actually
has an edge** — rather than assuming it does.

It runs with no server and no database: a scheduled GitHub Action runs a Python pipeline
that fetches data, runs a statistical model, pulls odds, and commits a `data.json` that a
single vanilla-JS page renders. Hosted free on GitHub Pages.

> **Design philosophy:** a flagged "value" bet is noise until it's *measured*. The project
> ships with a backtest harness and a closing-line-value logger, and is deliberately honest
> about what's proven (the priors help, calibration is good) and what isn't (it is **not**
> yet shown to beat the market). The tooling to find out is the point.

---

## The core problem it solves

Estimating a team or player from their ≤3 World Cup games is hopelessly noisy. The whole
approach is therefore:

> **strong historical prior  +  small World-Cup update  +  measure everything.**

The sparse tournament data becomes a minor adjustment on top of a low-variance prior built
from years of history, and every modelling decision is gated on a backtest instead of a hunch.

---

## Architecture

```
 API-Football ─┐
 FBref (xG) ───┼─►  update.py  ─►  model.py   (probability math)
 OpenAI agent ─┘   (orchestrator) ├─ elo.py    (historical prior + shrinkage)
                                  ├─ agent.py  (LLM → lineups only)
                                  ├─ xg.py     (optional xG ratings)
                                  └─ clv.py    (closing-line-value log)
                                       │
                                       ▼
                                   data.json  ─►  index.html  (UI: EV / Kelly / board)

 backtest.py  ─  evaluation harness (Brier / log-loss / calibration, walk-forward)
```

| File | Role |
|---|---|
| `model.py` | All probability math — pure functions, no network. Dixon-Coles-ready bivariate-Poisson goals (one corrected score matrix drives 20 mutually-consistent markets), negative-binomial corners, per-90 player props. |
| `elo.py` | The historical prior: a time-decayed, margin-adjusted Elo over each team's past internationals, shrunk with sparse WC form. |
| `agent.py` | The AI layer — an LLM that projects **lineups only** (see below). It never outputs a probability. |
| `xg.py` | Optional expected-goals team ratings from FBref; graceful fallback to goals. |
| `update.py` | Orchestrator: fetch → build prior → (gated) agent → model → odds → CLV → write `data.json`. |
| `backtest.py` | The scoreboard: walk-forward Brier/log-loss + calibration, on real or synthetic data. |
| `clv.py` | Closing-line-value logger — the forward test of edge. |
| `index.html` | Vanilla-JS UI: odds entry, EV/Kelly, ranked best-bets board, track-record panel. |
| `.github/workflows/daily.yml` | Scheduled CI that runs the pipeline and commits the output. |

---

## How the model works

- **Team strength — historical Elo prior + shrinkage.** A margin-adjusted Elo (goal
  difference scales the update) with a ~15-month time-decay half-life, fit over each team's
  international results. The sparse WC form is then a small correction:
  `rating = w·prior + (1−w)·form`, with `w` high when few WC games exist and decaying as
  they accumulate. Elo is mapped to the model's attack/defence rates so it feeds goals,
  result, over/under, BTTS and (via supremacy) corners.
- **Goals — bivariate Poisson with Dixon-Coles correction.** Every full-time market (result,
  totals, BTTS, clean sheet, win-to-nil, correct score, double chance, halves, HT/FT) is read
  off **one** corrected score matrix, so prices are internally consistent.
- **Corners — negative binomial** (overdispersed), convolved for totals, with a game-state
  multiplier for the favourite.
- **Player props — per-90 × minutes × context.** Each rate is anchored to the player's
  **club-season** per-90s and shrunk toward sparse WC form (a 35-game club season beats 3 WC
  games); shots-on-target props scale with the opponent's shots-conceded rate.
- **xG (optional).** Where a free source has it, season xG drives attack/defence ratings
  (xG stabilises faster than goals); otherwise it falls back to goals.

**Markets priced (20 groups/game):** match result, double chance, total goals (over *and*
under 0.5–4.5), each team's total goals, BTTS, clean sheet, win-to-nil, correct score,
first- and second-half result, first- and second-half goals, half-time/full-time, corners
(totals + per-team), and player props split by type (to score / shots on target /
score-or-assist / fouls committed / fouled).

---

## The AI agent — a deliberately bounded role

The project uses an LLM (OpenAI Responses API, GPT-5.5, with the hosted web-search tool),
but under a strict architectural rule:

> **The LLM never computes a probability.** Every number a user could bet on comes from the
> statistical model. The agent's *only* job is to read the latest team news and return, as
> strict JSON, each side's **projected starting XI, minutes, and who's ruled out.**

This boundary is intentional: LLMs hallucinate and are poorly calibrated, so letting one set
a betting probability would be the wrong design. Keeping it to a perception task it's
genuinely good at — *reading the news and projecting a lineup* — is where it adds value.

**How the lineup feeds the model:**
1. Projected minutes scale each player's per-90 rate into a player-prop probability (a player
   marked "out" drops off entirely).
2. A rested or missing key player also **docks the team's attack/defence rating** — so a
   benched star ripples through the result, totals and corners markets, not just his own props.

**Cost-aware engineering (the agent is pay-per-call):**
- It only calls the LLM when a fixture is **within ~3 hours of kickoff** — when lineups
  actually matter and start to confirm. Fixtures further out get a no-call placeholder.
- Projections are **cached** and reused (refresh ≤ 60 min) rather than re-queried every run.
- Net effect: **~90% fewer API calls** than a naive per-fixture-per-run approach, *and*
  sharper projections (made closer to the confirmed XI).
- The pipeline runs perfectly without the agent (no key → fallback minutes), so the AI layer
  is an enhancement, never a dependency.

---

## Evaluation — earning the edge before trusting it

`backtest.py` is the scoreboard every change is judged on: **walk-forward** (train on data
strictly before each match — no look-ahead leakage), reporting **Brier score, log-loss and
reliability/calibration** per market, with a naive base-rate baseline and a skill%. It runs
against real history (API-Football) or a synthetic known-truth generator (no key), and the
rating method is pluggable (`RATING_MODEL=baseline|elo|elo2d|xgelo`).

**Verified on real Premier League data** (708 walk-forward matches):

| market | baseline Brier (skill%) | Elo prior (skill%) |
|---|:---:|:---:|
| result (1X2) | 0.666 (−5.2%) | **0.614 (+3.1%)** |
| over 2.5 | 0.278 (−14.6%) | 0.247 (−1.7%) |
| both-teams | 0.281 (−14.2%) | 0.253 (−3.1%) |
| corners 8.5 | 0.231 (−9.6%) | 0.214 (−1.4%) |

The old rolling-form baseline is **negative-skill on every market** (over-confident, worse
than the base rate). The Elo prior + shrinkage turns the result market positive and roughly
halves the deficit elsewhere; calibration goes from wildly over-confident to near-diagonal.

**Validated on the actual domain — internationals.** Pointed at a multi-confederation basket
of **3,061 senior international matches** (WC, Euro, Copa, AFCON, Asian Cup, Gold Cup, Nations
Leagues, qualifiers, friendlies), the prior transfers and transfers *better*: **+6.9% skill
on the result market** (vs +3.1% on the PL).

### Honest engineering decisions (all evidence-gated)

- **1D over 2D ratings.** A two-dimensional attack/defence rating beat 1D on data-rich club
  football — but **lost** on internationals, where teams play too few games to estimate two
  parameters reliably. The product is international, so the live pipeline uses 1D. *A gain
  measured in the wrong domain doesn't always hold in the right one.*
- **Dixon-Coles defaulted OFF.** The correction is implemented, but fitting `rho` by
  exact-scoreline likelihood on real data didn't support a nonzero value — so it ships at 0
  (plain Poisson) rather than shipping an un-evidenced edge. The machinery stays for any
  competition where the data does justify it.
- **Home advantage deliberately omitted.** WC venues are neutral, so a home term would bias
  the predictions the product makes. (Side-effect: the club-league backtest *understates* the
  model — add a home multiplier first if ever retargeting at domestic leagues.)

---

## The betting layer

The page does the value math from the user's (or the auto-fed) odds:

- **Auto odds feed.** Pulls bookmaker prices, **line-shops the best price** across the user's
  UK books, and **de-vigs** each market to a fair probability.
- **Expected value & staking.** `EV = model_p × odds − 1`; a positive EV is flagged "VALUE"
  with a **fractional-Kelly** stake (`edge ÷ (odds−1)`, scaled by a user multiplier).
- **Ranked best-bets board** with guardrails: it drops the model-vs-market blow-ups that are
  usually model error (the headline trap — a model's biggest "edges" are its biggest
  mistakes), keeping only small, believable edges, clearly labelled *unproven*.
- **Per-game best-value** and inline `model% · book price · implied%` on every market.

## Closing-line value — the forward test

`clv.py` is the real judge of edge. For every flagged bet it logs the **opening price**, rolls
the **closing line** as kickoff approaches, computes **CLV** (did we beat the close?), and
**settles** the result for realised P/L. The website's "Model track record" panel surfaces it.

CLV is the leading indicator: if the market consistently moves toward your side by kickoff,
you're genuinely ahead of it — and it's meaningful in far fewer bets than realised profit.
The panel makes the honest point explicitly: **a flattering ROI on a small sample with flat
CLV is luck, not edge.**

---

## Honest status

The priors measurably improve calibration over the naive baseline, and the system is
internally sound — but it is **not yet proven to beat the market**. CLV is still accumulating,
and the largest known issue is a systematic over-bias on totals. Nothing here should be bet
with real money until CLV is clearly positive over a meaningful sample. The reliable money
during the tournament is promotional/matched-betting value; the model is the moonshot, and
the CLV log is what tells you — risk-free — whether the moonshot is real.

---

## Tech stack

- **Modelling:** Python, `scipy`/`numpy` (Poisson/NB distributions, score matrices, Elo,
  shrinkage, calibration metrics). Optional `soccerdata` (FBref xG) and `matplotlib`
  (calibration plots).
- **AI layer:** OpenAI Responses API (GPT-5.5 + hosted web search), strictly bounded to
  lineup projection.
- **Data:** API-Football (PRO) for fixtures, results, stats, odds; FBref for xG.
- **Frontend:** a single hand-written `index.html` — vanilla JS, no framework, no build step.
- **Infra:** GitHub Actions (cron + manual dispatch) as the entire backend; GitHub Pages for
  hosting. State lives in committed JSON — no server, no database. Secrets in encrypted
  Actions secrets; nothing hardcoded.

---

## Running it

**Locally (no keys needed — writes labelled sample data):**
```bash
pip install scipy
python update.py        # writes a sample data.json
python -m http.server   # open http://localhost:8000
```

**Backtest:**
```bash
python backtest.py                                  # synthetic, baseline ratings
RATING_MODEL=elo python backtest.py                 # swap the rating method
python backtest.py --recalibrate                    # data-fit the constants
BACKTEST_MODE=live BACKTEST_LEAGUE=39 BACKTEST_SEASONS=2022,2023 python backtest.py
```

**Deploy:** add `API_FOOTBALL_KEY` (and optionally `OPENAI_API_KEY`) to GitHub → Settings →
Secrets and variables → Actions, enable Pages on `main`, and trigger the workflow.

---

## What this project demonstrates

- **Statistical modelling under data scarcity** — Bayesian-flavoured priors + shrinkage,
  time-decay, calibration, and walk-forward evaluation.
- **Disciplined LLM integration** — an AI agent bounded to the one task it's reliable at, with
  cost-aware gating and caching, and zero dependence on it.
- **Evidence over intuition** — every modelling choice gated on a backtest, with decisions
  (1D vs 2D, Dixon-Coles off, home advantage omitted) driven by data and documented honestly.
- **End-to-end delivery** — data pipeline → model → odds/EV engine → web UI → CI/CD, plus a
  forward-validation loop (CLV) that keeps the whole thing honest.
