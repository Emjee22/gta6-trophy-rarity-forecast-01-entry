#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster, version 2.

make.fun competition gta6-trophy-rarity-forecast-01: forecast every PS5
trophy's Exophase earn rate at a target instant, as five quantiles, from
the house's recorded reads of the game's trophy page.

Method: count-consistent decomposition. An earn rate is a ratio,

    rate = earners / players_tracked,

so the model forecasts the two counts separately and divides, instead of
extrapolating the ratio blindly. For each trophy it fits two views of the
same history and blends them in logit space:

  (1) rate view: logit(rate) is linear in log-time since launch
      (x = ln(1 + hours since the 19 Nov 00:00Z launch)). The slope is
      fitted per trophy, then shrunk toward its tier's mean slope, so a
      noisy or barely-moved trophy borrows strength from its peers.
      The line is anchored at the last observed rate, which keeps the
      forecast glued to the freshest read instead of to an old intercept.

  (2) counts view: log(earners) is linear in the same log-time, fitted
      per trophy (shrunk the same way) and anchored at the last
      reconstructed earner count, then divided by a players curve fitted
      the same way (log players linear in log-time). This is the view
      that understands dilution: when new players join faster than
      earners grow, the rate falls even though more people earn the
      trophy.

The median is the average of the two views in logit space. The spread
is honest: the regression residual (floored by a pooled value so one
lucky trophy cannot claim false precision), the regression prediction
error which widens as the target moves away from the data, half the
disagreement between the two views, and a horizon term that grows like
a random walk. The 5% / 95% band is meant to cover about 90% of the
time, so the fitted spread is calibrated (scaled) toward that coverage
rather than left wide.

Fits are recency-weighted: over a two-day history the growth pattern
drifts (the launch surge fades), so a recent read counts more than one
from launch night. The weight halves every HALF_LIFE_H hours before
the checkpoint. The residual scale stays an unweighted pooled measure,
so the trend adapts while the noise estimate stays stable.

When the history is flat -- every read shows the same players and the
same rates, which is what Exophase looks like before its first refresh
-- there is no trend to fit. The model then falls back to a
launch-dynamics dilution prior instead of pretending the stale page
will persist: on launch day new players flood in far faster than
trophies are earned, so players are assumed to grow like
(1 + hours since launch) and earners much more slowly, so rates fall,
with a wide spread because the size of the coming refresh is genuinely
unknown.

A trophy stuck at 100% is forecast at 100% with only a small lower
tail. A trophy never seen in any read gets its tier's typical rate
with a wide spread. Everything is deterministic: no randomness, no
wall-clock use, no network, standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Normal quantiles for the levels 0.05, 0.25, 0.5, 0.75, 0.95.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Launch-dynamics prior used only when the history is flat.
D_PRIOR = 1.0    # players grow like (1 + hours since launch) ** D_PRIOR
B_PRIOR = 0.2    # earners grow like (1 + hours since launch) ** B_PRIOR
SIGMA_FLAT = 1.2  # logit spread when no refresh has been seen yet

K_SHRINK = 6.0   # prior strength when shrinking a slope to its tier mean
SIGMA_H_COEF = 0.25  # horizon term: SIGMA_H_COEF * sqrt(days ahead)
SIGMA_FLOOR = 0.20
SIGMA_CAP = 2.0
SIGMA_SCALE = 0.75  # spread calibration toward ~90% coverage
D_CAP = 2.0      # fastest players growth (log slope) allowed
B_CAP = 3.0      # fastest earners growth (log slope) allowed
BETA_CAP = 3.0   # fastest logit-rate slope allowed
P_MULT_CAP = 50.0  # players forecast never exceeds this x the last count
HALF_LIFE_H = 24.0  # recency weight halves every this many hours

# Typical rate (percent) for a trophy never observed, by tier.
TIER_PRIOR = {"platinum": 0.10, "gold": 0.50, "silver": 1.00,
              "bronze": 1.00}


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    """logit of a percent, clamped so 0 and 100 stay finite."""
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(l):
    if l >= 0.0:
        return 1.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return e / (1.0 + e)


def parse_time(s):
    """Parse an ISO 8601 timestamp (with Z, offset, or fractions)."""
    if not isinstance(s, str):
        raise ValueError("not a timestamp: %r" % (s,))
    txt = s.strip()
    if txt.endswith("Z") or txt.endswith("z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_launch(game):
    """Launch instant from the case's game.release_date (YYYY-MM-DD)."""
    rd = (game or {}).get("release_date")
    if isinstance(rd, str) and len(rd) >= 10:
        try:
            return datetime(int(rd[0:4]), int(rd[5:7]), int(rd[8:10]),
                            tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def log_time(t, t0):
    """x = ln(1 + hours since launch), clamped at 0 before launch."""
    h = (t - t0).total_seconds() / 3600.0
    if h < 0.0:
        h = 0.0
    return math.log1p(h)


def recency_weight(read_time, checkpoint):
    """Weight of a read: halves every HALF_LIFE_H hours before checkpoint."""
    age_h = (checkpoint - read_time).total_seconds() / 3600.0
    if age_h <= 0.0:
        return 1.0
    return 0.5 ** (age_h / HALF_LIFE_H)


def wols(xs, ys, ws):
    """Recency-weighted least squares y = a + b*x.

    Returns (a, b, s, xbar, sxx) where s is the weighted residual RMS
    (0.0 for a two-point fit), or None when the slope is not identified
    (fewer than 2 points, or no spread in x).
    """
    n = len(xs)
    if n < 2:
        return None
    wsum = sum(ws)
    if wsum <= 0.0:
        return None
    xbar = sum(w * x for w, x in zip(ws, xs)) / wsum
    ybar = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = 0.0
    sxy = 0.0
    for w, x, y in zip(ws, xs, ys):
        dx = x - xbar
        sxx += w * dx * dx
        sxy += w * dx * (y - ybar)
    if sxx <= 1e-12:
        return None
    b = sxy / sxx
    a = ybar - b * xbar
    if n > 2:
        ss = 0.0
        for w, x, y in zip(ws, xs, ys):
            r = y - (a + b * x)
            ss += w * r * r
        s = math.sqrt(ss / wsum)
    else:
        s = 0.0
    return (a, b, s, xbar, sxx)


def _same_state(a, b):
    """True when two reads show the same page values.

    A rate that is null in either read is ignored: a trophy missing
    from one read is a gap in what that read captured, not a change
    in the page's values.
    """
    if a.get("players_tracked") != b.get("players_tracked"):
        return False
    ra = a.get("rates") or []
    rb = b.get("rates") or []
    if len(ra) != len(rb):
        return False
    for x, y in zip(ra, rb):
        if x is None or y is None:
            continue
        if x != y:
            return False
    return True


def dedupe_reads(history):
    """Keep only reads that change the page.

    Between Exophase refreshes the page is byte-identical, so the
    house's frequent reads repeat one state many times. Fitting on the
    repeats would let long flat stretches outweigh the actual changes;
    the distinct states are the information. Each state is kept at the
    last read that showed it, the moment closest to when the page
    moved on to the next state, so a change is not smeared over the
    whole stretch the old state sat there.
    """
    out = []
    for r in history:
        if not isinstance(r, dict):
            continue
        if out and _same_state(out[-1], r):
            out[-1] = r
        else:
            out.append(r)
    return out


def fit_players(reads, t0, x_target, p_last, x_last, checkpoint):
    """Forecast players_tracked at the target.

    log players is linear in log-time, fitted on the distinct reads
    (recency-weighted) and anchored at the last observed count, so the
    forecast grows from where the page actually is. Floored at the last
    count (tracked players do not shrink) and capped at a generous
    multiple of it.
    """
    pts = []
    for r in reads:
        p = r.get("players_tracked")
        if isinstance(p, (int, float)) and p > 0:
            try:
                t = parse_time(r["at"])
            except Exception:
                continue
            pts.append((log_time(t, t0), math.log(float(p)),
                        recency_weight(t, checkpoint)))
    d = 0.0
    if len(pts) >= 2:
        xs = [q[0] for q in pts]
        if max(xs) - min(xs) > 1e-9:
            res = wols(xs, [q[1] for q in pts], [q[2] for q in pts])
            if res is not None:
                d = clamp(res[1], 0.0, D_CAP)
    p_t = p_last * math.exp(d * (x_target - x_last))
    return clamp(p_t, p_last, p_last * P_MULT_CAP)


def forecast_case(case):
    """Return a list of (trophy id, median logit, sigma) for one case."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_min = case.get("horizon_minutes")
    horizon_days = (float(horizon_min) / 1440.0
                    if isinstance(horizon_min, (int, float)) else 0.0)
    t0 = parse_launch(case.get("game")) or at
    x_target = log_time(target, t0)

    reads = dedupe_reads(history)
    flat = len(reads) < 2

    # Anchor: the last read overall (same state as the last distinct
    # read, but confirmed most recently).
    last = history[-1] if history else None
    p_last = 1.0
    x_last = 0.0
    if isinstance(last, dict):
        p = last.get("players_tracked")
        if isinstance(p, (int, float)) and p > 0:
            p_last = float(p)
        try:
            x_last = log_time(parse_time(last["at"]), t0)
        except Exception:
            x_last = 0.0

    p_target = p_last if flat else fit_players(
        reads, t0, x_target, p_last, x_last, at)

    # Hours since launch, for the flat-history dilution prior. The
    # stale page was last fresh at or before the first read, so the
    # growth the coming refresh reveals runs from there to the target.
    h_target = max(0.0, (target - t0).total_seconds() / 3600.0)
    h_stale = 0.0
    if history:
        try:
            h_stale = max(0.0, (parse_time(history[0]["at"]) - t0
                                ).total_seconds() / 3600.0)
        except Exception:
            h_stale = 0.0
    flat_factor = ((1.0 + h_target) / (1.0 + h_stale)) ** (B_PRIOR - D_PRIOR)

    # Per-trophy series over the distinct reads: (x, rate, players, w).
    n = len(trophies)
    series = []
    for j in range(n):
        pts = []
        for r in reads:
            rates = r.get("rates") or []
            p = r.get("players_tracked")
            if (j < len(rates) and isinstance(rates[j], (int, float))
                    and isinstance(p, (int, float)) and p > 0):
                try:
                    t = parse_time(r["at"])
                except Exception:
                    continue
                pts.append((log_time(t, t0), float(rates[j]), float(p),
                            recency_weight(t, at)))
        series.append(pts)

    # First pass: raw slopes and residual scales, per trophy.
    raw_beta = []
    raw_b = []
    resid = []
    xbar_of = []
    sxx_of = []
    for pts in series:
        beta = None
        b_slope = None
        s = None
        xb = None
        sxx = None
        if len(pts) >= 2:
            xs = [q[0] for q in pts]
            if max(xs) - min(xs) > 1e-9:
                ws = [q[3] for q in pts]
                res1 = wols(xs, [logit(q[1]) for q in pts], ws)
                if res1 is not None:
                    beta = clamp(res1[1], -BETA_CAP, BETA_CAP)
                    s = res1[2]
                    xb = res1[3]
                    sxx = res1[4]
                us = [math.log(max(q[1] * q[2] / 100.0, 0.5))
                      for q in pts]
                res2 = wols(xs, us, ws)
                if res2 is not None:
                    b_slope = clamp(res2[1], 0.0, B_CAP)
        raw_beta.append(beta)
        raw_b.append(b_slope)
        resid.append(s)
        xbar_of.append(xb)
        sxx_of.append(sxx)

    tiers = [(tr.get("tier") or "bronze") if isinstance(tr, dict)
             else "bronze" for tr in trophies]

    def group_mean(vals, tier):
        same = [v for v, t in zip(vals, tiers)
                if v is not None and t == tier]
        if len(same) >= 2:
            return sum(same) / len(same)
        allv = [v for v in vals if v is not None]
        if allv:
            return sum(allv) / len(allv)
        return 0.0

    beta_group = {}
    b_group = {}
    for t in set(tiers):
        beta_group[t] = group_mean(raw_beta, t)
        b_group[t] = group_mean(raw_b, t)

    # Pooled residual scale (median of the per-trophy ones that exist).
    have_s = sorted(s for s in resid if s is not None and s > 0.0)
    s_pool = have_s[len(have_s) // 2] if have_s else 0.3

    sigma_h = (SIGMA_H_COEF * math.sqrt(horizon_days)
               if horizon_days > 0 else 0.0)

    rows = []
    for j, tr in enumerate(trophies):
        tid = tr.get("id") if isinstance(tr, dict) else None
        if tid is None:
            tid = j + 1
        tier = tiers[j]
        pts = series[j]
        if not pts:
            # Never observed: the tier's typical rate, wide spread.
            rows.append((tid, logit(TIER_PRIOR.get(tier, 1.0)), 1.5))
            continue
        last_rate = pts[-1][1]
        last_players = pts[-1][2]
        if flat:
            # No refresh seen yet: dilute by the launch prior, wide.
            if last_rate >= 99.99:
                med_logit = logit(99.99)
            else:
                med_logit = logit(last_rate * flat_factor)
            rows.append((tid, med_logit, SIGMA_FLAT))
            continue
        if last_rate >= 99.99:
            # Stuck at 100%: stay there, with a small lower tail that
            # grows with the horizon.
            sigma = clamp(0.2 + 0.5 * math.sqrt(horizon_days),
                          SIGMA_FLOOR, SIGMA_CAP)
            rows.append((tid, logit(99.99), sigma))
            continue
        npts = len(pts)
        w = npts / (npts + K_SHRINK)
        if raw_beta[j] is None:
            beta = beta_group[tier]
        else:
            beta = w * raw_beta[j] + (1.0 - w) * beta_group[tier]
        if raw_b[j] is None:
            b = b_group[tier]
        else:
            b = w * raw_b[j] + (1.0 - w) * b_group[tier]
        dx = x_target - x_last
        # View 1: logit-rate trend anchored at the last rate.
        logit1 = logit(last_rate) + beta * dx
        # View 2: earners trend anchored at the last earner count,
        # divided by the players forecast (dilution-aware).
        e_last = max(last_rate * last_players / 100.0, 0.5)
        rate2 = math.exp(math.log(e_last) + b * dx) / p_target * 100.0
        logit2 = logit(rate2)
        med_logit = 0.5 * (logit1 + logit2)
        # Spread: regression prediction error, view disagreement,
        # horizon term. The leverage term is capped so a clustered
        # history cannot blow the band up without bound. The total is
        # calibrated (SIGMA_SCALE) toward ~90% coverage.
        s_eff = resid[j] if (resid[j] is not None and npts >= 4) else 0.3
        s_fit = max(s_eff, 0.5 * s_pool)
        if xbar_of[j] is not None and sxx_of[j] and sxx_of[j] > 0:
            lev = (x_target - xbar_of[j]) ** 2 / sxx_of[j]
            pred_var = 1.0 + 1.0 / npts + min(lev, 4.0)
        else:
            pred_var = 1.0 + 1.0 / max(npts, 1)
        sigma_pred = s_fit * math.sqrt(max(pred_var, 1.0))
        sigma_dis = abs(logit1 - logit2) / 2.0
        sigma = math.sqrt(sigma_pred ** 2 + sigma_dis ** 2 + sigma_h ** 2)
        sigma = clamp(sigma * SIGMA_SCALE, SIGMA_FLOOR, SIGMA_CAP)
        rows.append((tid, med_logit, sigma))
    return rows


def fallback_output(case):
    """Last-resort valid output: persistence with a wide spread.

    Used only when the main forecast raises; the case's trophies are
    still answered so the run stays valid.
    """
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    last_rates = []
    if history:
        last = history[-1]
        if isinstance(last, dict):
            last_rates = last.get("rates") or []
    rows = []
    for j, tr in enumerate(trophies):
        tid = tr.get("id") if isinstance(tr, dict) else None
        if tid is None:
            tid = j + 1
        tier = (tr.get("tier") or "bronze") if isinstance(tr, dict) \
            else "bronze"
        rate = None
        if j < len(last_rates) and isinstance(last_rates[j], (int, float)):
            rate = float(last_rates[j])
        if rate is None:
            rate = TIER_PRIOR.get(tier, 1.0)
        med = logit(rate)
        qs = []
        for z in Z:
            v = clamp(100.0 * sigmoid(med + z * 1.5), 0.0, 100.0)
            qs.append(round(v, 4))
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tid, "q": qs})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def process_case(case):
    """Forecast one case into the exact output shape."""
    rows = forecast_case(case)
    out_rows = []
    for tid, med_logit, sigma in rows:
        qs = []
        for z in Z:
            v = clamp(100.0 * sigmoid(med_logit + z * sigma), 0.0, 100.0)
            qs.append(round(v, 4))
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out_rows.append({"id": tid, "q": qs})
    out_rows.sort(key=lambda r: r["id"])
    return {"trophies": out_rows}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(fn for fn in os.listdir(in_dir)
                       if fn.endswith(".in"))
    except OSError as e:
        sys.stderr.write("cannot list input dir %s: %s\n" % (in_dir, e))
        return 1
    for fn in names:
        path = os.path.join(in_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                case = json.load(f)
        except Exception as e:
            sys.stderr.write("skip %s: %s\n" % (fn, e))
            continue
        cid = case.get("id") if isinstance(case, dict) else None
        if not cid:
            cid = fn[:-3]
        try:
            obj = process_case(case)
        except Exception as e:
            sys.stderr.write("fallback %s: %s\n" % (fn, e))
            try:
                obj = fallback_output(case)
            except Exception:
                continue
        data = json.dumps(obj, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        out_path = os.path.join(out_dir, str(cid) + ".out")
        with open(out_path, "wb") as f:
            f.write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
