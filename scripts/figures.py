"""Regenerate every figure and number quoted in docs/writeup.md.

    make figures   # MLFLOW_TRACKING_URI=http://localhost:5001 uv run python scripts/figures.py

Uses the model at models:/smartbuilding-detector@production in the *stack's* registry — the one
the scorer runs — and the stack's database for the replay numbers, so the paper, the registry and
the dashboard describe the same model. Writes docs/img/fig*.png and docs/img/numbers.json; every
figure in the write-up is quoted from that file.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import psycopg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from smartbuilding.config import Env, Rules, load_config  # noqa: E402
from smartbuilding.data.demand import gap_report  # noqa: E402
from smartbuilding.data.splits import temporal_split  # noqa: E402
from smartbuilding.data.weather import load_weather_metric  # noqa: E402
from smartbuilding.eval import (  # noqa: E402
    ANOMALY_TYPES,
    BaselineDetector,
    event_metrics,
    run_eval,
)
from smartbuilding.features import FEATURES, build_features  # noqa: E402
from smartbuilding.model import Detector  # noqa: E402
from smartbuilding.pipeline import load_dataset, pinball_loss  # noqa: E402
from smartbuilding.registry import load_detector  # noqa: E402

BLUE, ORANGE, AQUA, RED, GRAY, INK, GRID = (
    "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9CA3AF", "#111827", "#E5E7EB",
)  # fmt: skip
plt.rcParams.update(
    {
        "font.family": "Helvetica Neue", "font.size": 10, "axes.edgecolor": GRAY,
        "axes.labelcolor": "#374151", "xtick.color": "#374151", "ytick.color": "#374151",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#EEF0F3", "grid.linewidth": 0.7, "axes.axisbelow": True,
        "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlecolor": INK, "legend.frameon": False,
    }
)  # fmt: skip
OUT = "docs/img"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

K_GRID, H_GRID, Z_GRID = (1.0, 1.5, 2.0), (12, 16), (4, 5)


def tuning_grid(det: Detector, cal: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """Clean false alarms per week and recall on small (+10 %, 6 h) offsets for each rule cell on
    the calibration year. The default injected anomalies are large enough that every cell catches
    them; the small offsets are what separates the cells."""
    keep = det.rules
    rows = []
    for k, h, z in itertools.product(K_GRID, H_GRID, Z_GRID):
        det.rules = rules.model_copy(update={"cusum_k": k, "cusum_h": h, "spike_z": z})
        clean_sc, _ = det.score(cal)
        fa = event_metrics(pd.DataFrame(columns=["start", "end", "type"]), clean_sc, 0)
        small = offset_sweep(det, cal, mags=(0.10,), n_events=20, seed=3)[0]
        rows.append(
            {"k": k, "h": h, "spike": z, "fa": fa["fa_per_week"], "recall_10pct": small["recall"]}
        )
    det.rules = keep
    return pd.DataFrame(rows)


def offset_sweep(det: Detector, frame: pd.DataFrame, mags, n_events: int, seed: int) -> list[dict]:
    """Inject `n_events` six-hour sustained offsets of each relative size at random start slots
    (at least four days apart) and report event recall and median time to detect. A size of 0.0
    injects nothing and measures how often a random window overlaps a background alert."""
    rng = np.random.default_rng(seed)
    f = frame.reset_index(drop=True)
    out = []
    for mag in mags:
        starts: list[int] = []
        while len(starts) < n_events:
            s0 = int(rng.integers(96, len(f) - 96 * 3))
            if all(abs(s0 - t) >= 96 * 4 for t in starts):
                starts.append(s0)
        injected = f.copy()
        labels = []
        for s0 in sorted(starts):
            injected.loc[s0 : s0 + 23, "kw"] *= 1 + mag
            labels.append({"start": s0, "end": s0 + 23, "type": "offset"})
        sc, _ = det.score(injected)
        m = event_metrics(pd.DataFrame(labels), sc, tolerance_slots=4)
        out.append({"mag": mag, "recall": round(m["recall"], 2), "ttd": m["ttd_median_min"]})
    return out


def matched_fa_baseline(
    train: pd.DataFrame, cal: pd.DataFrame, rules: Rules, tz: str, target_fa: float
) -> float:
    """The baseline's CUSUM allowance k that brings its clean false-alarm rate on the calibration
    year closest to the detector's, so the 2019 comparison is at (roughly) equal alarm budget."""
    best_k, best_gap = rules.cusum_k, float("inf")
    for k in (1.5, 1.25, 1.0, 0.75, 0.5):
        base = BaselineDetector.fit(train, rules.model_copy(update={"cusum_k": k}), tz)
        sc, _ = base.score(cal)
        fa = event_metrics(pd.DataFrame(columns=["start", "end", "type"]), sc, 0)["fa_per_week"]
        if abs(fa - target_fa) < best_gap:
            best_k, best_gap = k, abs(fa - target_fa)
    return best_k


def replay_from_db(url: str) -> tuple[pd.DataFrame, dict]:
    """Alert events per month by direction, and band coverage per year, from the stack's database
    (the replay of 2020-01 → 2021-05 through the live scorer)."""
    with psycopg.connect(url) as conn:
        alerts = pd.read_sql(
            "SELECT to_char(started_at AT TIME ZONE 'America/New_York', 'YYYY-MM') AS mon, "
            "CASE WHEN alert_type LIKE '%%LOW' THEN 'low' ELSE 'high' END AS side, "
            "count(*) AS n, sum(excess_kwh) AS kwh FROM alerts GROUP BY 1, 2 ORDER BY 1, 2",
            conn,
        )
        cov = pd.read_sql(
            "SELECT extract(year FROM ts AT TIME ZONE 'America/New_York')::int AS yr, "
            "avg((actual_kw BETWEEN lower_kw AND upper_kw)::int) AS coverage, count(*) AS n "
            "FROM scores WHERE actual_kw IS NOT NULL GROUP BY 1 ORDER BY 1",
            conn,
        )
    table = alerts.pivot_table(index="mon", columns="side", values="n", fill_value=0).reset_index()
    for c in ("low", "high"):
        table[c] = table.get(c, 0)
    kwh = alerts.groupby("mon").kwh.sum()
    table["kwh"] = table.mon.map(kwh).fillna(0).round(0)
    return table, {int(r.yr): (round(float(r.coverage), 3), int(r.n)) for r in cov.itertuples()}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("saved", name)


def alert_runs(sc: pd.DataFrame, mask: pd.Series) -> list[tuple[int, int]]:
    active = (sc.alert_type != "") & mask
    runs, start = [], None
    for i in sc.index[mask]:
        if active[i] and start is None:
            start = i
        if not active[i] and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, sc.index[mask][-1]))
    return runs


def main() -> None:  # noqa: PLR0915 — one linear script, one figure per block
    mlflow.set_tracking_uri(Env().mlflow_tracking_uri)
    cfg = load_config()
    tz = cfg.site.tz
    det = load_detector("models:/smartbuilding-detector@production")
    n: dict = {"model": det.model_version}

    data = load_dataset(cfg, "2016-01-01", "2021-05-31")
    parts = temporal_split(data, cfg.splits.as_bounds())
    tr, cal, te = parts["train"], parts["calibrate"], parts["test"]

    # ------------------------------------------------------------ §3 profiling + Fig 1
    g = gap_report(data[data.ts < "2020-01-01"])
    g["year"] = g.gap_start.dt.year
    n["gaps_by_year"] = (
        g.groupby("year").agg(n=("n_slots", "size"), slots=("n_slots", "sum")).to_dict("index")
    )
    d = tr.dropna(subset=["kw"])
    loc = d.ts.dt.tz_convert(tz)
    n["weekday_mean_kw"] = round(float(d.kw[loc.dt.dayofweek < 5].mean()), 1)
    n["weekend_mean_kw"] = round(float(d.kw[loc.dt.dayofweek >= 5].mean()), 1)
    byh = d.kw.groupby(loc.dt.hour).mean()
    n["hour_min"], n["hour_max"] = int(byh.idxmin()), int(byh.idxmax())
    n["trough_below_peak_pct"] = round(float((byh.max() - byh.min()) / byh.max() * 100), 1)
    raw = pd.read_csv(cfg.paths.demand_csv, usecols=["Timestamp"]).Timestamp
    n["demand_clock"] = {  # evidence that the export clock is UTC (see §3.1)
        "first": str(raw.iloc[0]),
        "last": str(raw.iloc[-1]),
        "rows_02xx_on_2019-03-10": int(raw.str.startswith("2019-03-10 02:").sum()),
        "rows_01xx_on_2019-11-03": int(raw.str.startswith("2019-11-03 01:").sum()),
    }
    cdd = (tr.temp_c - 22).clip(lower=0)
    n["r_kw_cdd_train"] = round(float(tr.kw.corr(cdd)), 3)
    wk = d.kw[loc.dt.dayofweek < 5].groupby(loc.dt.hour[loc.dt.dayofweek < 5]).mean()
    we = d.kw[loc.dt.dayofweek >= 5].groupby(loc.dt.hour[loc.dt.dayofweek >= 5]).mean()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(wk.index, wk.values, color=BLUE, lw=2, marker="o", ms=3)
    ax.plot(we.index, we.values, color=ORANGE, lw=2, marker="o", ms=3)
    ax.text(23.4, wk.values[-1], "Weekday", color=BLUE, va="center")
    ax.text(23.4, we.values[-1], "Weekend", color=ORANGE, va="center")
    ax.set_xlim(0, 25.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel("Mean demand (kW)")
    ax.set_title(
        f"Hourly mean demand, 2016–2017 — trough at {n['hour_min']:02d}:00, "
        f"peak at {n['hour_max']:02d}:00 local"
    )
    save(fig, "fig1_daily_profile")

    # ------------------------------------------------------------ Fig 2 weather validation
    om = pd.read_parquet(cfg.paths.weather_parquet).set_index("ts")
    st = load_weather_metric(cfg.paths.weather_metric_csv)
    st["ts"] = st.ts_local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    st = st.dropna(subset=["ts"]).assign(ts=lambda x: x.ts.dt.tz_convert("UTC").dt.round("h"))
    st = st.drop_duplicates("ts", keep="last").set_index("ts")
    j = om[["temp_c"]].join(st[["temp_c"]], how="inner", lsuffix="_om", rsuffix="_st").dropna()
    r = float(j.temp_c_st.corr(j.temp_c_om))
    n["weather_overlap_hours"], n["temp_r"] = int(len(j)), round(r, 3)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.hexbin(j.temp_c_st, j.temp_c_om, gridsize=60, cmap="Blues", mincnt=1, linewidths=0)
    ax.plot([-25, 35], [-25, 35], color=ORANGE, lw=1.5, ls="--")
    ax.set_xlabel("Station export (°C)")
    ax.set_ylabel("Open-Meteo reanalysis (°C)")
    ax.set_title(f"Temperature: reanalysis vs station\nr = {r:.3f}, n = {len(j):,} hours")
    ax.grid(False)
    save(fig, "fig2_weather_validation")

    # ------------------------------------------------------------ §4.5 table numbers (no figure)
    x_tr = build_features(tr, det.imputer, tz)
    corr = pd.concat([x_tr, tr.kw.rename("kw")], axis=1).corr()["kw"].drop("kw")
    imp = pd.Series(det.model.median.booster_.feature_importance("gain"), index=FEATURES)
    imp = imp / imp.sum() * 100
    n["features"] = {
        f: {"r": round(float(corr[f]), 3), "gain_pct": round(float(imp[f]), 1)} for f in FEATURES
    }

    # ------------------------------------------------------------ §5–6 fit + calibration (2018)
    calc = cal.dropna(subset=["kw"])
    pred_cal = det.predict(calc)
    hw = (pred_cal.q95 - pred_cal.q05) / 2
    n["raw_half_width_kw_p10_p50_p90"] = [round(float(hw.quantile(q)), 1) for q in (0.1, 0.5, 0.9)]
    n["cal_mean_kw"] = round(float(calc.kw.mean()), 1)
    n["cal_mae"] = round(float(np.abs(calc.kw.to_numpy() - pred_cal.q50.to_numpy()).mean()), 2)
    n["cal_pinball_q50"] = round(pinball_loss(calc.kw.to_numpy(), pred_cal.q50.to_numpy(), 0.5), 2)
    n["cal_raw_coverage"] = round(
        float(((calc.kw >= pred_cal.q05) & (calc.kw <= pred_cal.q95)).mean()), 3
    )
    n["calibration_factor"] = round(det.calibrator.factor, 3)
    n["sigma_kw_median"] = round(float(det.calibrator.sigma(pred_cal).median()), 1)

    # ------------------------------------------------------------ §7.1 diagnosis + Fig 4
    pred_te = det.predict(te)
    raw_z = det.calibrator.z(te.kw, pred_te)
    sc_te, _ = det.score(te)
    m = te.ts.dt.tz_convert(tz).dt.month
    n["z2019_raw_mean"] = round(float(raw_z.mean()), 2)
    n["z2019_raw_lag1"] = round(float(raw_z.dropna().autocorr(1)), 2)
    n["z2019_raw_monthly"] = raw_z.groupby(m).mean().round(2).tolist()
    n["z2019_adj_monthly"] = sc_te.z.groupby(m).mean().round(2).tolist()
    ok = te.kw.notna().to_numpy()
    n["test_mae"] = round(
        float(np.abs(te.kw.to_numpy()[ok] - pred_te.q50.to_numpy()[ok]).mean()), 2
    )
    n["test_pinball_q50"] = round(
        pinball_loss(te.kw.to_numpy()[ok], pred_te.q50.to_numpy()[ok], 0.5), 2
    )
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.axhline(0, color=GRAY, lw=1)
    ax.plot(
        range(1, 13),
        n["z2019_raw_monthly"],
        color=ORANGE,
        lw=2,
        marker="o",
        ms=5,
        label="raw residual vs the 2016–17 model",
    )
    ax.plot(
        range(1, 13),
        n["z2019_adj_monthly"],
        color=BLUE,
        lw=2,
        marker="o",
        ms=5,
        label="after level tracking (what CUSUM sees)",
    )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("Mean z (σ units)")
    drift = max(abs(v) for v in n["z2019_raw_monthly"])
    ax.set_title(
        f"2019 monthly mean residual — the level drifts by up to {drift:.1f}σ; "
        "the tracker removes it"
    )
    ax.legend(loc="lower right")
    save(fig, "fig3_level_drift")

    # ------------------------------------------------------------ Fig 4 tuning grid (2018)
    grid = tuning_grid(det, cal, cfg.rules)
    n["tuning_grid"] = grid.round(3).to_dict("records")
    chosen = grid[
        (grid.k == cfg.rules.cusum_k)
        & (grid.h == cfg.rules.cusum_h)
        & (grid.spike == cfg.rules.spike_z)
    ].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    panels = (
        (axes[0], "fa", "False alarms / week (clean 2018)"),
        (axes[1], "recall_10pct", "Recall, +10 % offsets for 6 h (2018)"),
    )
    for ax, col, ylab in panels:
        for (h, sp), gg in grid.groupby(["h", "spike"]):
            ax.plot(
                gg.k,
                gg[col],
                marker="o",
                lw=1.8,
                color=BLUE if sp == 5 else ORANGE,
                ls="-" if h == 12 else "--",
                label=f"spike z > {sp}, h = {h}",
            )
        ax.scatter(
            [chosen.k], [chosen[col]], s=180, facecolors="none", edgecolors=INK, lw=1.5, zorder=5
        )
        ax.set_xlabel("CUSUM allowance k")
        ax.set_ylabel(ylab)
        ax.set_xticks(list(K_GRID))
    axes[0].axhline(3, color=RED, lw=1, ls=":")
    axes[0].text(2.02, 3.05, "budget: 3 / week", color=RED, fontsize=9, ha="right")
    axes[0].set_ylim(0, 3.5)
    axes[1].set_ylim(0, 1.08)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Threshold tuning on the calibration year — k trades false alarms for sensitivity; "
        "chosen setting circled",
        fontweight="bold",
        color=INK,
    )
    save(fig, "fig4_tuning")

    # ------------------------------------------------------------ §9.1 2019, two seed sets
    base = BaselineDetector.fit(calc, cfg.rules, tz)
    k_matched = matched_fa_baseline(
        tr.dropna(subset=["kw"]), cal, cfg.rules, tz, target_fa=float(chosen.fa)
    )
    n["baseline_matched_k"] = k_matched
    base_m = BaselineDetector.fit(calc, cfg.rules.model_copy(update={"cusum_k": k_matched}), tz)
    res = {}
    for label, seeds in (("seeds_0_4", range(5)), ("seeds_5_9", range(5, 10))):
        res[label] = {
            "detector": run_eval(det, te, n_per_type=3, seeds=seeds).round(3).to_dict("index"),
            "baseline": run_eval(base, te, n_per_type=3, seeds=seeds).round(3).to_dict("index"),
            "baseline_matched_fa": run_eval(base_m, te, n_per_type=3, seeds=seeds)
            .round(3)
            .to_dict("index"),
        }
    n["eval_2019"] = res
    r5 = res["seeds_5_9"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    x = np.arange(len(ANOMALY_TYPES))
    w = 0.34
    off = 0.2
    for ax, key, ylab, log in (
        (axes[0], "recall", "Recall", False),
        (axes[1], "ttd_median_min", "Time to detect", True),
    ):
        dv = [r5["detector"][t][key] for t in ANOMALY_TYPES]
        bv = [r5["baseline"][t][key] for t in ANOMALY_TYPES]
        ax.bar(
            x - off,
            dv,
            w,
            color=BLUE,
            label=f"Detector ({r5['detector']['clean']['fa_per_week']:.2f} FA/week)",
        )
        ax.bar(
            x + off,
            bv,
            w,
            color=ORANGE,
            label=f"Seasonal-naive ({r5['baseline']['clean']['fa_per_week']:.2f} FA/week)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", "\n") for t in ANOMALY_TYPES], fontsize=9)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        if log:
            ax.set_yscale("symlog", linthresh=15)
            ax.set_yticks([0, 15, 60, 240, 1440, 5760])
            ax.set_yticklabels(["0", "15 min", "1 h", "4 h", "1 d", "4 d"])
        else:
            ax.set_ylim(0, 1.12)
            for i, (a, b) in enumerate(zip(dv, bv, strict=True)):
                ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
                ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    fig.suptitle(
        "Held-out 2019 (injection seeds disjoint from tuning) — recall and time to detect",
        fontweight="bold",
        color=INK,
    )
    save(fig, "fig5_results_2019")

    # ------------------------------------------------------------ Fig 6 sensitivity floor
    sweep = offset_sweep(
        det, te, mags=(0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40), n_events=20, seed=11
    )
    n["offset_sweep_6h"] = sweep
    sw = pd.DataFrame(sweep)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(sw.mag * 100, sw.recall, color=BLUE, lw=2, marker="o")
    ax.axhline(sw.recall.iloc[0], color=GRAY, lw=1, ls=":")
    ax.text(
        41, sw.recall.iloc[0] + 0.03, "chance (0 % offset)", color="#374151", fontsize=8, ha="right"
    )
    for _, row in sw.iterrows():
        if row.recall > 0 and row.mag > 0:
            ax.text(
                row.mag * 100,
                row.recall + 0.04,
                f"{row.ttd:.0f} min",
                ha="center",
                fontsize=8,
                color="#374151",
            )
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Injected sustained offset (% of demand, 6 h long, 20 events per point, 2019)")
    ax.set_ylabel("Recall")
    ax.set_title(
        "Sensitivity floor — recall vs offset size (labels: median minutes to first alert)"
    )
    save(fig, "fig6_offset_sweep")

    # ------------------------------------------------------------ Fig 10 coverage drift 2018 → 2021
    full = data[data.ts >= "2018-01-01"].reset_index(drop=True)
    scf, _ = det.score(full)
    okm = scf.actual_kw.notna()
    inside = ((scf.actual_kw >= scf.lower_kw) & (scf.actual_kw <= scf.upper_kw))[okm]
    ts_ok = scf.ts[okm].dt.tz_convert(tz)
    covm = inside.groupby(ts_ok.dt.strftime("%Y-%m")).mean()
    covm = covm[covm.index >= "2018-01"]  # drop the partial local-time month at the UTC boundary
    cov_y = inside.groupby(ts_ok.dt.year).mean().round(3)
    n["coverage_by_year"] = cov_y[cov_y.index >= 2018].to_dict()  # drop the UTC-boundary sliver
    n["months_below_0_80"] = int((covm < 0.80).sum())
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(range(len(covm)), covm.values, color=BLUE, lw=2, marker="o", ms=3)
    ax.axhline(0.90, color=GRAY, lw=1, ls="--")
    ax.text(
        len(covm) - 0.5, 0.905, "calibration target 0.90", color="#374151", fontsize=9, ha="right"
    )
    ax.axhline(0.80, color=RED, lw=1, ls=":")
    ax.text(0.2, 0.805, "health alert below 0.80", color=RED, fontsize=9)
    ax.axvspan(-0.5, 11.5, color=AQUA, alpha=0.08, lw=0)
    ax.text(5.5, 0.45, "2018: calibration year", ha="center", color="#374151", fontsize=9)
    ax.set_xticks(range(0, len(covm), 3))
    ax.set_xticklabels(covm.index[::3], rotation=45, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Share of readings inside the band")
    ax.set_title(
        "Monthly band coverage with the model frozen at 2018 — why retraining is a component"
    )
    save(fig, "fig8_coverage_drift")

    # ------------------------------------------------------------ Fig 7 replay alerts
    rep, n["replay_coverage_by_year"] = replay_from_db(Env().db_url)
    n["replay_alerts_by_month"] = rep.to_dict("records")
    n["replay_alerts_total"] = int(rep.low.sum() + rep.high.sum())
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(rep.mon, rep.low, color=BLUE, label="LOW (below expectation)")
    ax.bar(rep.mon, rep.high, bottom=rep.low, color=ORANGE, label="HIGH (above expectation)")
    lock = [i for i, m in enumerate(rep.mon) if m in ("2020-03", "2020-04")]
    ymax = float((rep.low + rep.high).max()) * 1.25
    ax.axvspan(lock[0] - 0.5, lock[-1] + 0.5, color=GRAY, alpha=0.12, lw=0)
    ax.text(np.mean(lock), ymax * 0.9, "lockdown", ha="center", color="#374151", fontsize=10)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Alert events")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_title(
        "Replay Jan 2020 to May 2021 through the live system (real readings, model frozen at 2018)"
    )
    save(fig, "fig7_replay_alerts")

    # ------------------------------------------------------------ Fig 8 an alert up close
    rp = load_dataset(cfg, "2019-12-31", "2020-01-04").reset_index(drop=True)
    inj = rp.copy()
    s0, s1 = 96 + 96 + 33, 96 + 96 + 33 + 43
    inj.loc[s0:s1, "kw"] *= 1.25
    sc, _ = det.score(inj)
    t = sc.ts.dt.tz_convert(tz)
    win = (sc.ts >= pd.Timestamp("2020-01-01 17:00", tz="UTC")) & (
        sc.ts <= pd.Timestamp("2020-01-03 05:00", tz="UTC")
    )
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax.fill_between(
        t[win],
        sc.lower_kw[win],
        sc.upper_kw[win],
        color=BLUE,
        alpha=0.15,
        lw=0,
        label="90 % normal range",
    )
    ax.plot(
        t[win],
        sc.expected_kw[win],
        color=ORANGE,
        lw=1.6,
        ls="--",
        label="expected for time + weather",
    )
    ax.plot(
        t[win],
        sc.actual_kw[win],
        color=BLUE,
        lw=1.6,
        label="actual, with +25 % injected 03:15–14:00 on 2 Jan",
    )
    runs = alert_runs(sc, win)
    for a, b in runs:
        for axx in (ax, ax2):
            axx.axvspan(t[a], t[b], color=RED, alpha=0.10, lw=0)
        kinds = "/".join(
            sorted({k for k in sc.alert_type[a : b + 1].str.split("|").explode() if k})
        )
        ax.text(t[a], 292, f" {kinds}", color=RED, fontsize=9, va="top")
    ax.set_ylim(185, 297)
    ax.set_ylabel("kW")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("What the manager sees — an injected daytime offset, 2 Jan 2020 (local time)")
    ax2.plot(t[win], sc.z[win], color="#374151", lw=1.2, label="z")
    ax2.plot(t[win], sc.cusum_pos[win], color=ORANGE, lw=1.6, label="CUSUM, high side")
    ax2.axhline(5, color=GRAY, lw=1, ls=":")
    ax2.axhline(12, color=RED, lw=1, ls="--")
    ax2.text(t[win].iloc[0], 12.4, "h = 12", color=RED, fontsize=8)
    ax2.text(t[win].iloc[0], 5.4, "spike: z > 5", color="#374151", fontsize=8)
    ax2.set_ylabel("σ  /  CUSUM")
    ax2.legend(loc="upper left", fontsize=8)
    save(fig, "fig9_alert_example")
    n["fig9_alert_runs"] = [(str(t[a]), str(t[b])) for a, b in runs]

    out = json.dumps(n, indent=1, default=str)
    (Path(OUT) / "numbers.json").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
