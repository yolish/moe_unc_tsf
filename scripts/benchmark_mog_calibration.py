"""Benchmark CP-DVS against the variance-scaling conformal baselines on ETT.

Each checkpoint is run through the model once and the law-of-total-variance decomposition
(mu_bar, sigma^2_within, sigma^2_between) is handed to every calibrator, so all methods see
byte-identical inputs.

All methods share one protocol: fit on the validation split, evaluate on the test split,
with the conformal quantile taken per (horizon step, channel). CP-DVS additionally splits
the validation block in two (scale model / conformal quantile), so it calibrates its
quantile on *half* the data the baselines get -- a handicap, not an advantage.

Stages
------
  `--stage run`     (default) inference + evaluation over the requested grid; writes the
                    CSV incrementally so a long run survives a late failure. Pass
                    --cache_dir to also persist the decompositions for later reuse.
  `--stage eval`    re-evaluate from cached decompositions only (no model inference).
  `--stage report`  regenerate the Markdown summary from an existing CSV.

Usage
-----
    python scripts/benchmark_mog_calibration.py --stage run --model iTransformer \
        --datasets ETTh1 ETTh2 ETTm1 ETTm2 --pred_lens 96 192 336 720 --seeds 4021 4022 4023
    python scripts/benchmark_mog_calibration.py --stage report
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.cp_dvs_calibration import (  # noqa: E402
    CPDVSCalibrator, MoGPrediction, conformal_quantile_level, EPS,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHAS = (0.05, 0.10)


# ----------------------------------------------------------------------------------
# Stage 1: cache MoG head outputs
# ----------------------------------------------------------------------------------

def gpu_is_usable():
    """True only if a CUDA device exists *and* this torch build has kernels for it.

    The torch install here (1.7.1 / CUDA 10.2) predates sm_80, so the A100s raise
    "no kernel image is available" at the first kernel launch rather than at init.
    Checking the compute capability against torch's compiled arch list catches that up
    front instead of failing halfway through a cache run.
    """
    if not torch.cuda.is_available():
        return False
    try:
        archs = {int(a.split("_")[1]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")}
        major, minor = torch.cuda.get_device_capability(0)
        return (major * 10 + minor) in archs
    except Exception:
        return False


def build_args_from_runpy(overrides):
    """Instantiate run.py's argparse defaults without executing run.py's main body.

    The parser block is exec'd out of run.py's source so this harness cannot silently
    drift from the defaults the training runs actually used.
    """
    src = open(os.path.join(REPO, "run.py")).read().split("\n")
    start = next(i for i, l in enumerate(src) if "argparse.ArgumentParser" in l)
    end = next(i for i, l in enumerate(src) if "args = parser.parse_args()" in l)
    block = "\n".join(l[4:] if l.startswith("    ") else l for l in src[start:end])
    ns = {"argparse": argparse}
    exec(block, ns)  # noqa: S102 - executing our own repo's parser definition
    parser = ns["parser"]

    argv = []
    for k, v in overrides.items():
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
        else:
            argv.extend([f"--{k}", str(v)])
    args = parser.parse_args(argv)
    args.moe = (args.num_experts > 1) or args.prob_expert
    # --use_gpu is declared type=bool in run.py, so it cannot be switched off from argv
    # (bool("False") is True); set it here instead.
    args.use_gpu = gpu_is_usable()
    args.device = torch.device(f"cuda:{args.gpu}") if args.use_gpu else torch.device("cpu")
    return args


SETTING_RE = re.compile(
    r"^long_term_forecast_(?P<model_id>[^_]+)_(?P<model>[^_]+)_(?P<data>[^_]+)"
    r"_ne(?P<ne>\d+)_pe(?P<pe>\d+)_ug(?P<ug>\d+)_ft(?P<ft>[A-Z]+)"
    r"_sl(?P<sl>\d+)_ll(?P<ll>\d+)_pl(?P<pl>\d+)_dm(?P<dm>\d+)_nh(?P<nh>\d+)"
    r"_el(?P<el>\d+)_dl(?P<dl>\d+)_df(?P<df>\d+)_expand(?P<expand>\d+)_dc(?P<dc>\d+)"
    r"_fc(?P<fc>\d+)_eb(?P<eb>[a-zA-Z]+)_dt(?P<dt>True|False)_(?P<des>[^_]+)"
    r"_(?P<ii>\d+)_seed(?P<seed>\d+)$")

DATA_PATHS = {"ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv",
              "ETTm1": "ETTm1.csv", "ETTm2": "ETTm2.csv"}


def setting_to_args(setting):
    m = SETTING_RE.match(setting)
    if m is None:
        raise ValueError(f"Unparseable setting: {setting}")
    g = m.groupdict()
    return build_args_from_runpy({
        "task_name": "long_term_forecast", "is_training": 0,
        "model_id": g["model_id"], "model": g["model"], "data": g["data"],
        "root_path": "./data/long_term_forecast/ETT/", "data_path": DATA_PATHS[g["data"]],
        "features": g["ft"], "seq_len": int(g["sl"]), "label_len": int(g["ll"]),
        "pred_len": int(g["pl"]), "num_experts": int(g["ne"]),
        "prob_expert": g["pe"] == "1", "unc_gating": g["ug"] == "1",
        "d_model": int(g["dm"]), "n_heads": int(g["nh"]), "e_layers": int(g["el"]),
        "d_layers": int(g["dl"]), "d_ff": int(g["df"]), "expand": int(g["expand"]),
        "d_conv": int(g["dc"]), "factor": int(g["fc"]), "embed": g["eb"],
        "des": g["des"], "seed": int(g["seed"]), "batch_size": 32,
    }), g


@torch.no_grad()
def collect_mog(exp, flag, pred_len, features):
    """Run one split through the MoG model, returning the decomposition and targets.

    The law-of-total-variance reduction is applied per batch so the [N, K, H, C] tensors
    are never materialized for the whole split: at pred_len=720 on ETTm1 that is the
    difference between ~2 GB and ~0.9 GB of resident arrays.

    Returns
    -------
    mu_bar, v_within, v_between, y : float32 arrays [N, H, C]
    """
    _, loader = exp._get_data(flag=flag)
    mb, vw, vb, ys = [], [], [], []
    f_dim = -1 if features == "MS" else 0
    for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)

        outputs, expert_unc, expert_weights = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # [B, K, L, C] -> keep the forecast horizon and the evaluated channels only
        mu = outputs[:, :, -pred_len:, f_dim:]
        var = expert_unc[:, :, -pred_len:, f_dim:].clamp_min(0.0)
        pi = expert_weights[:, :, -pred_len:, f_dim:]
        pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-12)

        mu_bar = (pi * mu).sum(dim=1)
        mb.append(mu_bar.cpu().numpy().astype(np.float32))
        vw.append((pi * var).sum(dim=1).cpu().numpy().astype(np.float32))
        vb.append((pi * (mu - mu_bar.unsqueeze(1)) ** 2).sum(dim=1).cpu().numpy().astype(np.float32))
        ys.append(batch_y[:, -pred_len:, f_dim:].cpu().numpy().astype(np.float32))
    return (np.concatenate(mb), np.concatenate(vw),
            np.concatenate(vb), np.concatenate(ys))


def enumerate_settings(cli):
    """Settings in the requested grid that actually have a checkpoint on disk."""
    available = set(os.listdir(os.path.join(REPO, "checkpoints")))
    todo = []
    # Seed outermost so that a run interrupted partway still covers every
    # (dataset, horizon) cell at least once rather than a few cells deeply.
    for seed in cli.seeds:
        for data in cli.datasets:
            for pl in cli.pred_lens:
                s = (f"long_term_forecast_test_{cli.model}_{data}_ne{cli.num_experts}_pe1"
                     f"_ug{cli.unc_gating}_ftM_sl96_ll48_pl{pl}_dm512_nh8_el2_dl1_df2048"
                     f"_expand2_dc4_fc1_ebtimeF_dtTrue_test_0_seed{seed}")
                if s in available:
                    todo.append(s)
                else:
                    print(f"[skip] no checkpoint: {s}")
    return todo


def infer_setting(setting, cache_dir=None, overwrite=False):
    """Return (cal, test) dicts of MoGPrediction + targets for one checkpoint.

    Reads a cached decomposition when one exists; otherwise runs the model and
    (optionally) writes the cache.
    """
    cache_path = os.path.join(cache_dir, setting + ".npz") if cache_dir else None
    if cache_path and os.path.exists(cache_path) and not overwrite:
        z = np.load(cache_path)
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        args, _ = setting_to_args(setting)
        exp = Exp_Long_Term_Forecast(args)
        ckpt = os.path.join(REPO, "checkpoints", setting, "checkpoint.pth")
        exp.model.load_state_dict(torch.load(ckpt, map_location=args.device))
        exp.model.eval()
        z = {}
        for flag in ("val", "test"):
            mb, vw, vb, y = collect_mog(exp, flag, args.pred_len, args.features)
            z[f"{flag}_mu_bar"], z[f"{flag}_v_within"] = mb, vw
            z[f"{flag}_v_between"], z[f"{flag}_y"] = vb, y
        del exp
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache_path, **z)

    def pack(flag):
        return {"pred": MoGPrediction.from_decomposition(
                    z[f"{flag}_mu_bar"], z[f"{flag}_v_within"], z[f"{flag}_v_between"]),
                "y": z[f"{flag}_y"].astype(np.float64)}

    return pack("val"), pack("test")


# ----------------------------------------------------------------------------------
# Stage 2: calibrators
# ----------------------------------------------------------------------------------

def _split_conformal(scores_cal, alpha):
    """Per-(horizon, channel) conformal quantile of a [N, H, C] score array."""
    n = scores_cal.shape[0]
    return np.quantile(scores_cal, conformal_quantile_level(n, alpha), axis=0, method="higher")


def run_scaled_cp(cal, test, alpha, scale_fn):
    """Generic variance-scaled split CP: score = |y - mu_bar| / scale(x)."""
    u_cal = np.maximum(scale_fn(cal["pred"]), EPS)
    scores = np.abs(cal["y"] - cal["pred"].mu_bar) / u_cal
    q = _split_conformal(scores, alpha)
    half = q * np.maximum(scale_fn(test["pred"]), EPS)
    return test["pred"].mu_bar - half, test["pred"].mu_bar + half


def run_variance_ratio_cp(cal, test, alpha, var_fn):
    """Variance-scaling CP on squared residuals: score = r^2 / var(x), width = sqrt(q*var).

    This is the AleatoricOnlyCalibrator family (q calibrates a *variance* multiplier).
    Note it is algebraically identical to `run_scaled_cp` with scale sqrt(var): the
    quantile of r^2/v is the square of the quantile of r/sqrt(v) under a monotone
    transform, and both take the same 'higher' order statistic. It is kept as a separate
    entry so the equivalence is visible in the results rather than assumed.
    """
    v_cal = np.maximum(var_fn(cal["pred"]), EPS)
    scores = (cal["y"] - cal["pred"].mu_bar) ** 2 / v_cal
    q = np.maximum(_split_conformal(scores, alpha), 0.0)
    half = np.sqrt(np.maximum(q * np.maximum(var_fn(test["pred"]), EPS), 0.0))
    return test["pred"].mu_bar - half, test["pred"].mu_bar + half


def run_cp_dvs(cal, test, alpha, **kw):
    c = CPDVSCalibrator(alpha=alpha, **kw)
    c.calibrate(cal["pred"], cal["y"])
    lo, hi = c.predict_intervals(test["pred"])
    return lo, hi, c


METHODS = {
    "standard_cp": lambda cal, test, a: run_scaled_cp(
        cal, test, a, lambda p: np.ones_like(p.mu_bar)),
    "cpvs": lambda cal, test, a: run_scaled_cp(
        cal, test, a, lambda p: np.sqrt(np.maximum(p.v_total, EPS))),
    "cpvs_within": lambda cal, test, a: run_scaled_cp(
        cal, test, a, lambda p: np.sqrt(np.maximum(p.v_within, EPS))),
    "aleatoric_only": lambda cal, test, a: run_variance_ratio_cp(
        cal, test, a, lambda p: p.v_within),
    "cp_dvs": lambda cal, test, a: run_cp_dvs(cal, test, a)[:2],
}


def metrics(y, lo, hi, alpha, n_size_bins=5, n_time_blocks=10):
    """Success metrics for a set of prediction intervals, arrays [N, H, C].

    Marginal coverage and mean width are necessary but jointly gameable: a method can
    trade one against the other, and both are blind to *where* the coverage sits. The
    additional metrics below are grouped by the failure each is meant to catch.

    Joint quality (cannot be gamed by trading coverage against width)
        interval_score       Winkler/interval score, the proper scoring rule for a
                             central (1-alpha) interval:
                                 (u-l) + (2/a)(l-y)1{y<l} + (2/a)(y-u)1{y>u}
                             Lower is better. This is the right single headline number:
                             an over-wide interval pays through (u-l), an under-covering
                             one pays through the 2/a penalty.
        interval_score_norm  Same, divided by the target's mean absolute deviation, so it
                             can be averaged across datasets/channels of different scale.
        pinball_lo/hi        Pinball loss of the two endpoints at levels a/2 and 1-a/2.

    Conditional coverage (marginal coverage hides all of this)
        ssc_min_coverage     Size-Stratified Coverage: bin test points into equal-count
                             bins by interval width and take the worst bin's coverage.
                             A method that covers on easy points and misses on hard ones
                             is marginally valid and practically useless; this is the
                             standard diagnostic for it.
        ssc_max_gap          Largest |bin coverage - nominal| over those bins.
        worst_horizon_cov    Min over horizon steps h of coverage(h). Forecast error grows
                             with h, so a calibrator can hold marginal coverage while
                             systematically under-covering the far horizon.
        horizon_cov_mae      Mean over h of |coverage(h) - nominal|.
        worst_channel_cov    Same idea across the C channels.

    Temporal robustness (time series specific)
        min_block_coverage   Split the test period into `n_time_blocks` contiguous blocks
                             and take the worst block's coverage. Exposes calibration
                             decaying under distribution shift, which marginal coverage
                             over the whole test set averages away.
        max_block_gap        Largest |block coverage - nominal|.
        violation_clustering P(miss_t | miss_{t-1}) / P(miss) along the time axis. 1.0 is
                             independent violations; >1 means misses arrive in bursts,
                             which is far worse operationally than the same number spread
                             out. (Overlapping forecast windows inflate this for every
                             method alike, so read it comparatively.)

    Adaptivity / efficiency shape
        width_cv             std(width)/mean(width). Standard CP scores 0 by construction;
                             a variance-scaled method that also scores ~0 is not actually
                             adapting to anything.
        avg_width_norm       Mean width over the target's mean absolute deviation.
    """
    y = np.asarray(y, dtype=np.float64)
    covered = (y >= lo) & (y <= hi)
    width = np.asarray(hi - lo, dtype=np.float64)
    nominal = 1.0 - alpha

    # Scale reference for cross-dataset comparability.
    y_mad = float(np.mean(np.abs(y - np.mean(y)))) or 1.0

    interval_score = (width
                      + (2.0 / alpha) * np.clip(lo - y, 0.0, None)
                      + (2.0 / alpha) * np.clip(y - hi, 0.0, None))

    def pinball(pred, tau):
        e = y - pred
        return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))

    out = {
        "coverage": float(np.mean(covered)),
        "coverage_gap": float(np.mean(covered)) - nominal,
        "avg_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "avg_width_norm": float(np.mean(width)) / y_mad,
        "interval_score": float(np.mean(interval_score)),
        "interval_score_norm": float(np.mean(interval_score)) / y_mad,
        "pinball_lo": pinball(lo, alpha / 2.0),
        "pinball_hi": pinball(hi, 1.0 - alpha / 2.0),
        "width_cv": float(np.std(width) / max(np.mean(width), EPS)),
    }

    # --- Size-stratified coverage: equal-count bins on interval width.
    w_flat, c_flat = width.ravel(), covered.ravel()
    edges = np.quantile(w_flat, np.linspace(0.0, 1.0, n_size_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_id = np.searchsorted(edges, w_flat, side="right") - 1
    bin_id = np.clip(bin_id, 0, n_size_bins - 1)
    bin_cov = np.array([c_flat[bin_id == b].mean() if np.any(bin_id == b) else np.nan
                        for b in range(n_size_bins)])
    out["ssc_min_coverage"] = float(np.nanmin(bin_cov))
    out["ssc_max_gap"] = float(np.nanmax(np.abs(bin_cov - nominal)))

    # --- Coverage by horizon step and by channel.
    cov_h = covered.mean(axis=(0, 2))
    cov_c = covered.mean(axis=(0, 1))
    out["worst_horizon_cov"] = float(cov_h.min())
    out["horizon_cov_mae"] = float(np.mean(np.abs(cov_h - nominal)))
    out["worst_channel_cov"] = float(cov_c.min())

    # --- Coverage over contiguous blocks of the test period.
    n = covered.shape[0]
    blocks = min(n_time_blocks, n)
    block_cov = np.array([b.mean() for b in np.array_split(covered, blocks, axis=0)])
    out["min_block_coverage"] = float(block_cov.min())
    out["max_block_gap"] = float(np.max(np.abs(block_cov - nominal)))

    # --- Are violations independent in time, or bursty?
    miss = ~covered
    p_miss = float(miss.mean())
    if n > 1 and p_miss > 0:
        prev = miss[:-1]
        p_prev = float(prev.mean())
        p_joint = float((miss[1:] & prev).mean())
        cond = p_joint / p_prev if p_prev > 0 else np.nan
        out["violation_clustering"] = float(cond / p_miss) if p_miss > 0 else float("nan")
    else:
        out["violation_clustering"] = float("nan")

    return out


def evaluate_settings(cli, todo):
    """Run every method at every alpha over `todo`, writing the CSV incrementally."""
    import csv

    rows = []
    for i, setting in enumerate(todo, 1):
        t0 = time.time()
        g = SETTING_RE.match(setting).groupdict()
        cal, test = infer_setting(setting, cli.cache_dir, cli.overwrite)

        for alpha in ALPHAS:
            for name, fn in METHODS.items():
                lo, hi = fn(cal, test, alpha)
                rows.append({
                    "dataset": g["data"], "model": g["model"], "pred_len": int(g["pl"]),
                    "num_experts": int(g["ne"]), "unc_gating": int(g["ug"]),
                    "seed": int(g["seed"]), "alpha": alpha, "method": name,
                    "target_coverage": 1 - alpha, **metrics(test["y"], lo, hi, alpha),
                    "n_cal": int(cal["y"].shape[0]), "n_test": int(test["y"].shape[0]),
                })
        del cal, test

        # Write after every checkpoint so a long run is never lost to a late failure.
        with open(cli.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[{i}/{len(todo)}] {g['data']} pl={g['pl']} seed={g['seed']} "
              f"({time.time() - t0:.1f}s)")

    print(f"\nWrote {len(rows)} rows -> {cli.out_csv}")
    write_markdown(rows, cli.out_md)


def stage_run(cli):
    todo = enumerate_settings(cli)
    if not todo:
        raise SystemExit("No checkpoints matched the requested grid.")
    print(f"Evaluating {len(todo)} checkpoint(s)\n")
    evaluate_settings(cli, todo)


def stage_eval(cli):
    """Re-evaluate from cached decompositions only (no model inference)."""
    if not cli.cache_dir or not os.path.isdir(cli.cache_dir):
        raise SystemExit("--stage eval needs a populated --cache_dir.")
    todo = sorted(f[:-4] for f in os.listdir(cli.cache_dir) if f.endswith(".npz"))
    if not todo:
        raise SystemExit(f"No cached decompositions in {cli.cache_dir}.")
    cli.overwrite = False
    evaluate_settings(cli, todo)


METHOD_ORDER = ["standard_cp", "cpvs", "cpvs_within", "aleatoric_only", "cp_dvs"]


def write_markdown(rows, path):
    """Aggregate the tidy rows into per-dataset, per-horizon and headline tables."""
    order = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    datasets = sorted({r["dataset"] for r in rows})
    alphas = sorted({r["alpha"] for r in rows})

    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    def agg(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else float("nan")

    n_runs = len({(r["dataset"], r["pred_len"], r["seed"]) for r in rows})
    lines = [
        "# MoG conformal calibration on ETT", "",
        "Split-conformal protocol, identical for every method: fit on the validation "
        "split, evaluate on the test split, conformal quantile taken per "
        "(horizon step, channel). CP-DVS additionally splits validation 50/50 into "
        "scale-model fit and conformal quantile, so its quantile sees **half** the "
        "calibration data the baselines get.", "",
        "`cpvs_within` and `aleatoric_only` are algebraically the same estimator "
        "(variance-ratio CP on the within-component); both are listed so the equivalence "
        "is visible in the numbers.", "",
        "Time series are not exchangeable, so coverage here is empirical, not guaranteed "
        "-- for every method alike.", "",
        f"Aggregated over {n_runs} (dataset, horizon, seed) runs.", "",
    ]

    for alpha in alphas:
        lines += [f"## alpha = {alpha:.2f}  (target coverage {1 - alpha:.2f})", "",
                  "### By dataset", "",
                  "| dataset | metric | " + " | ".join(order) + " |",
                  "|---|---|" + "---|" * len(order)]
        for ds in datasets:
            cov = [f"{agg(sel(dataset=ds, alpha=alpha, method=m), 'coverage'):.4f}" for m in order]
            wid = [f"{agg(sel(dataset=ds, alpha=alpha, method=m), 'avg_width'):.4f}" for m in order]
            lines.append(f"| {ds} | coverage | " + " | ".join(cov) + " |")
            lines.append(f"| {ds} | avg width | " + " | ".join(wid) + " |")

        lines += ["", "### By horizon (avg width)", "",
                  "| pred_len | " + " | ".join(order) + " | CP-DVS vs CP-VS |",
                  "|---|" + "---|" * (len(order) + 1)]
        for pl in sorted({r["pred_len"] for r in rows}):
            w = {m: agg(sel(pred_len=pl, alpha=alpha, method=m), "avg_width") for m in order}
            rel = 100.0 * (w["cp_dvs"] / w["cpvs"] - 1.0) if w.get("cpvs") else float("nan")
            lines.append(f"| {pl} | " + " | ".join(f"{w[m]:.4f}" for m in order)
                         + f" | {rel:+.2f}% |")

        base = {m: agg(sel(alpha=alpha, method=m), "avg_width") for m in order}
        base_is = {m: agg(sel(alpha=alpha, method=m), "interval_score_norm") for m in order}
        lines += ["", "### Overall -- headline", "",
                  "| method | coverage | avg width | width vs CP-VS | interval score (norm) "
                  "| IS vs CP-VS |", "|---|---|---|---|---|---|"]
        for m in order:
            rs = sel(alpha=alpha, method=m)
            rel = 100.0 * (base[m] / base["cpvs"] - 1.0) if base.get("cpvs") else float("nan")
            rel_is = (100.0 * (base_is[m] / base_is["cpvs"] - 1.0)
                      if base_is.get("cpvs") else float("nan"))
            lines.append(f"| {m} | {agg(rs, 'coverage'):.4f} | {base[m]:.4f} | {rel:+.2f}% | "
                         f"{base_is[m]:.4f} | {rel_is:+.2f}% |")

        lines += ["", "### Overall -- conditional coverage and robustness", "",
                  "(nominal = target coverage; closer to nominal is better for the "
                  "coverage columns, lower is better for gaps and clustering)", "",
                  "| method | SSC min cov | SSC max gap | worst horizon cov | horizon cov MAE "
                  "| worst channel cov | min block cov | violation clustering | width CV |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m in order:
            rs = sel(alpha=alpha, method=m)
            lines.append(
                f"| {m} | {agg(rs, 'ssc_min_coverage'):.4f} | {agg(rs, 'ssc_max_gap'):.4f} | "
                f"{agg(rs, 'worst_horizon_cov'):.4f} | {agg(rs, 'horizon_cov_mae'):.4f} | "
                f"{agg(rs, 'worst_channel_cov'):.4f} | {agg(rs, 'min_block_coverage'):.4f} | "
                f"{agg(rs, 'violation_clustering'):.3f} | {agg(rs, 'width_cv'):.4f} |")

        # Per-run head-to-head: how often is CP-DVS narrower than CP-VS while still
        # holding nominal coverage? Aggregate means can hide a method that wins on
        # average by losing badly on a few cells.
        keys = {(r["dataset"], r["pred_len"], r["seed"]) for r in rows}
        wins = valid = 0
        for k in keys:
            d = {r["method"]: r for r in rows
                 if (r["dataset"], r["pred_len"], r["seed"]) == k and r["alpha"] == alpha}
            if "cp_dvs" not in d or "cpvs" not in d:
                continue
            valid += 1
            if d["cp_dvs"]["avg_width"] < d["cpvs"]["avg_width"]:
                wins += 1
        held = sum(1 for k in keys for r in rows
                   if (r["dataset"], r["pred_len"], r["seed"]) == k and r["alpha"] == alpha
                   and r["method"] == "cp_dvs" and r["coverage"] >= 1 - alpha)
        lines += ["", f"CP-DVS narrower than CP-VS on **{wins}/{valid}** runs; "
                      f"CP-DVS held nominal coverage on **{held}/{valid}** runs.", ""]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote Markdown summary -> {path}")


def stage_report(cli):
    """Regenerate the Markdown summary from an existing CSV (no inference, no caches)."""
    import csv
    with open(cli.out_csv) as f:
        rows = []
        for r in csv.DictReader(f):
            for k in ("pred_len", "num_experts", "unc_gating", "seed", "n_cal", "n_test"):
                r[k] = int(r[k])
            for k, v in r.items():
                if k not in ("dataset", "model", "method") and not isinstance(v, int):
                    r[k] = float(v)
            rows.append(r)
    print(f"Read {len(rows)} rows from {cli.out_csv}")
    write_markdown(rows, cli.out_md)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["run", "eval", "report"], default="run")
    p.add_argument("--cache_dir", default=None,
                   help="Optional directory for cached decompositions. Omit to keep the "
                        "run purely in memory (long horizons on ETTm are ~1 GB each).")
    p.add_argument("--model", default="iTransformer")
    p.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    p.add_argument("--pred_lens", nargs="+", type=int, default=[96, 192, 336, 720])
    p.add_argument("--seeds", nargs="+", type=int, default=[4021, 4022, 4023, 4024, 4025])
    p.add_argument("--num_experts", type=int, default=3)
    p.add_argument("--unc_gating", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out_csv", default="./results_mog_calibration.csv")
    p.add_argument("--out_md", default="./results_mog_calibration.md")
    cli = p.parse_args()

    if cli.stage == "run":
        stage_run(cli)
    elif cli.stage == "eval":
        stage_eval(cli)
    else:
        stage_report(cli)


if __name__ == "__main__":
    main()
