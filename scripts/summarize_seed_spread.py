"""Per-horizon mean +/- std across seeds for exchange-rate / national-illness.

Answers the question the single-seed tables could not: is the exchange-rate long-horizon
coverage collapse (and the national-illness under-coverage) a real effect, or one unlucky
training run? Reads docs/calibration_results_tsf.csv, so run
scripts/collect_calibration_results.py first.
"""
import argparse
import collections
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "docs", "calibration_results_tsf.csv")

LABEL = {
    "standard_cp": "Standard CP",
    "cpvs": "Adaptive CPVS",
    "cqr_quantile": "CQR quantile",
    "aleatoric_mog": "Aleatoric MoG",
    "aleatoric_only": "Aleatoric only",
}


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["exchange-rate", "national-illness"])
    ap.add_argument("--methods", nargs="+", default=["standard_cp", "cpvs"])
    ap.add_argument("--by-backbone", action="store_true",
                    help="also break the per-horizon rows down by backbone")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(CSV))
            if r["dataset"] in args.datasets and r["method"] in args.methods
            and r["features"] == "M"]

    # One config = everything except the seed. Spread across seeds within a config is the
    # quantity of interest; averaging configs first would hide it.
    per_cfg = collections.defaultdict(dict)
    for r in rows:
        cfg = (r["dataset"], r["method"], int(r["pred_len"]), r["model"],
               r["num_experts"], r["variant"])
        per_cfg[cfg][int(r["seed"])] = (float(r["coverage"]), float(r["width"]))

    for ds in args.datasets:
        print("=" * 92)
        print(ds)
        for method in args.methods:
            sel = {k: v for k, v in per_cfg.items() if k[0] == ds and k[1] == method}
            if not sel:
                continue
            print(f"\n  {LABEL.get(method, method)}")
            print(f"    {'pl':>5} {'cfgs':>5} {'seeds':>6} {'cov mean':>9} {'cov sd':>8} "
                  f"{'cov min':>8} {'cov max':>8} {'width mean':>11} {'width sd':>9}")
            for pl in sorted({k[2] for k in sel}):
                at_pl = {k: v for k, v in sel.items() if k[2] == pl}
                covs = [c for v in at_pl.values() for c, _ in v.values()]
                wids = [w for v in at_pl.values() for _, w in v.values()]
                nseed = sorted({s for v in at_pl.values() for s in v})
                cm, cs = mean_sd(covs)
                wm, ws = mean_sd(wids)
                print(f"    {pl:>5} {len(at_pl):>5} {len(nseed):>6} {cm:>9.4f} {cs:>8.4f} "
                      f"{min(covs):>8.4f} {max(covs):>8.4f} {wm:>11.4f} {ws:>9.4f}")

                if args.by_backbone:
                    for bb in sorted({k[3] for k in at_pl}):
                        sub = {k: v for k, v in at_pl.items() if k[3] == bb}
                        c2 = [c for v in sub.values() for c, _ in v.values()]
                        m2, s2 = mean_sd(c2)
                        print(f"        {bb:<14} n={len(c2):<3} cov {m2:.4f} +/- {s2:.4f}"
                              f"  [{min(c2):.4f}, {max(c2):.4f}]")

    # Within-config seed spread: how much does coverage move when only the seed changes?
    print("=" * 92)
    print("within-config seed spread (configs with >=2 seeds)")
    print(f"  {'dataset':<18} {'method':<16} {'configs':>8} {'median range':>13} {'max range':>10}")
    for ds in args.datasets:
        for method in args.methods:
            ranges = [max(c for c, _ in v.values()) - min(c for c, _ in v.values())
                      for k, v in per_cfg.items()
                      if k[0] == ds and k[1] == method and len(v) >= 2]
            if not ranges:
                continue
            ranges.sort()
            print(f"  {ds:<18} {LABEL.get(method, method):<16} {len(ranges):>8} "
                  f"{ranges[len(ranges) // 2]:>13.4f} {max(ranges):>10.4f}")


if __name__ == "__main__":
    main()
