"""Median aleatoric/epistemic variance ratio (A/E) per dataset, over the MoGE test split.

Reproduces the quantity in the paper's "Median ratio of Aleatoric to Epistemic variance"
table and extends it past ETT. For each (dataset, seed) it loads the MoGE checkpoint
(K=3, --prob_expert, unc_gating=0, H=96), runs the test split, and takes the median of
A/E over every (origin, horizon step, channel) cell; the table then reports mean +/-
sample std across seeds.

A and E come from Exp_Long_Term_Forecast._collect_separated_uncertainty, i.e. the exact
same tensors the CP-MoG calibrator consumes -- not a re-derivation -- so the ratio
describes the variances the calibrators actually see.

The argument namespace is built by replaying run.py's own parser rather than
hand-constructing one, so a default that changes in run.py cannot silently drift here.

    python scripts/compute_ae_ratio.py --gpu 0
    python scripts/compute_ae_ratio.py --gpu 0 --datasets ETTh1,ETTh2
"""
import argparse
import os
import sys
import textwrap

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEEDS = [4021, 4022, 4023, 4024, 4025]

# (root subdir, csv, data loader class, channels, batch size)
DATA = {
    "ETTh1":       ("ETT/", "ETTh1.csv", "ETTh1", 7, 8),
    "ETTh2":       ("ETT/", "ETTh2.csv", "ETTh2", 7, 8),
    "ETTm1":       ("ETT/", "ETTm1.csv", "ETTm1", 7, 8),
    "ETTm2":       ("ETT/", "ETTm2.csv", "ETTm2", 7, 8),
    "electricity": ("electricity/", "electricity.csv", "custom", 321, 4),
    "traffic":     ("traffic/", "traffic.csv", "custom", 862, 4),
}
ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic"]


def build_args(dataset, seed, gpu):
    """Replay run.py's parser under a controlled argv, so every default matches a real run."""
    src = open(os.path.join(ROOT, "run.py")).read().splitlines()
    start = next(i for i, l in enumerate(src) if l.startswith("if __name__"))
    end = next(i for i, l in enumerate(src) if l.strip().startswith("args = parser.parse_args()"))
    body = textwrap.dedent("\n".join(src[start + 1:end + 1]))

    sub, csv_name, loader, ch, bs = DATA[dataset]
    argv = ["run.py",
            "--task_name", "long_term_forecast", "--is_training", "0",
            "--root_path", f"./data/long_term_forecast/{sub}", "--data_path", csv_name,
            "--model_id", "test", "--model", "iTransformer", "--data", loader,
            "--features", "M", "--seq_len", "96", "--label_len", "48", "--pred_len", "96",
            "--batch_size", str(bs), "--enc_in", str(ch), "--dec_in", str(ch),
            "--c_out", str(ch), "--seed", str(seed), "--num_experts", "3", "--prob_expert"]

    ns = {"argparse": __import__("argparse"), "__name__": "__main__"}
    old = sys.argv
    sys.argv = argv
    try:
        exec(compile(body, "run.py<parser>", "exec"), ns)
    finally:
        sys.argv = old
    args = ns["args"]

    # run.py sets these right after parse_args; mirror them.
    args.moe = (args.num_experts > 1) or args.prob_expert
    args.use_gpu = True
    args.gpu = gpu
    args.use_multi_gpu = False
    args.devices = str(gpu)
    return args


def setting_of(args, dataset):
    return ('long_term_forecast_{}_{}_{}_ne{}_pe{}_ug{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_'
            'el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}_seed{}').format(
        args.model_id, args.model, dataset, args.num_experts, int(args.prob_expert),
        int(args.unc_gating), args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.expand,
        args.d_conv, args.factor, args.embed, args.distil, args.des, 0, args.seed)


def ratio_for(dataset, seed, gpu):
    import torch
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    args = build_args(dataset, seed, gpu)
    setting = setting_of(args, dataset)
    ckpt = os.path.join(ROOT, "checkpoints", setting, "checkpoint.pth")
    if not os.path.exists(ckpt):
        return None, "no checkpoint"

    exp = Exp_Long_Term_Forecast(args)
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()
    _, ale, epi, _, _ = exp._collect_separated_uncertainty("test")

    # Cells with a numerically zero epistemic term would make the ratio infinite; they are
    # dropped rather than clipped, so the median is taken over cells where the ratio is
    # actually defined. The count is reported so a heavy drop is visible, not silent.
    a = np.asarray(ale, dtype=np.float64).ravel()
    e = np.asarray(epi, dtype=np.float64).ravel()
    ok = e > 0
    ratio = a[ok] / e[ok]
    return float(np.median(ratio)), f"{ok.sum()}/{ok.size} cells"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--datasets", default=",".join(ORDER))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    want = [d.strip() for d in a.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    os.chdir(ROOT)

    print(f"{'dataset':14s} {'seed':6s} {'median A/E':>12s}   note", flush=True)
    out = {}
    for ds in want:
        vals = []
        for s in seeds:
            try:
                r, note = ratio_for(ds, s, a.gpu)
            except Exception as exc:                       # keep going; report at the end
                r, note = None, f"ERROR {type(exc).__name__}: {exc}"
            print(f"{ds:14s} {s:<6d} {('%.1f' % r) if r is not None else '--':>12s}   {note}",
                  flush=True)
            if r is not None:
                vals.append(r)
        out[ds] = vals

    print("\n=== median A/E, mean +/- sample std across seeds ===", flush=True)
    for ds in want:
        v = out[ds]
        if not v:
            print(f"{ds:14s} no data")
        elif len(v) == 1:
            print(f"{ds:14s} {v[0]:.0f}  (1 seed, no std)")
        else:
            print(f"{ds:14s} {np.mean(v):.0f} +/- {np.std(v, ddof=1):.0f}   (n={len(v)})")


if __name__ == "__main__":
    main()
