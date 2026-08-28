"""Fill seeds 4022-4025 for the non-CQR calibrators on exchange-rate / national-illness.

Every cell here already has a trained `model_id=test` checkpoint, so this only ever runs
`run.py --is_training 0`: load weights, forward pass, calibrate. Nothing is retrained and
no checkpoint is written.

Methods: Standard CP (every config) + Adaptive CPVS (prob_expert configs only -- CPVS
consumes the predicted MoG variance, which MOE configs do not have).

Resumable: disk state (checkpoints + result files) is re-scanned on every start, so a
killed run just continues from what is still missing.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

ROOT = "/home/dsi/giladaviv/moe_unc_tsf"
PY = os.path.join(ROOT, "unc_moe", "bin", "python")

DATASETS = ["exchange-rate", "national-illness"]
SEEDS = [4022, 4023, 4024, 4025]

DATA_CFG = {
    "exchange-rate": dict(root="./data/long_term_forecast/exchange_rate/",
                          data_path="exchange_rate.csv", data="custom",
                          enc_in=8, dec_in=8, c_out=8),
    "national-illness": dict(root="./data/long_term_forecast/illness/",
                             data_path="national_illness.csv", data="custom",
                             enc_in=7, dec_in=7, c_out=7),
}

# result file -> method label written into it, and whether the method needs prob_expert
METHOD_FILES = {
    "standard_cp": ("result_calibration_mse_cp.txt", "Standard CP with Sliding Window", False),
    "cpvs": ("result_calibration_cpvs.txt", "Adaptive CPVS Delayed", True),
}

VARIANT_OF = {("0", "0"): "MOE", ("1", "0"): "MOG", ("1", "1"): "MOGU"}

SETTING_RE = re.compile(
    r"^long_term_forecast_test_([A-Za-z]+)_(.+?)_ne(\d+)_pe([01])_ug([01])_"
    r"ft(\w+?)_sl(\d+)_ll(\d+)_pl(\d+)_.*?seed(\d+)"
)

LOG_DIR = os.path.join(ROOT, "logs", "seed_gap")
MANIFEST = os.path.join(LOG_DIR, "manifest.csv")


def parse(name):
    m = SETTING_RE.match(name)
    if not m:
        return None
    model, dataset, ne, pe, ug, feat, sl, ll, pl, seed = m.groups()
    variant = VARIANT_OF.get((pe, ug))
    if variant is None or feat != "M":
        return None
    return dict(model=model, dataset=dataset, ne=int(ne), variant=variant,
                sl=int(sl), ll=int(ll), pl=int(pl), seed=int(seed))


def cell(s):
    return (s["dataset"], s["model"], s["ne"], s["variant"], s["sl"], s["ll"], s["pl"], s["seed"])


def scan_checkpoints():
    """Cells that have real weights on disk, restricted to our datasets/seeds."""
    out = {}
    ckpt_root = os.path.join(ROOT, "checkpoints")
    for name in os.listdir(ckpt_root):
        if not name.startswith("long_term_forecast_test_"):
            continue
        if not os.path.exists(os.path.join(ckpt_root, name, "checkpoint.pth")):
            continue  # dir left behind by a killed run, weights never written
        s = parse(name)
        if s and s["dataset"] in DATASETS and s["seed"] in SEEDS:
            out[cell(s)] = s
    return out


def scan_done():
    """(cell, method) pairs already recorded in the result files."""
    done = set()
    for method, (fname, label, _) in METHOD_FILES.items():
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("long_term_forecast_test_") or not line.endswith(f"({label})"):
                    continue
                s = parse(line.split(" (")[0])
                if s and s["dataset"] in DATASETS and s["seed"] in SEEDS:
                    done.add((cell(s), method))
    return done


def build_jobs():
    ckpts = scan_checkpoints()
    done = scan_done()
    jobs = []
    for key, s in sorted(ckpts.items()):
        prob = s["variant"] in ("MOG", "MOGU")
        need = [m for m, (_, _, needs_prob) in METHOD_FILES.items()
                if (not needs_prob or prob) and (key, m) not in done]
        if need:
            jobs.append((s, need))
    return jobs


def build_cmd(s, methods):
    cfg = DATA_CFG[s["dataset"]]
    args = [
        PY, "-u", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "0",
        "--root_path", cfg["root"],
        "--data_path", cfg["data_path"],
        "--model_id", "test",
        "--model", s["model"],
        "--data", cfg["data"],
        "--features", "M",
        "--seq_len", str(s["sl"]),
        "--label_len", str(s["ll"]),
        "--pred_len", str(s["pl"]),
        "--batch_size", "8",
        "--seed", str(s["seed"]),
        "--num_experts", str(s["ne"]),
        "--enc_in", str(cfg["enc_in"]),
        "--dec_in", str(cfg["dec_in"]),
        "--c_out", str(cfg["c_out"]),
    ]
    if s["variant"] in ("MOG", "MOGU"):
        args.append("--prob_expert")
    if s["variant"] == "MOGU":
        args.append("--unc_gating")
    if "standard_cp" in methods:
        args.append("--do_cp_calibration")
    if "cpvs" in methods:
        args.append("--do_cpvs_calibration")
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_jobs = build_jobs()
    if args.dry_run:
        import collections
        per_ds = collections.Counter()
        per_method = collections.Counter()
        for s, need in all_jobs:
            per_ds[s["dataset"]] += 1
            for m in need:
                per_method[m] += 1
        print(f"{len(all_jobs)} run.py invocations still missing")
        for k, v in sorted(per_ds.items()):
            print(f"  {k:<18} {v} runs")
        for k, v in sorted(per_method.items()):
            print(f"  {k:<18} {v} calibration results")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    new_manifest = not os.path.exists(MANIFEST)
    mf = open(MANIFEST, "a", newline="")
    writer = csv.writer(mf)
    if new_manifest:
        writer.writerow(["dataset", "model", "ne", "variant", "pl", "seed",
                         "methods", "status", "seconds", "log_file"])
        mf.flush()

    # Deterministic ordering -> disjoint partition, workers never touch the same cell.
    jobs = all_jobs[args.worker_id::args.num_workers]
    print(f"[{time.strftime('%F %T')}] worker {args.worker_id}/{args.num_workers} on GPU "
          f"{args.gpu}: {len(jobs)}/{len(all_jobs)} jobs", flush=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for i, (s, need) in enumerate(jobs, 1):
        tag = (f"{s['dataset']}_{s['model']}_ne{s['ne']}_{s['variant']}"
               f"_pl{s['pl']}_seed{s['seed']}")
        log_path = os.path.join(LOG_DIR, f"{tag}.log")
        cmd = build_cmd(s, need)
        print(f"[{time.strftime('%F %T')}] w{args.worker_id} ({i}/{len(jobs)}) "
              f"{tag} [{'+'.join(need)}]", flush=True)
        t0 = time.time()
        with open(log_path, "w") as lf:
            lf.write("CMD: " + " ".join(cmd) + "\n\n")
            lf.flush()
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"fail(rc={proc.returncode})"
        writer.writerow([s["dataset"], s["model"], s["ne"], s["variant"], s["pl"],
                         s["seed"], "+".join(need), status, f"{dt:.1f}", log_path])
        mf.flush()
        print(f"    -> w{args.worker_id} {status} in {dt:.1f}s", flush=True)

    mf.close()
    print(f"[{time.strftime('%F %T')}] worker {args.worker_id} done: {len(jobs)} attempted", flush=True)


if __name__ == "__main__":
    main()
