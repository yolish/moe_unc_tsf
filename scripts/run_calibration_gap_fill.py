"""Fill every remaining cell in the (dataset x backbone x n_experts x MOE/MOG/MOGU) grid
for long_term_forecast: train what has no checkpoint, calibrate (Standard CP + CPVS)
whatever has a checkpoint but no calibration log. Resumable: re-scans disk state before
each run, so a killed/restarted process just continues from what's still missing.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

ROOT = "/home/dsi/giladaviv/moe_unc_tsf"
PY = os.path.join(ROOT, "unc_moe", "bin", "python")

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange-rate", "national-illness"]
HORIZONS = {
    "ETTh1": [96, 192, 336, 720], "ETTh2": [96, 192, 336, 720],
    "ETTm1": [96, 192, 336, 720], "ETTm2": [96, 192, 336, 720],
    "exchange-rate": [96, 192, 336, 720], "national-illness": [24, 36, 48, 60],
}
DATA_CFG = {
    "ETTh1": dict(root="./data/long_term_forecast/ETT/", data_path="ETTh1.csv", data="ETTh1"),
    "ETTh2": dict(root="./data/long_term_forecast/ETT/", data_path="ETTh2.csv", data="ETTh2"),
    "ETTm1": dict(root="./data/long_term_forecast/ETT/", data_path="ETTm1.csv", data="ETTm1"),
    "ETTm2": dict(root="./data/long_term_forecast/ETT/", data_path="ETTm2.csv", data="ETTm2"),
    "exchange-rate": dict(root="./data/long_term_forecast/exchange_rate/", data_path="exchange_rate.csv",
                           data="custom", enc_in=8, dec_in=8, c_out=8),
    "national-illness": dict(root="./data/long_term_forecast/illness/", data_path="national_illness.csv",
                              data="custom"),
}
BACKBONES = ["iTransformer", "DLinear", "PatchTST", "TimeMixer"]
VARIANTS = ["MOE", "MOG", "MOGU"]
N_EXPERTS = [1, 3]
SEED = 4021
VARIANT_OF = {("0", "0"): "MOE", ("1", "0"): "MOG", ("1", "1"): "MOGU"}

SETTING_RE = re.compile(
    r"^long_term_forecast_test_([A-Za-z]+)_(.+?)_ne(\d+)_pe([01])_ug([01])_"
    r"ft(\w+?)_sl(\d+)_ll\d+_pl(\d+)_.*?seed(\d+)"
)

LOG_DIR = os.path.join(ROOT, "logs", "gap_fill")
MANIFEST = os.path.join(LOG_DIR, "manifest.csv")


def parse(name):
    m = SETTING_RE.match(name)
    if not m:
        return None
    model, dataset, ne, pe, ug, feat, sl, pl, seed = m.groups()
    v = VARIANT_OF.get((pe, ug))
    if v is None:
        return None
    return dict(model=model, dataset=dataset, ne=int(ne), variant=v, pl=int(pl), seed=int(seed))


def scan_state():
    ckpt_cells = set()
    ckpt_root = os.path.join(ROOT, "checkpoints")
    for name in os.listdir(ckpt_root):
        if not name.startswith("long_term_forecast"):
            continue
        if not os.path.exists(os.path.join(ckpt_root, name, "checkpoint.pth")):
            continue  # dir exists (e.g. from a killed run) but weights were never written
        s = parse(name)
        if s:
            ckpt_cells.add((s["dataset"], s["model"], s["ne"], s["variant"], s["pl"]))

    cal_cells = set()
    for fname in ["result_calibration_mse_cp.txt", "result_calibration_cpvs.txt"]:
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                if line.startswith("long_term_forecast"):
                    s = parse(line.strip())
                    if s:
                        cal_cells.add((s["dataset"], s["model"], s["ne"], s["variant"], s["pl"]))
    return ckpt_cells, cal_cells


def build_jobs():
    ckpt_cells, cal_cells = scan_state()
    jobs = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for ne in N_EXPERTS:
                for variant in VARIANTS:
                    if variant == "MOGU" and ne == 1:
                        continue
                    for pl in HORIZONS[dataset]:
                        key = (dataset, backbone, ne, variant, pl)
                        has_ckpt = key in ckpt_cells
                        has_cal = key in cal_cells
                        if not has_ckpt:
                            jobs.append(("train", key))
                        elif not has_cal:
                            jobs.append(("calib", key))
    return jobs


def setting_str(model, dataset, ne, variant, pl, model_id="test"):
    pe = 1 if variant in ("MOG", "MOGU") else 0
    ug = 1 if variant == "MOGU" else 0
    return (f"long_term_forecast_{model_id}_{model}_{dataset}_ne{ne}_pe{pe}_ug{ug}_"
            f"ftM_sl96_ll48_pl{pl}_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_"
            f"ebtimeF_dtTrue_test_0_seed{SEED}")


def build_cmd(mode, key):
    dataset, model, ne, variant, pl = key
    cfg = DATA_CFG[dataset]
    args = [
        PY, "-u", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1" if mode == "train" else "0",
        "--root_path", cfg["root"],
        "--data_path", cfg["data_path"],
        "--model_id", "test",
        "--model", model,
        "--data", cfg["data"],
        "--features", "M",
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", str(pl),
        "--batch_size", "8",
        "--seed", str(SEED),
        "--num_experts", str(ne),
        "--do_cp_calibration",
        "--do_cpvs_calibration",
    ]
    if "enc_in" in cfg:
        args += ["--enc_in", str(cfg["enc_in"]), "--dec_in", str(cfg["dec_in"]), "--c_out", str(cfg["c_out"])]
    if variant in ("MOG", "MOGU"):
        args.append("--prob_expert")
    if variant == "MOGU":
        args.append("--unc_gating")
    if dataset == "national-illness" and variant in ("MOG", "MOGU"):
        args += ["--max_grad_norm", "1"]
    if mode == "train":
        args.append("--overwrite")
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value for this worker")
    ap.add_argument("--num-workers", type=int, default=1, help="total number of parallel workers")
    ap.add_argument("--worker-id", type=int, default=0, help="this worker's index in [0, num-workers)")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    new_manifest = not os.path.exists(MANIFEST)
    mf = open(MANIFEST, "a", newline="")
    writer = csv.writer(mf)
    if new_manifest:
        writer.writerow(["mode", "dataset", "model", "ne", "variant", "pl", "status", "seconds", "log_file"])
        mf.flush()

    all_jobs = build_jobs()
    # Deterministic ordering (build_jobs() iterates fixed lists) -> disjoint partition,
    # so parallel workers never touch the same cell.
    jobs = all_jobs[args.worker_id::args.num_workers]
    print(f"[{time.strftime('%F %T')}] worker {args.worker_id}/{args.num_workers} on GPU {args.gpu}: "
          f"{len(jobs)}/{len(all_jobs)} jobs "
          f"({sum(1 for m,_ in jobs if m=='train')} train, "
          f"{sum(1 for m,_ in jobs if m=='calib')} calib-only)", flush=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    done = 0
    for mode, key in jobs:
        dataset, model, ne, variant, pl = key
        tag = f"{mode}_{dataset}_{model}_ne{ne}_{variant}_pl{pl}"
        log_path = os.path.join(LOG_DIR, f"{tag}.log")
        cmd = build_cmd(mode, key)
        print(f"[{time.strftime('%F %T')}] w{args.worker_id} ({done+1}/{len(jobs)}) {tag}", flush=True)
        t0 = time.time()
        with open(log_path, "w") as lf:
            lf.write("CMD: " + " ".join(cmd) + "\n\n")
            lf.flush()
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"fail(rc={proc.returncode})"
        writer.writerow([mode, dataset, model, ne, variant, pl, status, f"{dt:.1f}", log_path])
        mf.flush()
        print(f"    -> w{args.worker_id} {status} in {dt:.1f}s", flush=True)
        done += 1

    mf.close()
    print(f"[{time.strftime('%F %T')}] worker {args.worker_id} done: {done} jobs attempted", flush=True)


if __name__ == "__main__":
    main()
