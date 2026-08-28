"""Wide calibration sweep: expert counts 1-5 x 9 datasets x every applicable calibrator.

Design notes
------------
One run.py invocation per (dataset, n_experts, variant) cell, carrying the flags for every
calibrator that cell is still missing. run.py's own guards print "Skipping ..." for methods
that need --prob_expert on a MOE cell, so passing a flag that does not apply is free -- and
one model load then serves a dozen calibrators instead of reloading per method.

Work is claimed through atomic O_EXCL claim files rather than a static index partition, so a
worker that draws a 40-minute traffic job does not leave the other idle. Both workers share
one job list; either can take any job.

Resumable. Before each job the driver rescans the result_calibration_*.txt files and drops
methods that already have a row for that exact setting string, so a killed run resumes at
method granularity, not cell granularity.

Priority is cheap-and-informative first: the four ETT datasets already have checkpoints for
ne 1-5 across MOE/MOG/MOGU (56 cells, calibration only), so those land before anything that
needs training, and electricity/traffic training -- hours per cell on 321/862 channels with
another tenant holding ~150 GB of the two A100s -- goes last.

    python scripts/run_campaign_25h.py --gpu 0 --worker-id 0 --deadline-hours 24
    python scripts/run_campaign_25h.py --gpu 1 --worker-id 1 --deadline-hours 24
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
SEED = 4021
RUN_DIR = os.path.join(ROOT, "logs", "campaign")
CLAIM_DIR = os.path.join(RUN_DIR, "claims")
MANIFEST = os.path.join(RUN_DIR, "manifest.csv")

DATA = {
    "ETTh1":            dict(root="ETT/",           file="ETTh1.csv",            data="ETTh1",  ch=7,   pl=96),
    "ETTh2":            dict(root="ETT/",           file="ETTh2.csv",            data="ETTh2",  ch=7,   pl=96),
    "ETTm1":            dict(root="ETT/",           file="ETTm1.csv",            data="ETTm1",  ch=7,   pl=96),
    "ETTm2":            dict(root="ETT/",           file="ETTm2.csv",            data="ETTm2",  ch=7,   pl=96),
    "exchange-rate":    dict(root="exchange_rate/", file="exchange_rate.csv",    data="custom", ch=8,   pl=96),
    "national-illness": dict(root="illness/",       file="national_illness.csv", data="custom", ch=7,   pl=24),
    "weather":          dict(root="weather/",       file="weather.csv",          data="custom", ch=21,  pl=96),
    "electricity":      dict(root="electricity/",   file="electricity.csv",      data="custom", ch=321, pl=96),
    "traffic":          dict(root="traffic/",       file="traffic.csv",          data="custom", ch=862, pl=96),
}
PE_UG = {"MOE": ("0", "0"), "MOG": ("1", "0"), "MOGU": ("1", "1")}

# (flag, result file, label prefix written into that file, needs_prob_expert)
METHODS = [
    ("--do_cp_calibration",                          "result_calibration_mse_cp.txt",                       "Standard CP with Sliding Window", False),
    ("--do_cp_dvs_calibration",                      "result_calibration_cp_dvs.txt",                       "CP-DVS",                          False),
    ("--do_cpvs_calibration",                        "result_calibration_cpvs.txt",                         "Adaptive CPVS Delayed",           True),
    ("--do_aleatoric_only_calibration",              "result_calibration_aleatoric_only.txt",               "Aleatoric Only CP Delayed",       True),
    ("--do_aleatoric_scale_calibration",             "result_calibration_aleatoric_scale_tsf.txt",          "Aleatoric Scale CP Delayed",      True),
    ("--do_aci_aleatoric_scale_calibration",         "result_calibration_aci_aleatoric_scale_tsf.txt",      "ACI Aleatoric Scale CP Delayed",  True),
    ("--do_aci_aleatoric_scale_g001_calibration",    "result_calibration_aci_aleatoric_scale_g001_tsf.txt", "ACI Aleatoric Scale CP (g=0.001)", True),
    ("--do_aleatoric_mog_calibration",               "result_calibration_aleatoric_mog.txt",                "Aleatoric MoG CP Delayed",        True),
    ("--do_aleatoric_mog_calibration_second_option", "result_calibration_aleatoric_mog_v2.txt",             "second option",                   True),
    ("--do_adaptive_variance_calibration",           "result_calibration_adaptive_variance_tsf.txt",        "Adaptive Variance CP Delayed",    True),
    ("--do_adaptive_window_calibration",             "result_calibration_adaptive_window_tsf.txt",          "Adaptive Window CP Delayed",      True),
    ("--do_moecp_calibration",                       "result_calibration_moecp_tsf.txt",                    "MoECP delayed",                   True),
]
CQR_METHOD = ("--do_cqr_calibration", "result_calibration_cqr_quantile.txt", "CQR Quantile", False)


def setting(dataset, ne, variant, model_id="test"):
    pe, ug = PE_UG[variant]
    return (f"long_term_forecast_{model_id}_iTransformer_{dataset}_ne{ne}_pe{pe}_ug{ug}_"
            f"ftM_sl96_ll48_pl{DATA[dataset]['pl']}_dm512_nh8_el2_dl1_df2048_expand2_dc4_"
            f"fc1_ebtimeF_dtTrue_test_0_seed{SEED}")


def has_result(result_file, label, setting_str):
    """True if this file already holds a metrics row for (setting, method)."""
    path = os.path.join(ROOT, result_file)
    try:
        lines = open(path, errors="ignore").read().splitlines()
    except FileNotFoundError:
        return False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(setting_str + " (") and label in s:
            if i + 1 < len(lines) and "Coverage:" in lines[i + 1]:
                return True
    return False


def has_checkpoint(dataset, ne, variant, model_id="test"):
    return os.path.exists(os.path.join(ROOT, "checkpoints",
                                       setting(dataset, ne, variant, model_id), "checkpoint.pth"))


def build_jobs():
    """(priority, tag, dataset, ne, variant, kind) -- lower priority runs first."""
    ett = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    small = ["exchange-rate", "national-illness"]
    mid = ["weather"]
    wide = ["electricity", "traffic"]
    jobs = []

    def add(prio, ds, ne, variant, kind):
        jobs.append(dict(prio=prio, dataset=ds, ne=ne, variant=variant, kind=kind,
                         tag=f"{kind}_{ds}_ne{ne}_{variant}"))

    for ds in ett + small + mid + wide:            # calibration on what already exists
        for ne in range(1, 6):
            for variant in ("MOG", "MOGU", "MOE"):
                if variant == "MOGU" and ne == 1:
                    continue                        # no gate to make uncertain with one expert
                if not has_checkpoint(ds, ne, variant):
                    continue
                prio = {True: 10}.get(ds in ett, 20 if ds in small else 30 if ds in mid else 40)
                add(prio, ds, ne, variant, "calib")

    for ds in small + mid:                          # train what is missing, cheap datasets first
        for ne in range(1, 6):
            for variant in ("MOG", "MOGU", "MOE"):
                if variant == "MOGU" and ne == 1:
                    continue
                if not has_checkpoint(ds, ne, variant):
                    add(50 if ds in small else 60, ds, ne, variant, "train")

    for ds in ett + small + mid + wide:             # CQR: its own pinball-loss model
        for ne in (1, 3, 5):
            prio = 70 if ds in ett + small else 80 if ds in mid else 90
            add(prio, ds, ne, "MOE", "cqr")

    for ds in wide:                                 # widest training last: hours per cell
        for ne in range(1, 6):
            for variant in ("MOG", "MOGU", "MOE"):
                if variant == "MOGU" and ne == 1:
                    continue
                if not has_checkpoint(ds, ne, variant):
                    add(95, ds, ne, variant, "train")

    jobs.sort(key=lambda j: (j["prio"], j["dataset"], j["ne"], j["variant"]))
    return jobs


def pending_methods(job):
    """Flags this cell still needs. Empty means nothing to do -- skip the model load."""
    ds, ne, variant, kind = job["dataset"], job["ne"], job["variant"], job["kind"]
    if kind == "cqr":
        flag, rf, label, _ = CQR_METHOD
        s = setting(ds, ne, "MOE", model_id="cqr")
        return [] if has_result(rf, label, s) else [flag]
    s = setting(ds, ne, variant)
    prob = variant in ("MOG", "MOGU")
    out = []
    for flag, rf, label, needs_prob in METHODS:
        if needs_prob and not prob:
            continue
        if not has_result(rf, label, s):
            out.append(flag)
    return out


def build_cmd(job, flags):
    ds, ne, variant, kind = job["dataset"], job["ne"], job["variant"], job["kind"]
    cfg = DATA[ds]
    wide = cfg["ch"] >= 300
    args = [
        PY, "-u", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1" if kind in ("train", "cqr") else "0",
        "--root_path", f"./data/long_term_forecast/{cfg['root']}",
        "--data_path", cfg["file"],
        "--model_id", "cqr" if kind == "cqr" else "test",
        "--model", "iTransformer",
        "--data", cfg["data"],
        "--features", "M",
        "--seq_len", "96", "--label_len", "48", "--pred_len", str(cfg["pl"]),
        "--batch_size", "4" if wide else "8",
        "--enc_in", str(cfg["ch"]), "--dec_in", str(cfg["ch"]), "--c_out", str(cfg["ch"]),
        "--seed", str(SEED),
        "--num_experts", str(ne),
        # MoECP recomputes an H x C weighted-quantile grid per origin and defaults to a
        # single worker (run.py:226); serial electricity produced nothing in 111 minutes.
        "--moecp_workers", "16" if wide else "8",
    ]
    if kind == "cqr":
        args.append("--use_quantile_loss")
    else:
        if variant in ("MOG", "MOGU"):
            args.append("--prob_expert")
        if variant == "MOGU":
            args.append("--unc_gating")
        # ILI + probabilistic experts diverges without gradient clipping.
        if ds == "national-illness" and variant in ("MOG", "MOGU") and kind == "train":
            args += ["--max_grad_norm", "1"]
    return args + flags


def claim(tag):
    """Atomic cross-worker claim. Returns False if another worker already took this job."""
    try:
        fd = os.open(os.path.join(CLAIM_DIR, tag), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()} {time.strftime('%F %T')}\n".encode())
    os.close(fd)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--deadline-hours", type=float, default=24.0)
    a = ap.parse_args()

    os.makedirs(CLAIM_DIR, exist_ok=True)
    deadline = time.time() + a.deadline_hours * 3600
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = a.gpu
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    new = not os.path.exists(MANIFEST)
    mf = open(MANIFEST, "a", newline="")
    w = csv.writer(mf)
    if new:
        w.writerow(["worker", "tag", "kind", "dataset", "ne", "variant",
                    "n_methods", "status", "seconds", "log"])
        mf.flush()

    jobs = build_jobs()
    print(f"[w{a.worker_id}] gpu {a.gpu}: {len(jobs)} candidate jobs, "
          f"deadline in {a.deadline_hours:.1f}h", flush=True)

    done = 0
    for job in jobs:
        if time.time() > deadline:
            print(f"[w{a.worker_id}] deadline reached, stopping", flush=True)
            break
        if not claim(job["tag"]):
            continue
        flags = pending_methods(job)          # rescan now: the other worker may have filled it
        if not flags:
            w.writerow([a.worker_id, job["tag"], job["kind"], job["dataset"], job["ne"],
                        job["variant"], 0, "skip-complete", 0, ""])
            mf.flush()
            continue
        log = os.path.join(RUN_DIR, job["tag"] + ".log")
        cmd = build_cmd(job, flags)
        print(f"[{time.strftime('%T')}] w{a.worker_id} {job['tag']} "
              f"({len(flags)} methods)", flush=True)
        t0 = time.time()
        with open(log, "w") as lf:
            lf.write("CMD: " + " ".join(cmd) + "\n\n")
            lf.flush()
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        status = "ok" if rc == 0 else f"fail(rc={rc})"
        w.writerow([a.worker_id, job["tag"], job["kind"], job["dataset"], job["ne"],
                    job["variant"], len(flags), status, f"{dt:.0f}", log])
        mf.flush()
        print(f"    w{a.worker_id} {job['tag']} {status} {dt:.0f}s", flush=True)
        done += 1

    mf.close()
    print(f"[w{a.worker_id}] finished, {done} jobs run", flush=True)


if __name__ == "__main__":
    main()
