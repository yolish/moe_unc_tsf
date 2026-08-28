"""Fill seeds 4022-4025 on electricity and traffic for every calibrator in the report.

Scope is exactly the cells docs/results_tables_multiseed.tex reports at H=96: the three
`model_id=test` backbones (MoGE = MOG ne3, Single Gaussian = MOG ne1, Single Expert =
MOE ne1) plus the separate pinball-loss trunk (`model_id=cqr`, ne1 only) that the two
CQR rows read off. Seed 4021 already exists everywhere; only 4022-4025 are built here.

Do NOT add an ne3 CQR cell here: build_results_tex.py's "Single Expert" block is
hardcoded to read the CQR columns from the ne1 CSV for every dataset (BLOCKS_ACI /
BLOCKS_ACI_MS), and the ETT expert sweep -- the only table that varies CQR across expert
counts -- is hardcoded to ETT only. An ne3 CQR run for electricity/traffic feeds no table
in this report. This cost ~50 GPU-hours on traffic before being caught and killed.

What actually has to run differs sharply between the two datasets:

  electricity  the three `test` checkpoints already exist at all four seeds (they were
               trained for the ACI sweep), so those cells are calibration-only. Just the
               two cqr trunks need training.
  traffic      nothing beyond seed 4021 exists, so every cell trains first. At 862
               channels this is the expensive half of the job by a wide margin.

Resumable at method granularity: before each job the result_calibration_*.txt files are
rescanned and any calibrator that already has a row for that exact setting string is
dropped, so a killed run resumes where it stopped rather than redoing a whole cell. Work
is claimed through atomic O_EXCL files, so two workers share one job list and neither can
take the other's cell.

    python scripts/run_elec_traffic_seeds_gap.py --dry-run
    python scripts/run_elec_traffic_seeds_gap.py --gpu 0 --worker-id 0
    python scripts/run_elec_traffic_seeds_gap.py --gpu 1 --worker-id 1

Afterwards regenerate the report:
    python scripts/collect_calibration_results.py
    python scripts/build_headline_table_multiseed.py --num-experts 3 --variant MOG
    python scripts/build_headline_table_multiseed.py --num-experts 1 --variant MOG
    python scripts/build_headline_table_multiseed.py --num-experts 1 --variant MOE
    python scripts/build_results_tex.py --multiseed --out docs/results_tables_multiseed.tex
"""
import argparse
import csv
import os
import subprocess
import time

ROOT = "/home/dsi/giladaviv/moe_unc_tsf"
PY = os.path.join(ROOT, "unc_moe", "bin", "python")
SEEDS = [4022, 4023, 4024, 4025]
RUN_DIR = os.path.join(ROOT, "logs", "elec_traffic_seeds")
CLAIM_DIR = os.path.join(RUN_DIR, "claims")
MANIFEST = os.path.join(RUN_DIR, "manifest.csv")

DATA = {
    "electricity": dict(root="electricity/", file="electricity.csv", ch=321),
    "traffic":     dict(root="traffic/",     file="traffic.csv",     ch=862),
}
PE_UG = {"MOE": ("0", "0"), "MOG": ("1", "0")}

# (method key, flag, result file, label as written into that file, needs --prob_expert).
# Labels come from exp/exp_long_term_forecasting.py's writers; the file scoping is what
# keeps "Aleatoric Scale CP" from matching "ACI Aleatoric Scale CP" and so on -- the two
# always live in different files, never the same one.
METHODS_TEST = [
    ("standard_cp",              "--do_cp_calibration",
     "result_calibration_mse_cp.txt",                       "Standard CP with Sliding Window", False),
    ("aci_cp",                   "--do_aci_cp_calibration",
     "result_calibration_aci_cp_tsf.txt",                   "ACI CP Sliding Window",           False),
    ("cpvs",                     "--do_cpvs_calibration",
     "result_calibration_cpvs.txt",                         "Adaptive CPVS Delayed",           True),
    ("aci_cpvs",                 "--do_aci_cpvs_calibration",
     "result_calibration_aci_cpvs_tsf.txt",                 "ACI CP-VS Sliding Window",        True),
    ("aleatoric_only",           "--do_aleatoric_only_calibration",
     "result_calibration_aleatoric_only.txt",               "Aleatoric Only CP",               True),
    ("aci_aleatoric_only",       "--do_aci_aleatoric_only_calibration",
     "result_calibration_aci_aleatoric_only_tsf.txt",       "ACI Aleatoric Only CP",           True),
    ("cp_aleatoric_scale",       "--do_aleatoric_scale_calibration",
     "result_calibration_aleatoric_scale_tsf.txt",          "Aleatoric Scale CP",              True),
    ("aci_aleatoric_scale_g001", "--do_aci_aleatoric_scale_g001_calibration",
     "result_calibration_aci_aleatoric_scale_g001_tsf.txt", "ACI Aleatoric Scale CP (g=0.001)", True),
    ("moecp",                    "--do_moecp_calibration",
     "result_calibration_moecp_tsf.txt",                    "MoECP delayed",                   True),
]

# The CQR trunk: model_id=cqr, --use_quantile_loss, no --prob_expert (run.py refuses the
# combination), so these never appear on a `test` cell.
METHODS_CQR = [
    ("cqr_quantile",     "--do_cqr_calibration",
     "result_calibration_cqr_quantile.txt",         "CQR Quantile"),
    ("aci_cqr_quantile", "--do_aci_cqr_calibration",
     "result_calibration_aci_cqr_quantile_tsf.txt", "ACI CQR Quantile"),
    ("cqr_retrain",      "--do_cqr_retrain_calibration",
     "result_calibration_cqr_retrain.txt",          "Retrained CQR"),
    ("aci_cqr_retrain",  "--do_aci_cqr_retrain_calibration",
     "result_calibration_aci_cqr_retrain_tsf.txt",  "ACI Retrained CQR"),
]

# ACI-MoECP runs serially over (horizon step, channel) and has no worker support, which is
# impractical at 321/862 channels -- the same permanent gap the report's ACI footnote
# already states. Deliberately absent from METHODS_TEST rather than filtered later, so the
# reason lives next to the omission.
#
# aci_aleatoric_scale at gamma=0.01 is likewise absent: the report shows gamma=0.001 only.

# --methods aci restricts every cell to the ACI-adapted calibrators, i.e. the rows of the
# report's two ACI tables. This is much the cheaper half of the job: it drops MoECP (the
# single most expensive calibrator at 862 channels) and the four fixed-alpha base rows.
#
# aci_aleatoric_only (the "CPVS-aleatoric + ACI" row) is excluded at the user's request --
# not a capability gap, so it stays in METHODS_TEST and comes back by adding the key here.
ACI_ONLY = {"aci_cp", "aci_cpvs", "aci_aleatoric_scale_g001",
            "aci_cqr_quantile", "aci_cqr_retrain"}

# (dataset, model_id, num_experts, variant) for every cell the report reads.
CELLS = [
    ("test", 3, "MOG"),   # MoGE
    ("test", 1, "MOG"),   # Single Gaussian
    ("test", 1, "MOE"),   # Single Expert
    ("cqr",  1, "MOE"),   # pinball trunk behind the CQR rows -- ne1 only, see module note
]


def setting(dataset, model_id, ne, variant, seed):
    pe, ug = PE_UG[variant]
    return (f"long_term_forecast_{model_id}_iTransformer_{dataset}_ne{ne}_pe{pe}_ug{ug}_"
            f"ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_"
            f"fc1_ebtimeF_dtTrue_test_0_seed{seed}")


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
            # A header with no metrics line under it is a run that died mid-method.
            if i + 1 < len(lines) and "Coverage:" in lines[i + 1]:
                return True
    return False


def has_checkpoint(setting_str):
    return os.path.exists(os.path.join(ROOT, "checkpoints", setting_str, "checkpoint.pth"))


def pending(job, only=None):
    """Calibrators this cell still needs. Empty means skip it -- no model load at all."""
    methods = METHODS_CQR if job["model_id"] == "cqr" else METHODS_TEST
    prob = job["variant"] == "MOG"
    out = []
    for m in methods:
        if job["model_id"] == "cqr":
            key, flag, rf, label = m
        else:
            key, flag, rf, label, needs_prob = m
            if needs_prob and not prob:
                continue
        if only is not None and key not in only:
            continue
        if not has_result(rf, label, job["setting"]):
            out.append((key, flag))
    return out


def build_jobs():
    """Cheapest-and-already-trained first; traffic training last."""
    jobs = []
    for dataset in ("electricity", "traffic"):
        for model_id, ne, variant in CELLS:
            for seed in SEEDS:
                s = setting(dataset, model_id, ne, variant, seed)
                trained = has_checkpoint(s)
                # electricity-calibration (0) < electricity-train (1)
                #   < traffic-calibration (2) < traffic-train (3)
                prio = (0 if dataset == "electricity" else 2) + (0 if trained else 1)
                jobs.append(dict(dataset=dataset, model_id=model_id, ne=ne,
                                 variant=variant, seed=seed, setting=s,
                                 needs_train=not trained, prio=prio,
                                 tag=f"{dataset}_{model_id}_ne{ne}_{variant}_seed{seed}"))
    jobs.sort(key=lambda j: (j["prio"], j["dataset"], j["model_id"], j["ne"], j["seed"]))
    return jobs


def build_cmd(job, flags):
    cfg = DATA[job["dataset"]]
    args = [
        PY, "-u", "run.py",
        "--task_name", "long_term_forecast",
        # Training and calibration are one invocation: run.py calibrates after fitting,
        # so an untrained cell does not need a separate pass.
        "--is_training", "1" if job["needs_train"] else "0",
        "--root_path", f"./data/long_term_forecast/{cfg['root']}",
        "--data_path", cfg["file"],
        "--model_id", job["model_id"],
        "--model", "iTransformer",
        "--data", "custom",
        "--features", "M",
        "--seq_len", "96", "--label_len", "48", "--pred_len", "96",
        # 321/862 channels do not fit the batch size the small datasets use.
        "--batch_size", "4",
        "--enc_in", str(cfg["ch"]), "--dec_in", str(cfg["ch"]), "--c_out", str(cfg["ch"]),
        "--seed", str(job["seed"]),
        "--num_experts", str(job["ne"]),
        # MoECP rebuilds an H x C weighted-quantile grid per origin; serial electricity
        # produced nothing in 111 minutes during the earlier campaign.
        "--moecp_workers", "16",
        "--moecp_temperature", "1.0",
        # Every ACI row in the report is gamma=0.001 (the 0.1/H rule of thumb at H=96).
        "--aci_gamma", "0.001", "--aci_alpha", "0.1",
    ]
    if job["model_id"] == "cqr":
        args.append("--use_quantile_loss")
    elif job["variant"] == "MOG":
        args.append("--prob_expert")
    return args + flags


def claim(tag):
    """Atomic cross-worker claim; False means another worker already took this job."""
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
    ap.add_argument("--deadline-hours", type=float, default=1e9,
                    help="stop claiming new jobs after this long; a running job finishes")
    ap.add_argument("--datasets", default="electricity,traffic",
                    help="comma-separated subset, e.g. --datasets electricity")
    ap.add_argument("--methods", default="aci", choices=["aci", "all"],
                    help="'aci' (default) runs only the ACI-adapted calibrators; "
                         "'all' adds the fixed-alpha base rows, including MoECP")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    only = ACI_ONLY if a.methods == "aci" else None
    keep = {d.strip() for d in a.datasets.split(",") if d.strip()}
    jobs = [j for j in build_jobs() if j["dataset"] in keep]

    if a.dry_run:
        n_train = n_calib = n_results = 0
        for j in jobs:
            need = pending(j, only)
            if not need and not j["needs_train"]:
                continue
            n_results += len(need)
            if j["needs_train"]:
                n_train += 1
            else:
                n_calib += 1
            print(f"  [{'TRAIN' if j['needs_train'] else 'calib'}] {j['tag']:<48} "
                  f"{len(need)} methods: {','.join(k for k, _ in need)}")
        print(f"\n{n_train} cells need training, {n_calib} are calibration-only; "
              f"{n_results} calibration results missing in total")
        return

    os.makedirs(CLAIM_DIR, exist_ok=True)
    deadline = time.time() + a.deadline_hours * 3600
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = a.gpu
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    new = not os.path.exists(MANIFEST)
    mf = open(MANIFEST, "a", newline="")
    w = csv.writer(mf)
    if new:
        w.writerow(["worker", "tag", "kind", "dataset", "ne", "variant", "seed",
                    "methods", "status", "seconds", "log"])
        mf.flush()

    print(f"[w{a.worker_id}] gpu {a.gpu}: {len(jobs)} candidate cells", flush=True)
    for job in jobs:
        if time.time() > deadline:
            print(f"[w{a.worker_id}] deadline reached, stopping", flush=True)
            break
        # Rescan right before claiming: the other worker may have finished this cell.
        need = pending(job, only)
        if not need:
            continue
        if not claim(job["tag"]):
            continue

        kind = "train" if job["needs_train"] else "calib"
        log = os.path.join(RUN_DIR, f"{job['tag']}.log")
        cmd = build_cmd(job, [f for _, f in need])
        print(f"[{time.strftime('%F %T')}] w{a.worker_id} {kind} {job['tag']} "
              f"[{','.join(k for k, _ in need)}]", flush=True)
        t0 = time.time()
        with open(log, "w") as lf:
            lf.write("CMD: " + " ".join(cmd) + "\n\n")
            lf.flush()
            rc = subprocess.run(cmd, cwd=ROOT, env=env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = int(time.time() - t0)
        if rc != 0:
            # Release the claim so the other worker -- or a later restart -- can retry.
            # A GPU-local failure (an OOM on the busier card) usually succeeds elsewhere,
            # and pending() has already recorded whichever methods did finish, so a retry
            # resumes rather than repeating them. Worst case the cell is attempted once
            # more per worker, since each walks the job list only once.
            try:
                os.remove(os.path.join(CLAIM_DIR, job["tag"]))
            except OSError:
                pass
        print(f"[{time.strftime('%F %T')}] w{a.worker_id} {job['tag']} rc={rc} ({dt}s)",
              flush=True)
        w.writerow([a.worker_id, job["tag"], kind, job["dataset"], job["ne"],
                    job["variant"], job["seed"], "+".join(k for k, _ in need),
                    "ok" if rc == 0 else f"rc{rc}", dt, log])
        mf.flush()

    print(f"[w{a.worker_id}] no work left", flush=True)


if __name__ == "__main__":
    main()
