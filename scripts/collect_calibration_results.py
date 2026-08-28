"""Collect every long_term_forecast calibration result and render docs/calibration_results_tsf.{md,csv}.

Sources: the result_calibration_*.txt files written by the calibration code, plus logs/**/*.log
for the methods that never got a result file. Re-run after new experiments:

    python scripts/collect_calibration_results.py
"""
import re, glob, os, collections, csv, datetime, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETTING_RE = re.compile(
    r'^long_term_forecast_(?P<model_id>.+?)_(?P<model>[A-Za-z0-9]+)_(?P<dataset>.+?)'
    r'_ne(?P<ne>\d+)_pe(?P<pe>\d+)_ug(?P<ug>\d+)_ft(?P<ft>[A-Z]+)_sl(?P<sl>\d+)_ll(?P<ll>\d+)'
    r'_pl(?P<pl>\d+)_dm(?P<dm>\d+)_nh(?P<nh>\d+)_el(?P<el>\d+)_dl(?P<dl>\d+)_df(?P<df>\d+)'
    r'_expand(?P<expand>\d+)_dc(?P<dc>\d+)_fc(?P<fc>\d+)_eb(?P<eb>\w+?)_dt(?P<dt>True|False)'
    r'_(?P<des>.+)_(?P<itr>\d+)_seed(?P<seed>\d+)$')


def parse_setting(s):
    m = SETTING_RE.match(s)
    return m.groupdict() if m else None


# ---------------------------------------------------------------- method keys
def norm_method(label):
    l = label.lower()
    # ACI variants must be classified before every base branch below. Each ACI label also
    # contains its base method's keyword ("ACI MoECP" contains "moecp", "ACI Retrained CQR"
    # contains "retrained cqr", ...), so a base branch reached first would silently merge
    # ACI runs into the base method's published row -- wrong numbers, no error.
    # Order inside this block matters too: "aleatoric only" before "aleatoric scale", and
    # the CP-VS spellings before the bare "cp" fallback, or the ACI variants collide with
    # each other instead.
    if 'aci' in l.split():
        if 'moecp' in l:
            return 'aci_moecp'
        if 'retrained cqr' in l or 'cqr retrain' in l:
            return 'aci_cqr_retrain'
        if 'cqr' in l:
            return 'aci_cqr_quantile'
        if 'aleatoric only' in l:
            return 'aci_aleatoric_only'
        if 'aleatoric scale' in l:
            return 'aci_aleatoric_scale_g001' if 'g=0.001' in l else 'aci_aleatoric_scale'
        if 'cp-vs' in l or 'cpvs' in l or 'cp_vs' in l:
            return 'aci_cpvs'
        if 'cp' in l:
            return 'aci_cp'
    if 'standard cp' in l:
        return 'standard_cp'
    if 'cpvs' in l or 'cp_vs' in l:
        if 'aleatoric' in l:
            return 'cpvs_aleatoric'
        if 'static' in l:
            return 'cpvs_static'
        return 'cpvs'
    if 'aleatoric mog' in l or 'aleatoric_mog' in l:
        return 'aleatoric_mog_v2' if 'second option' in l else 'aleatoric_mog'
    if 'aleatoric only' in l:
        return 'aleatoric_only'
    if 'retrained cqr' in l:
        return 'cqr_retrain'
    if 'cqr quantile' in l:
        return 'cqr_quantile'
    if l.startswith('cqr'):
        return 'cqr'
    if 'cp-dvs' in l or 'cp_dvs' in l:
        return 'cp_dvs'
    if 'moecp' in l:
        return 'moecp'
    if 'moce' in l:
        return 'moce'
    if 'adaptive scalar' in l:
        return 'adaptive_scalar'
    # (The ACI branch that used to sit here now runs first, above -- it has to precede the
    # base-method branches, not just the plain aleatoric-scale one.)
    if 'cp aleatoric scale' in l or 'aleatoric scale' in l:
        return 'cp_aleatoric_scale'
    if 'adaptive variance' in l:
        return 'adaptive_variance'
    if 'hpd' in l:
        return l.split()[0].lower().replace('-hpd', '') + '_hpd'
    # Catch-all: never drop a result just because its label is new. Slugify it instead --
    # it lands in the report flagged as undescribed, which is how a newly added calibrator
    # announces itself. (Dropping silently is how result_calibration_aleatoric_scale_tsf.txt
    # went unnoticed for a day.)
    slug = re.sub(r'[^a-z0-9]+', '_', re.split(r'\balpha\b|,|\(', l)[0]).strip('_')
    slug = re.sub(r'_(delayed|update|results?)$', '', slug)
    return slug or None


NUM = r'-?\d+\.?\d*(?:e-?\d+)?|nan'

# ------------------------------------------------------------------- txt files
records = {}   # (setting, method) -> dict
extra_meta = collections.defaultdict(dict)


# MoECP is kept at tau=1 only. Its coverage counts an unbounded interval as a hit while
# width is a finite-only mean, so the two metrics are computed on different populations;
# higher tau localizes harder and drives more origins into that unbounded/abstained state
# (up to 16% at tau=50 on national-illness), which is not comparable to the other
# calibrators. The width itself barely moves across tau (~1.6% over the tau=1..50 sweep on
# national-illness) so tau=1 costs nothing real while removing the inflation.
#
# Non-tau=1 entries were removed from result_calibration_moecp_tsf.txt (backed up in
# result_calibration_moecp_tau100.removed.txt and result_calibration_moecp_nontau1.removed.txt),
# but logs/ still contains them scattered inside logs that also carry unrelated results
# that must not be deleted -- hence the filter here rather than at the source. Drop this
# check to bring other tau values back.
def _is_dropped(method, label):
    return method in ('moecp', 'aci_moecp') and not re.search(r'tau=1\.0\b', label or '')


# 2026-08-13: MoECPCalibrator.localized_quantile switched from returning +inf when the
# target level falls in the test point's delta_inf mass to clipping at the window's own max
# residual (calibration/moecp_calibration.py), so width and coverage are computed over the
# same population instead of coverage getting a free pass on unbounded cells. That changes
# the *numbers*, not just their reporting, so pre-fix `moecp` observations must not blend
# with post-fix ones under the same (setting, method) key. result_calibration_moecp_tsf.txt
# was archived (result_calibration_moecp_preclip.removed_20260813.txt) and truncated, so
# only log-sourced entries need this filter. aci_moecp is untouched by the fix and must not
# be dropped here -- ACIMoECPCalibrator still relies on the +inf branch as its ACI restoring
# force (see that calibrator's docstring), so it keeps the old behavior on purpose.
MOECP_CLIP_FIX_MTIME = datetime.datetime(2026, 8, 13, 9, 18, 42).timestamp()


def _is_stale_preclip_moecp(method, source_path):
    if method != 'moecp' or not source_path:
        return False
    try:
        return os.stat(source_path).st_mtime < MOECP_CLIP_FIX_MTIME
    except OSError:
        return False


def add(setting, method, metrics, source, label, source_path=None):
    """Collect every observation; the canonical value is picked later (txt beats log)."""
    cfg = parse_setting(setting)
    if cfg is None:
        return
    if _is_dropped(method, label):
        return
    if _is_stale_preclip_moecp(method, source_path):
        return
    key = (setting, method)
    if key not in records:
        rec = dict(cfg)
        rec.update(setting=setting, method=method, obs=[])
        records[key] = rec
    obs = dict(metrics)
    obs.update(source=source, label=label, is_txt=source.endswith('.txt'))
    records[key]['obs'].append(obs)


def kvpairs(text):
    out = {}
    for k, v in re.findall(r'([A-Za-z_][A-Za-z0-9_^ ]*?)\s*[:=]\s*(' + NUM + r'|True|False|[\d.]+s)', text):
        out[k.strip().lower().replace(' ', '_')] = v
    return out


for path in sorted(glob.glob(os.path.join(ROOT, 'result_calibration_*.txt'))):
    fname = os.path.basename(path)
    lines = open(path, errors='ignore').read().splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line.startswith('long_term_forecast'):
            continue
        m = re.match(r'^(\S+)\s+\((.*)\)$', line)
        if not m:
            continue
        setting, label = m.group(1), m.group(2)
        if 'Diagnostics' in label:
            continue
        method = norm_method(label)
        if method is None:
            continue
        body = lines[i + 1].strip() if i + 1 < len(lines) else ''
        vals = kvpairs(body)
        metrics = {}
        for src, dst in [('coverage', 'coverage'), ('width', 'width'),
                         ('median_width', 'median_width'), ('q_mean', 'q_mean'),
                         ('unbounded', 'unbounded'), ('retrains', 'retrains')]:
            if src in vals:
                metrics[dst] = vals[src]
        metrics['method_params'] = label
        add(setting, method, metrics, fname, label, source_path=path)

# ---------------------------------------------------------------------- logs
LOG_FILES = sorted(glob.glob(os.path.join(ROOT, 'logs', '*.log')) +
                   glob.glob(os.path.join(ROOT, 'logs', '*', '*.log')))

SET_RE = re.compile(r'(long_term_forecast_\S*?)(?:<<<|\.\.\.|\s|$)')
RES_RE = re.compile(r'^([A-Za-z][A-Za-z0-9_() .,\'^*=+-]*?) Results(?: \(([^)]*)\))?:\s*$')

for path in LOG_FILES:
    cur = None
    second_option = False
    lines = open(path, errors='ignore').read().splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        hit = SET_RE.search(s)
        if hit and ('calibrating' in s or 'testing' in s or ' for ' in s or 'training' in s):
            cand = hit.group(1).rstrip('.')
            if parse_setting(cand):
                cur = cand
        if s.startswith('Running ') or s.startswith('>>>>>>> Start '):
            second_option = 'second option' in s
        m = RES_RE.match(s)
        if not m or cur is None:
            continue
        label = m.group(1) + (' (' + m.group(2) + ')' if m.group(2) else '')
        method = norm_method(label)
        if method == 'aleatoric_mog' and second_option:
            method, label = 'aleatoric_mog_v2', label + ' [second option]'
        if method is None:
            continue
        block = '\n'.join(lines[i + 1:i + 8])
        block = block.split('\n\n')[0]
        metrics = {}
        for key, pat in [('coverage', r'Coverage:\s*(' + NUM + ')'),
                         ('width', r'Avg Width:\s*(' + NUM + ')'),
                         ('median_width', r'Median Width:\s*(' + NUM + ')'),
                         ('q_mean', r'Mean q[^:]*:\s*(' + NUM + ')')]:
            mm = re.search(pat, block)
            if mm:
                metrics[key] = mm.group(1)
        if 'coverage' not in metrics:
            continue
        metrics['method_params'] = label
        add(cur, method, metrics, 'logs/' + os.path.relpath(path, os.path.join(ROOT, 'logs')), label,
            source_path=path)

# --------------------------------------------------------- base forecast MSE
base = {}
lines = open(os.path.join(ROOT, 'result_long_term_forecast.txt'), errors='ignore').read().splitlines()
for i, line in enumerate(lines):
    s = line.strip()
    if not s.startswith('long_term_forecast'):
        continue
    if not parse_setting(s):
        continue
    if i + 1 < len(lines):
        mm = re.search(r'mse:(' + NUM + r'), mae:(' + NUM + ')', lines[i + 1])
        if mm:
            base[s] = (float(mm.group(1)), float(mm.group(2)))

# also harvest mse/mae from logs (testing block)
for path in LOG_FILES:
    cur = None
    lines = open(path, errors='ignore').read().splitlines()
    for line in lines:
        s = line.strip()
        hit = SET_RE.search(s)
        if hit and ('testing' in s or 'calibrating' in s):
            cand = hit.group(1).rstrip('.')
            if parse_setting(cand):
                cur = cand
        mm = re.match(r'^mse:(' + NUM + r'), mae:(' + NUM + ')', s)
        if mm and cur and cur not in base:
            base[cur] = (float(mm.group(1)), float(mm.group(2)))

# ---------------------------------------- pick canonical value per (setting, method)
out = []
for rec in records.values():
    obs = rec.pop('obs')
    txt_obs = [o for o in obs if o['is_txt']]
    canon = txt_obs[-1] if txt_obs else obs[-1]
    r = dict(rec)
    r.update({k: v for k, v in canon.items() if k != 'is_txt'})
    covs = [float(o['coverage']) for o in obs if o.get('coverage')]
    wids = [float(o['width']) for o in obs if o.get('width')]
    r['n_runs'] = len(obs)
    r['coverage_min'] = f'{min(covs):.4f}' if covs else None
    r['coverage_max'] = f'{max(covs):.4f}' if covs else None
    r['width_min'] = f'{min(wids):.4f}' if wids else None
    r['width_max'] = f'{max(wids):.4f}' if wids else None
    r['rerun_disagrees'] = bool(covs) and (max(covs) - min(covs) > 5e-4)
    r['all_sources'] = ';'.join(sorted({o['source'] for o in obs}))
    out.append(r)

for r in out:
    if r['setting'] in base:
        r['mse'], r['mae'] = base[r['setting']]


# Calibrators the report covers. Everything else found in the result files / logs is
# dropped here rather than filtered downstream, so the .md, the .csv and the delta all
# agree on one method set. Nothing is deleted from result_calibration_*.txt or logs/, so
# removing a slug from this set is reversible: add it back and rerun.
REPORTED_METHODS = {
    'standard_cp', 'cpvs', 'aleatoric_only', 'cp_aleatoric_scale',
    'aci_aleatoric_scale', 'aci_aleatoric_scale_g001',
    'cqr_quantile', 'cqr_retrain', 'moecp', 'adaptive_window_cp',
    'aci_cp', 'aci_cpvs', 'aci_aleatoric_only',
    'aci_cqr_quantile', 'aci_cqr_retrain', 'aci_moecp',
}

recs = [r for r in out if r['method'] in REPORTED_METHODS]
OUT_MD = os.path.join(ROOT, 'docs', 'calibration_results_tsf.md')
OUT_CSV = os.path.join(ROOT, 'docs', 'calibration_results_tsf.csv')
OUT_CHANGELOG = os.path.join(ROOT, 'docs', 'calibration_results_changelog.md')
STATE = os.path.join(ROOT, 'docs', '.calibration_results_state.json')

# ------------------------------------------------------- delta vs previous run
# Every run rescans all of logs/ and result_calibration_*.txt (a full pass is ~3s),
# so newly created log files are picked up automatically. What this block adds is
# a record of *what is new since the last run*, so a daily rebuild is readable as
# "here is what arrived today" rather than a 1800-line file that silently changed.


def _sig(r):
    return f"{r.get('coverage')}/{r.get('width')}"


now_files = {}
for p in LOG_FILES + sorted(glob.glob(os.path.join(ROOT, 'result_calibration_*.txt'))):
    try:
        st = os.stat(p)
        now_files[os.path.relpath(p, ROOT)] = [st.st_size, int(st.st_mtime)]
    except OSError:
        pass
now_results = {r['setting'] + '|' + r['method']: _sig(r) for r in recs}

prev = {}
if os.path.exists(STATE):
    try:
        prev = json.load(open(STATE))
    except (ValueError, OSError):
        prev = {}
prev_files = prev.get('files', {})
prev_results = prev.get('results', {})
prev_run = prev.get('generated')

first_run = not prev
new_files = sorted(k for k in now_files if k not in prev_files)
grown_files = sorted(k for k in now_files
                     if k in prev_files and now_files[k] != prev_files[k])
gone_files = sorted(k for k in prev_files if k not in now_files)
new_results = sorted(k for k in now_results if k not in prev_results)
changed_results = sorted(k for k in now_results
                         if k in prev_results and now_results[k] != prev_results[k])
gone_results = sorted(k for k in prev_results if k not in now_results)


def _facet(keys, field):
    """Which dataset / method / seed values appear among these result keys."""
    seen = collections.Counter()
    idx = {r['setting'] + '|' + r['method']: r for r in recs}
    for k in keys:
        r = idx.get(k)
        if r:
            seen[r[field]] += 1
    return seen


new_settings = sorted({k.split('|')[0] for k in new_results} - {k.split('|')[0] for k in prev_results})
new_methods = sorted({k.split('|')[1] for k in now_results} - {k.split('|')[1] for k in prev_results})
delta = {
    'first_run': first_run,
    'prev_run': prev_run,
    'new_files': new_files, 'grown_files': grown_files, 'gone_files': gone_files,
    'new_results': new_results, 'changed_results': changed_results,
    'gone_results': gone_results, 'new_settings': new_settings,
    'new_methods': new_methods,
    'by_dataset': _facet(new_results, 'dataset'),
    'by_method': _facet(new_results, 'method'),
    'by_seed': _facet(new_results, 'seed'),
}

# Preferred display order, over the REPORTED_METHODS set. A calibrator that starts
# producing long_term_forecast results but is missing here is appended automatically (see
# the METHOD_ORDER extension below) -- but only if it is in REPORTED_METHODS, which is
# the whitelist.
# Each ACI variant is placed immediately after its base method so the report reads
# base/ACI adjacent.
METHOD_ORDER = ['standard_cp', 'aci_cp',
                'cpvs', 'aci_cpvs',
                'aleatoric_only', 'aci_aleatoric_only',
                'cp_aleatoric_scale',
                'aci_aleatoric_scale', 'aci_aleatoric_scale_g001',
                'cqr_quantile', 'aci_cqr_quantile',
                'cqr_retrain', 'aci_cqr_retrain',
                'moecp', 'aci_moecp',
                'adaptive_window_cp']
METHOD_LABEL = {
    'standard_cp': 'Standard CP',
    'cpvs': 'Adaptive CPVS',
    'aleatoric_mog': 'Aleatoric MoG',
    'aleatoric_mog_v2': 'Aleatoric MoG v2',
    'aleatoric_only': 'Aleatoric only',
    'cqr_quantile': 'CQR quantile',
    'cqr_retrain': 'CQR retrain',
    'cp_dvs': 'CP-DVS',
    'moecp': 'MoECP',
    'adaptive_scalar': 'Adaptive scalar',
    'adaptive_variance': 'Adaptive variance-ratio',
    'cp_aleatoric_scale': 'Aleatoric scale CP',
    'aci_aleatoric_scale': 'ACI aleatoric scale (g=0.01)',
    'aci_aleatoric_scale_g001': 'ACI aleatoric scale (g=0.001)',
    'adaptive_window_cp': 'Adaptive window CP',
    'aci_cp': 'ACI CP (g=0.001)',
    'aci_cpvs': 'ACI CP-VS (g=0.001)',
    'aci_aleatoric_only': 'ACI aleatoric only (g=0.001)',
    'aci_cqr_quantile': 'ACI CQR quantile (g=0.001)',
    'aci_cqr_retrain': 'ACI CQR retrain (g=0.001)',
    'aci_moecp': 'ACI MoECP (g=0.001)',
}
METHOD_DESC = {
    'aci_cp': ('`ACICPCalibrator`, standard_cp\'s absolute-residual score with the target '
               'level driven by the Gibbs & Candes (2021) recursion per (horizon, channel) '
               'cell. gamma=0.001 (the 0.1/pred_len rule of thumb at pred_len=96). '
               'gamma=0 reproduces standard_cp exactly. Works for any model.'),
    'aci_cpvs': ('`ACICPVSCalibrator`, cpvs\'s sigma-normalised residual score with the ACI '
                 'level recursion, width = q*sigma. gamma=0 reproduces cpvs exactly. '
                 'Needs `--prob_expert` (MOG/MOGU only).'),
    'aci_aleatoric_only': ('`ACIAleatoricOnlyCalibrator`, aleatoric_only\'s r^2/var_ale score '
                           'with the ACI level recursion. gamma=0 reproduces aleatoric_only '
                           'exactly. MOG/MOGU only.'),
    'aci_cqr_quantile': ('`ACICQRCalibrator`, cqr_quantile\'s signed conformity score with the '
                         'ACI level recursion. gamma=0 reproduces cqr_quantile exactly. '
                         'Needs `--use_quantile_loss`.'),
    'aci_cqr_retrain': ('`ACICQRCalibrator` behind the rolling-retrain driver: Gibbs & Candes '
                        'Sec 2.2 (refit on a trailing window) plus their Sec 2.1 (adapt the '
                        'level), where cqr_retrain runs only the former. gamma=0 reproduces '
                        'cqr_retrain exactly.'),
    'aci_moecp': ('`ACIMoECPCalibrator`, moecp\'s gate-localized weighted quantile read off at '
                  'a per-(horizon, channel) level 1-alpha_t. The one family member where the '
                  '[0,1] clip costs nothing: alpha_t -> 0 makes the weighted CDF fall short, '
                  'the interval goes unbounded, err_t = 0, and alpha_t is pulled back up. '
                  'tau=1 only, serial only. gamma=0 reproduces moecp bit-for-bit.'),
    'standard_cp': ('`AdaptiveCP`, sliding-window split CP on absolute residuals, '
                    'alpha=0.1, delayed update. Works for any model.'),
    'cpvs': ('`AdaptiveCPVS`, residual normalised by the predicted MoG variance, alpha=0.1, '
             'delayed update. Needs `--prob_expert` (MOG/MOGU only).'),
    'aleatoric_mog': ('`AleatoricMOGCalibrator`, alpha=0.1, epistemic term kept. '
                      'MOG/MOGU only.'),
    'aleatoric_mog_v2': ('`AleatoricMOGCalibratorSecondOption`, same but with the alternative '
                         'variance decomposition.'),
    'aleatoric_only': ('`AleatoricOnlyCalibrator`, alpha=0.1, epistemic term dropped. '
                       'MOG/MOGU only.'),
    'cqr_quantile': ('CQR on a model trained with pinball loss (`--use_quantile_loss`, `model_id=cqr`), '
                     'alpha=0.1, delayed update.'),
    'cqr_retrain': ('Retrained CQR, rolling window: mode=finetune, window=1000, retrain interval=96 '
                    'steps, 3 epochs per retrain, alpha=0.1.'),
    'cp_dvs': '`CPDVSCalibrator`, alpha=0.1, fit_frac=0.5, quadratic difficulty model.',
    'moecp': ('`MoECPCalibrator`, gate-localised CP, alpha=0.1, temperature=1.0. Lowered from the '
             "paper's tabular default of 100 -- at 100 the test point's own normalized weight "
             'exceeded alpha (unbounded interval) up to 25% of the time on this grid; tau=1 keeps '
             'that near 0 while width stays well under standard_cp\'s, so it is not a collapse to '
             'uniform weights. Validity (Theorem 1) holds at any tau, so this is a comparability '
             'choice, not a correctness fix.'),
    'cp_aleatoric_scale': ('`AleatoricScaleCalibrator`, alpha=0.1, delayed update. '
                           'Scales the interval by the aleatoric component only. MOG/MOGU only.'),
    'aci_aleatoric_scale': ('`ACIAleatoricScaleCalibrator`, alpha=0.1, gamma=0.01, delayed update. '
                            'Aleatoric scale score/interval with the target level driven by the ACI '
                            'recursion alpha_t += gamma*(alpha - err_t), per (horizon, channel) cell. '
                            'alpha_t is clipped to [0, 1] (and the quantile level capped at 1.0), so '
                            'width stays finite on every cell at the cost of some windup / '
                            'undercoverage relative to the exact (unbounded) recursion -- see the '
                            'calibrator docstring for the trade. gamma=0 recovers '
                            '`AleatoricScaleCalibrator`. MOG/MOGU only.'),
    'aci_aleatoric_scale_g001': ('Same `ACIAleatoricScaleCalibrator`, gamma fixed at 0.001 instead of '
                                 '0.01. Under the delayed-update protocol the effective step is '
                                 '~gamma*pred_len, so the smaller gamma keeps alpha_t off the clip more '
                                 'often at the cost of tracking drift more slowly. Run alongside the '
                                 'g=0.01 variant for direct comparison, not as its replacement. '
                                 'MOG/MOGU only.'),
    'adaptive_window_cp': ('`AdaptiveWindowAleatoricScaleCalibrator`, alpha=0.1, delayed update. '
                           'Same score and interval as `AleatoricScaleCalibrator`, but the sliding-'
                           'window size is chosen per run by replaying the delayed protocol on the '
                           'tail of validation and scoring each candidate W by the interval '
                           '(Winkler) score. MOG/MOGU only.'),
    'adaptive_variance': ('`AdaptiveVarianceCalibrator`, alpha=0.1, delayed update. '
                          'Learns a variance-ratio scaling on top of the MoG variance. MOG/MOGU only.'),
    'adaptive_scalar': ('Early `AdaptiveCPVS` printed under the label "Adaptive Scalar" before the '
                        'rename (commit 3f54bba, `exp.test()`); alpha=0.1. XL-dataset sweep only.'),
}
# Sliding-window length actually used, and where that value comes from. Only CQR retrain
# writes its window into the result line; for every other method the per-run output does
# not record it, so the value is read from the call site in exp/exp_long_term_forecasting.py.
METHOD_WINDOW = {
    'aci_cp': ('1000', 'exp_long_term_forecasting.py:calibrate_aci_cp'),
    'aci_cpvs': ('1000', 'exp_long_term_forecasting.py:calibrate_aci_cpvs'),
    'aci_aleatoric_only': ('1000', 'exp_long_term_forecasting.py:calibrate_aci_aleatoric_only'),
    'aci_cqr_quantile': ('1000', 'exp_long_term_forecasting.py:calibrate_aci_cqr'),
    'aci_cqr_retrain': ('1000', 'recorded per run in the result line'),
    'aci_moecp': ('1000', 'exp_long_term_forecasting.py:calibrate_aci_moecp'),
    'standard_cp': ('1000', 'exp_long_term_forecasting.py:663'),
    'cpvs': ('1000', 'exp_long_term_forecasting.py:452'),
    'aleatoric_mog': ('1000', 'exp_long_term_forecasting.py:894'),
    'aleatoric_mog_v2': ('1000', 'exp_long_term_forecasting.py:903'),
    'aleatoric_only': ('1000', 'exp_long_term_forecasting.py:912'),
    'cqr_quantile': ('1000', 'exp_long_term_forecasting.py:560'),
    'cqr_retrain': ('1000', 'recorded per run in the result line'),
    'moecp': ('1000', 'exp_long_term_forecasting.py:999'),
    'cp_dvs': ('n/a', 'split conformal on validation (fit_frac=0.5), no sliding window'),
    'adaptive_scalar': ('300', 'historic AdaptiveCPVS in commit 3f54bba'),
    'adaptive_variance': ('1000', 'exp_long_term_forecasting.py:928'),
    'cp_aleatoric_scale': ('1000', 'exp_long_term_forecasting.py:921'),
    'aci_aleatoric_scale': ('1000', 'exp_long_term_forecasting.py:930'),
    'aci_aleatoric_scale_g001': ('1000', 'exp_long_term_forecasting.py:1057'),
    'adaptive_window_cp': ('n/a', 'window size selected per run, exp_long_term_forecasting.py:1088'),
}


WINDOW_EPOCH = datetime.datetime(2026, 1, 22)   # commit bef8056 set 500 -> 1000


def _log_dates(r):
    """Timestamps of the log files this record was seen in (empty if txt-only)."""
    ds = []
    for src in (r.get('all_sources') or r['source']).split(';'):
        if src.startswith('logs/'):
            try:
                ds.append(datetime.datetime.fromtimestamp(
                    os.stat(os.path.join(ROOT, 'logs', src[len('logs/'):])).st_mtime))
            except OSError:
                pass
    return ds


def window_of(r):
    """Window for this run, or blank when it cannot be established.

    The window is only a fact about a run if the run recorded it, or if the run can be
    dated to a period when the code held a known value. It moved 300 -> 500 -> 1000
    (3f54bba, 0ca1060/3182ab8, bef8056), so stamping the current source value onto a
    row that cannot be dated would assert something unknown. Those are left blank.
    """
    m = re.search(r'window=(\d+)', r.get('method_params') or '')
    if m:
        return m.group(1), 'recorded in the run output'
    val = METHOD_WINDOW.get(r['method'], ('', ''))[0]
    if val == 'n/a':
        return 'n/a', 'calibrator has no sliding window'
    if r['method'] == 'adaptive_scalar':
        # The "Adaptive Scalar" print was deleted by 0ca1060, the same commit that moved
        # 300 -> 500, so a log carrying that label necessarily predates it.
        return '300', 'pinned by label lifetime (pre-0ca1060)'
    dates = _log_dates(r)
    if val and dates and min(dates) >= WINDOW_EPOCH:
        return val, 'code, run dated after 2026-01-22'
    return '', 'unknown - run cannot be dated'
# Which result_*.txt file each method lands in -- derived from the data, so a new
# result file added by the calibration code is reflected without editing this script.
TXT_FILES = {}
for _r in recs:
    for _src in (_r.get('all_sources') or _r['source']).split(';'):
        if _src.endswith('.txt'):
            TXT_FILES.setdefault(_r['method'], set()).add(_src)
TXT_FILES = {k: ', '.join('`' + x + '`' for x in sorted(v)) for k, v in TXT_FILES.items()}

# Any reported method present in the data but not in METHOD_ORDER gets appended.
for _m in sorted({r['method'] for r in recs}):
    if _m not in METHOD_ORDER:
        METHOD_ORDER.append(_m)
        METHOD_LABEL.setdefault(_m, _m.replace('_', ' ').title())
        METHOD_DESC.setdefault(_m, 'NEW — not yet described. Add an entry to METHOD_DESC / '
                                   'METHOD_WINDOW in scripts/collect_calibration_results.py.')

DATASET_ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange-rate', 'national-illness',
                 'traffic', 'electricity', 'weather', 'SynthMoE3']


def variant(r):
    pe, ug = r['pe'], r['ug']
    if pe == '0' and ug == '0':
        return 'MOE'
    if pe == '1' and ug == '0':
        return 'MOG'
    if pe == '1' and ug == '1':
        return 'MOGU'
    return f'pe{pe}/ug{ug}'


def f(v, nd=4):
    if v is None or v == '':
        return None
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return str(v)


# ------------------------------------------------------------------ csv dump
cols = ['dataset', 'model', 'model_id', 'num_experts', 'variant', 'prob_expert', 'unc_gating',
        'features', 'seq_len', 'label_len', 'pred_len', 'seed', 'method', 'coverage', 'width',
        'median_width', 'q_mean', 'alpha', 'window_size', 'window_source', 'retrains',
        'unbounded', 'mse', 'mae', 'n_runs_seen',
        'coverage_min', 'coverage_max', 'width_min', 'width_max', 'rerun_disagrees',
        'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff', 'expand', 'd_conv', 'factor',
        'embed', 'distil', 'des', 'itr', 'method_params', 'source', 'all_sources', 'setting']
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in sorted(recs, key=lambda x: (DATASET_ORDER.index(x['dataset']) if x['dataset'] in DATASET_ORDER else 99,
                                         int(x['pl']), x['model'], int(x['ne']), variant(x), x['seed'], x['method'])):
        w.writerow({
            'dataset': r['dataset'], 'model': r['model'], 'model_id': r['model_id'],
            'num_experts': r['ne'], 'variant': variant(r), 'prob_expert': r['pe'],
            'unc_gating': r['ug'], 'features': r['ft'], 'seq_len': r['sl'], 'label_len': r['ll'],
            'pred_len': r['pl'], 'seed': r['seed'], 'method': r['method'],
            'coverage': f(r.get('coverage')), 'width': f(r.get('width')),
            'median_width': f(r.get('median_width')), 'q_mean': f(r.get('q_mean')),
            'alpha': '0.1', 'window_size': window_of(r)[0], 'window_source': window_of(r)[1],
            'retrains': r.get('retrains'), 'unbounded': f(r.get('unbounded')),
            'mse': f(r.get('mse'), 6), 'mae': f(r.get('mae'), 6), 'n_runs_seen': r['n_runs'],
            'coverage_min': r.get('coverage_min'), 'coverage_max': r.get('coverage_max'),
            'width_min': r.get('width_min'), 'width_max': r.get('width_max'),
            'rerun_disagrees': int(bool(r.get('rerun_disagrees'))),
            'd_model': r['dm'], 'n_heads': r['nh'], 'e_layers': r['el'], 'd_layers': r['dl'],
            'd_ff': r['df'], 'expand': r['expand'], 'd_conv': r['dc'], 'factor': r['fc'],
            'embed': r['eb'], 'distil': r['dt'], 'des': r['des'], 'itr': r['itr'],
            'method_params': r.get('method_params', ''), 'source': r['source'],
            'all_sources': r.get('all_sources', ''),
            'setting': r['setting']})

# ------------------------------------------------------------------ markdown
by_setting = collections.defaultdict(dict)
cfg_of = {}
for r in recs:
    by_setting[r['setting']][r['method']] = r
    cfg_of[r['setting']] = r

L = []
A = L.append
A('# Time-series calibration results')
A('')
A(f'Generated {datetime.date.today().isoformat()} from `result_calibration_*.txt` and `logs/**/*.log` '
  'in this repo. Every row is one long-term-forecast run (`task_name=long_term_forecast`); '
  'tabular-regression calibration runs are excluded.')
A('')
A(f'**{len(recs)} (config, method) results** over **{len(by_setting)} distinct training configs**, '
  f'{len(DATASET_ORDER)} datasets, {len({r["model"] for r in recs})} backbones.')
A('')
A('Machine-readable companion with every field: [calibration_results_tsf.csv](calibration_results_tsf.csv). '
  'Both files are regenerated by `python scripts/collect_calibration_results.py`, which reruns '
  'daily at 06:00 via cron. Day-by-day history: '
  '[calibration_results_changelog.md](calibration_results_changelog.md).')
A('')


def render_delta(A, d, heading):
    """Shared renderer for the 'what is new' block (md section + changelog entry)."""
    A(heading)
    A('')
    if d['first_run']:
        A('First run — baseline snapshot, nothing to compare against yet.')
        A('')
        return
    if not (d['new_results'] or d['changed_results'] or d['new_files'] or
            d['grown_files'] or d['gone_results']):
        A(f'No change since the previous run ({d["prev_run"]}). '
          'No new logs, no new results, no value moved.')
        A('')
        return
    A(f'Compared with the previous run ({d["prev_run"]}):')
    A('')
    A('| change | count | detail |')
    A('|---|---|---|')
    if d.get('new_methods'):
        A('| **new calibration method** | ' + str(len(d['new_methods'])) + ' | ' +
          ', '.join('`' + m + '`' for m in d['new_methods']) +
          ' — first time this method has produced time-series results |')
    if d['new_results']:
        top = ', '.join(f'{k} ({v})' for k, v in d['by_dataset'].most_common(4))
        A(f'| new calibration results | **{len(d["new_results"])}** | {top} |')
    if d['new_settings']:
        A(f'| new training configs | **{len(d["new_settings"])}** | configs never seen before |')
    if d['changed_results']:
        A(f'| results that moved | {len(d["changed_results"])} | same config+method, different '
          'coverage/width (model retrained or rerun) |')
    if d['gone_results']:
        A(f'| results that disappeared | {len(d["gone_results"])} | source file truncated, '
          'rotated or deleted |')
    if d['new_files']:
        A(f'| new source files | {len(d["new_files"])} | {", ".join("`" + x + "`" for x in d["new_files"][:5])}'
          f'{" …" if len(d["new_files"]) > 5 else ""} |')
    if d['grown_files']:
        A(f'| source files appended to | {len(d["grown_files"])} | '
          f'{", ".join("`" + x + "`" for x in d["grown_files"][:5])}'
          f'{" …" if len(d["grown_files"]) > 5 else ""} |')
    if d['gone_files']:
        A(f'| source files gone | {len(d["gone_files"])} | '
          f'{", ".join("`" + x + "`" for x in d["gone_files"][:5])} |')
    A('')
    if d['new_results']:
        A('New results by method: ' +
          ', '.join(f'**{METHOD_LABEL.get(k, k)}** {v}' for k, v in d['by_method'].most_common()) + '.')
        A('')
        A('New results by seed: ' +
          ', '.join(f'{k} ({v})' for k, v in sorted(d['by_seed'].items())) + '.')
        A('')


render_delta(A, delta, '## 0. What arrived since the last run')

A('## 1. Where the numbers come from')
A('')
A('| source | results taken from it | note |')
A('|---|---|---|')
src_counts = collections.Counter(r['source'] if not r['source'].startswith('logs/') else 'logs/**/*.log' for r in recs)
# Methods that exist only in stdout, i.e. never reached a result_*.txt file.
_log_only = sorted({r['method'] for r in recs
                    if not any(s.endswith('.txt')
                               for s in (r.get('all_sources') or r['source']).split(';'))})
for s, n in src_counts.most_common():
    note = 'written by the calibration code itself' if s.endswith('.txt') else \
           ('recovered from stdout — older runs that never reached a result file'
            + (', and the only source for ' + ', '.join(METHOD_LABEL.get(m, m) for m in _log_only)
               if _log_only else ''))
    A(f'| `{s}` | {n} | {note} |')
A('')
n_multi = sum(1 for r in recs if r['n_runs'] > 1)
n_dis = sum(1 for r in recs if r.get('rerun_disagrees'))
A(f'{n_multi} of the {len(recs)} (config, method) pairs were observed more than once (the same '
  'experiment re-run, or the same run present in both a `result_*.txt` file and a log). '
  f'For **{n_dis}** of them the repeats do not agree — the model was retrained, so the calibration '
  'numbers moved slightly. In that case the tables show the value from the `result_*.txt` file '
  '(the canonical record written by the calibration code) and the CSV carries `coverage_min`, '
  '`coverage_max`, `width_min`, `width_max`, `n_runs_seen` and `rerun_disagrees` so the spread '
  'stays visible.')
A('')

A('## 2. Config encoding')
A('')
A('Every run is identified by the `setting` string built in [run.py:281](../run.py#L281):')
A('')
A('```')
A('long_term_forecast_{model_id}_{model}_{dataset}_ne{num_experts}_pe{prob_expert}_ug{unc_gating}')
A('  _ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_dm{d_model}_nh{n_heads}_el{e_layers}')
A('  _dl{d_layers}_df{d_ff}_expand{expand}_dc{d_conv}_fc{factor}_eb{embed}_dt{distil}')
A('  _{des}_{itr}_seed{seed}')
A('```')
A('')
A('**Varying fields** (the ones in the tables): dataset, model, `model_id`, `num_experts`, '
  '`prob_expert`/`unc_gating` (shown as the variant), `features`, `pred_len`, `seed`.')
A('')
A('**Features type** (`--features`, column `feat` in the tables):')
A('')
A('| feat | meaning | runs | configs |')
A('|---|---|---|---|')
for ftv, meaning in [('M', 'multivariate -> multivariate (all channels predicted)'),
                     ('MS', 'multivariate -> univariate (all channels in, target `OT` out)'),
                     ('S', 'univariate -> univariate (target `OT` only)')]:
    rs = [r for r in recs if r['ft'] == ftv]
    A(f'| `{ftv}` | {meaning} | {len(rs)} | {len({r["setting"] for r in rs})} |')
A('')
A('The `S` / `MS` runs are a small side-sweep: ETTh1/ETTh2/ETTm1/ETTm2, iTransformer + PatchTST, '
  '`num_experts=3`, `pred_len=96`, seed 4021 only. Their widths/MSE are on a different target '
  'scale (one channel instead of all), so do not compare them against the `M` rows.')
A('')
A('**Fixed across all runs collected here:**')
A('')
A('| field | value | | field | value |')
A('|---|---|---|---|---|')
fixed = [('seq_len', '96'), ('label_len', '48'),
         ('d_model', '512'), ('n_heads', '8'), ('e_layers', '2'), ('d_layers', '1'),
         ('d_ff', '2048'), ('expand', '2'), ('d_conv', '4'), ('factor', '1'),
         ('embed', 'timeF'), ('distil', 'True'), ('des', 'test'), ('itr', '0'), ('alpha', '0.1 (target coverage 0.90)')]
half = (len(fixed) + 1) // 2
for i in range(half):
    a = fixed[i]
    b = fixed[i + half] if i + half < len(fixed) else ('', '')
    A(f'| `{a[0]}` | {a[1]} | | ' + (f'`{b[0]}` | {b[1]} |' if b[0] else ' | |'))
A('')
A('**Variant** = the `prob_expert` / `unc_gating` pair:')
A('')
A('| variant | flags | meaning |')
A('|---|---|---|')
A('| MOE | `pe0 ug0` | deterministic mixture of experts, point forecast only |')
A('| MOG | `pe1 ug0` | probabilistic experts (`--prob_expert`), mixture-of-Gaussians head |')
A('| MOGU | `pe1 ug1` | probabilistic experts + uncertainty-derived gating (`--unc_gating`) |')
A('')
A('`MOGU` with `num_experts=1` does not exist: [run.py](../run.py) exits with '
  '`SKIPPING: Redundant experiment (Single Expert + Unc Gating)`, so those cells are structurally '
  'absent, not missing runs.')
A('')
A('`model_id` is `test` for the standard training runs and `cqr` for the models retrained with '
  'pinball loss (`--use_quantile_loss`) that the CQR methods need — a `cqr` row is a *different '
  'trained model*, not a different calibrator on the same model.')
A('')

A('## 3. Methods')
A('')
A('| method | configs run | window | result file | definition |')
A('|---|---|---|---|---|')
for m in METHOD_ORDER:
    rs = [r for r in recs if r['method'] == m]
    if not rs:
        continue
    tf = TXT_FILES.get(m, 'logs only')
    _w = collections.Counter(window_of(r)[0] or '?' for r in rs)
    if len(_w) == 1:
        win = next(iter(_w))
        win = 'unknown' if win == '?' else win
    else:
        _known = ', '.join(f'{k} ({v})' for k, v in _w.most_common() if k != '?')
        win = _known + (f' / unknown ({_w["?"]})' if _w.get('?') else '')
    A(f'| **{METHOD_LABEL[m]}** (`{m}`) | {len({r["setting"] for r in rs})} | {win} | {tf} | {METHOD_DESC[m]} |')
A('')
_n_unknown = sum(1 for r in recs if not window_of(r)[0])
A('**Window** is the sliding-window length (number of most recent calibration scores kept). '
  'Every method above runs at `alpha=0.1`. The window is only reported where it is actually '
  f'established; for **{_n_unknown} rows it is left blank** rather than guessed, and '
  '`window_size` is empty in the CSV with `window_source` = `unknown - run cannot be dated`.')
A('')
A('It is treated as known when the run recorded it itself, or when the run can be dated to a '
  'period where the code held one value. The window moved 300 -> 500 -> 1000 over the repo '
  '(3f54bba 2026-01-01, 0ca1060 2026-01-09 / 3182ab8 2026-01-13, bef8056 2026-01-22), and a row '
  'is datable only if it appears in a log file whose mtime pins it after the last change. Rows '
  'that exist solely in a `result_calibration_*.txt` carry no timestamp, so they get no window.')
A('')
A('')
A('| method | window | basis |')
A('|---|---|---|')
_methods_present = [m for m in METHOD_ORDER if any(r['method'] == m for r in recs)]
for m in _methods_present:
    _rs = [r for r in recs if r['method'] == m]
    _b = collections.Counter(window_of(r) for r in _rs)
    for (_val, _src), _n in _b.most_common():
        A(f'| {METHOD_LABEL[m]} | {_val or "**not recorded**"} | {_src}'
          f'{f" ({_n} rows)" if len(_b) > 1 else ""} |')
A('')
A('The one method whose window is exact for every row is **CQR retrain**, because it writes '
  '`window=` into its own result line. Making the other calibrators do the same would remove this '
  'whole class of uncertainty going forward.')
A('')
A('Metrics: **cov** = empirical marginal coverage of the 90% interval on the test split '
  '(target 0.90), **width** = mean interval width in the standardised scale, '
  '**q** = mean conformal quantile. MSE/MAE are the point-forecast metrics of the same '
  'trained model, from `result_long_term_forecast.txt` (or the test block of the log).')
A('')

A('## 4. Run counts — dataset x method')
A('')
present = [m for m in METHOD_ORDER if any(r['method'] == m for r in recs)]
A('| dataset | ' + ' | '.join(METHOD_LABEL[m] for m in present) + ' | total |')
A('|---' * (len(present) + 2) + '|')
for ds in DATASET_ORDER:
    row = [ds]
    tot = 0
    for m in present:
        n = sum(1 for r in recs if r['dataset'] == ds and r['method'] == m)
        tot += n
        row.append(str(n) if n else '–')
    row.append(f'**{tot}**')
    A('| ' + ' | '.join(row) + ' |')
row = ['**total**']
for m in present:
    row.append(f'**{sum(1 for r in recs if r["method"] == m)}**')
row.append(f'**{len(recs)}**')
A('| ' + ' | '.join(row) + ' |')
A('')

A('## 5. Aggregate summary')
A('')
A('Mean over all `features=M` runs of the method (unweighted; different methods cover different '
  'config subsets, so compare within a row of the per-dataset tables, not across these means). '
  'The `S`/`MS` runs are excluded here because their widths live on a different target scale.')
A('')
recs_m = [r for r in recs if r['ft'] == 'M']

A('### Overall')
A('')
A('| method | runs | mean cov | median cov | mean width | mean abs. coverage gap |')
A('|---|---|---|---|---|---|')


def agg(rs):
    covs = sorted(float(r['coverage']) for r in rs if r.get('coverage'))
    wids = [float(r['width']) for r in rs if r.get('width')]
    if not covs:
        return None
    med = covs[len(covs) // 2] if len(covs) % 2 else (covs[len(covs) // 2 - 1] + covs[len(covs) // 2]) / 2
    return (len(rs), sum(covs) / len(covs), med,
            sum(wids) / len(wids) if wids else float('nan'),
            sum(abs(c - 0.90) for c in covs) / len(covs))


for m in present:
    rs = [r for r in recs_m if r['method'] == m]
    a = agg(rs)
    if a:
        A(f'| {METHOD_LABEL[m]} | {a[0]} | {a[1]:.4f} | {a[2]:.4f} | {a[3]:.4f} | {a[4]:.4f} |')
A('')
A('### Mean coverage per dataset x method')
A('')
A('| dataset | ' + ' | '.join(METHOD_LABEL[m] for m in present) + ' |')
A('|---' * (len(present) + 1) + '|')
for ds in DATASET_ORDER:
    row = [ds]
    for m in present:
        a = agg([r for r in recs_m if r['dataset'] == ds and r['method'] == m])
        row.append(f'{a[1]:.4f}' if a else '–')
    A('| ' + ' | '.join(row) + ' |')
A('')
A('### Mean width per dataset x method')
A('')
A('| dataset | ' + ' | '.join(METHOD_LABEL[m] for m in present) + ' |')
A('|---' * (len(present) + 1) + '|')
for ds in DATASET_ORDER:
    row = [ds]
    for m in present:
        a = agg([r for r in recs_m if r['dataset'] == ds and r['method'] == m])
        row.append(f'{a[3]:.4f}' if a and a[3] == a[3] else '–')
    A('| ' + ' | '.join(row) + ' |')
A('')

A('## 6. ETT expert sweep — MOG, `num_experts` 1..5')
A('')
A('The one place in the repo where expert count is swept on a fixed everything-else: ETT only, '
  'MOG variant (`prob_expert=1`, `unc_gating=0`), iTransformer backbone, `features=M`, '
  '`model_id=test`, `seq_len=96`, alpha 0.1 (target coverage 0.90). Everything else in this file '
  'mixes expert counts across backbones and variants; these tables do not.')
A('')

ETT = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
NE = ['1', '2', '3', '4', '5']
sweep = [r for r in recs_m
         if r['dataset'] in ETT and variant(r) == 'MOG' and r['model'] == 'iTransformer'
         and r['model_id'] == 'test' and r['ne'] in NE and r.get('coverage')]


def _cells(rs):
    """(mean coverage, mean width, n) over a set of runs."""
    covs = [float(r['coverage']) for r in rs]
    wids = [float(r['width']) for r in rs if r.get('width')]
    if not covs:
        return None
    return (sum(covs) / len(covs),
            sum(wids) / len(wids) if wids else float('nan'),
            len(covs))


sweep_pls = [pl for pl in sorted({r['pl'] for r in sweep}, key=int)
             if {r['ne'] for r in sweep if r['pl'] == pl} == set(NE)]
for _i, pl in enumerate(sweep_pls, 1):
    pl_recs = [r for r in sweep if r['pl'] == pl]
    pl_methods = [m for m in present if any(r['method'] == m for r in pl_recs)]

    # A method is "balanced" only if every (dataset, n_exp) cell exists with the same seed
    # count. Mixing a 5-seed cell with a 1-seed cell in the same row would make the trend
    # across experts an artifact of which seeds happened to run, so those rows get flagged.
    def _seedset(m, ds, ne):
        return {r['seed'] for r in pl_recs
                if r['method'] == m and r['dataset'] == ds and r['ne'] == ne}

    full, partial = [], []
    for m in pl_methods:
        sizes = {len(_seedset(m, ds, ne)) for ds in ETT for ne in NE}
        (full if sizes and len(sizes) == 1 and 0 not in sizes else partial).append(m)
    ordered = full + partial

    def _label(m):
        return METHOD_LABEL[m] + ('' if m in full else ' \\*')

    seeds_pl = sorted({r['seed'] for r in pl_recs})
    A(f'### 6.{_i} — pred_len {pl}')
    A('')
    note = (f'{len(full)} of {len(pl_methods)} methods cover the full 4-dataset x 5-expert grid '
            'with an equal number of seeds per cell; the rest are marked **\\*** and their trend '
            'across experts is confounded with which configs happened to run.'
            if full else
            f'**No method covers the full 4-dataset x 5-expert grid with an equal number of seeds '
            f'per cell at this horizon** — every row is marked **\\***, and differences across '
            'expert counts are partly differences in which configs ran. Read '
            f'`pred_len {sweep_pls[0]}` for the balanced comparison.')
    A(f'{len(pl_recs)} results, seeds {", ".join(seeds_pl)}. ' + note)
    A('')

    A(f'**Mean coverage over the 4 ETT datasets** (target 0.90)')
    A('')
    A('| method | ' + ' | '.join(f'ne={n}' for n in NE) + ' |')
    A('|---' * (len(NE) + 1) + '|')
    for m in ordered:
        row = [_label(m)]
        for ne in NE:
            a = _cells([r for r in pl_recs if r['method'] == m and r['ne'] == ne])
            row.append(f'{a[0]:.4f}' if a else '–')
        A('| ' + ' | '.join(row) + ' |')
    A('')

    A('**Mean width over the 4 ETT datasets**')
    A('')
    A('| method | ' + ' | '.join(f'ne={n}' for n in NE) + ' |')
    A('|---' * (len(NE) + 1) + '|')
    for m in ordered:
        row = [_label(m)]
        for ne in NE:
            a = _cells([r for r in pl_recs if r['method'] == m and r['ne'] == ne])
            row.append(f'{a[1]:.4f}' if a and a[1] == a[1] else '–')
        A('| ' + ' | '.join(row) + ' |')
    A('')

    # Raw widths live on four different target scales, so their cross-dataset mean is dominated
    # by whichever dataset has the largest values. The ratio is taken per dataset first.
    A('**Width relative to `ne=1`** — per-dataset ratio, then averaged (< 1.00 = tighter than the '
      'single-expert model)')
    A('')
    A('| method | ' + ' | '.join(f'ne={n}' for n in NE) + ' |')
    A('|---' * (len(NE) + 1) + '|')
    for m in ordered:
        row = [_label(m)]
        for ne in NE:
            ratios = []
            for ds in ETT:
                a = _cells([r for r in pl_recs if r['method'] == m and r['ne'] == ne and r['dataset'] == ds])
                b = _cells([r for r in pl_recs if r['method'] == m and r['ne'] == '1' and r['dataset'] == ds])
                if a and b and a[1] == a[1] and b[1] == b[1] and b[1]:
                    ratios.append(a[1] / b[1])
            row.append(f'{sum(ratios) / len(ratios):.3f}' if len(ratios) == len(ETT) else '–')
        A('| ' + ' | '.join(row) + ' |')
    A('')

    A('**Runs behind each cell** (4 datasets x seeds)')
    A('')
    A('| method | ' + ' | '.join(f'ne={n}' for n in NE) + ' |')
    A('|---' * (len(NE) + 1) + '|')
    for m in ordered:
        row = [_label(m)]
        for ne in NE:
            a = _cells([r for r in pl_recs if r['method'] == m and r['ne'] == ne])
            row.append(str(a[2]) if a else '–')
        A('| ' + ' | '.join(row) + ' |')
    A('')

    A('**Point accuracy of the underlying models** — one row per (dataset, n_exp), mean over seeds, '
      'independent of the calibrator')
    A('')
    A('| dataset | ' + ' | '.join(f'ne={n}' for n in NE) + ' |')
    A('|---' * (len(NE) + 1) + '|')
    for ds in ETT:
        row = [f'{ds} (MSE / MAE)']
        for ne in NE:
            per_cfg = {}
            for r in pl_recs:
                if r['dataset'] == ds and r['ne'] == ne and r.get('mse') is not None:
                    per_cfg[r['setting']] = (float(r['mse']), float(r['mae']))
            if per_cfg:
                v = list(per_cfg.values())
                row.append(f'{sum(x[0] for x in v) / len(v):.4f} / {sum(x[1] for x in v) / len(v):.4f}')
            else:
                row.append('–')
        A('| ' + ' | '.join(row) + ' |')
    A('')

    A(f'**Per dataset** — cells are `coverage / width`, mean over seeds')
    A('')
    A('| dataset | method | ' + ' | '.join(f'ne={n}' for n in NE) + ' |')
    A('|---' * (len(NE) + 2) + '|')
    for ds in ETT:
        for m in ordered:
            row = [ds, _label(m)]
            for ne in NE:
                a = _cells([r for r in pl_recs if r['method'] == m and r['ne'] == ne and r['dataset'] == ds])
                row.append(f'{a[0]:.4f} / {a[1]:.4f}' if a and a[1] == a[1] else '–')
            if any(c != '–' for c in row[2:]):
                A('| ' + ' | '.join(row) + ' |')
    A('')

A('## 7. Full results per dataset')
A('')
A('One row per training config. Cells are `coverage / width`; `–` means that method was never run '
  'on that config. Sorted by horizon, backbone, expert count, variant, seed.')
A('')
for ds in DATASET_ORDER:
    ds_recs = [r for r in recs if r['dataset'] == ds]
    if not ds_recs:
        continue
    settings = sorted({r['setting'] for r in ds_recs})
    ds_methods = [m for m in present if any(r['method'] == m for r in ds_recs)]
    A(f'### {ds}')
    A('')
    A(f'{len(settings)} configs, {len(ds_recs)} calibration results, '
      f'seeds {", ".join(sorted({r["seed"] for r in ds_recs}))}, '
      f'horizons {", ".join(sorted({r["pl"] for r in ds_recs}, key=int))}.')
    A('')
    for pl in sorted({r['pl'] for r in ds_recs}, key=int):
        sub = [s for s in settings if cfg_of[s]['pl'] == pl]
        if not sub:
            continue
        sub_recs = [r for r in ds_recs if r['pl'] == pl]
        pl_methods = [m for m in ds_methods if any(r['method'] == m for r in sub_recs)]
        A(f'#### {ds} — pred_len {pl}')
        A('')
        hdr = ['model', 'n_exp', 'variant', 'feat', 'seed', 'train', 'MSE', 'MAE'] + [METHOD_LABEL[m] for m in pl_methods]
        A('| ' + ' | '.join(hdr) + ' |')
        A('|' + '---|' * len(hdr))
        sub.sort(key=lambda s: (cfg_of[s]['model'], int(cfg_of[s]['ne']), variant(cfg_of[s]),
                                {'M': 0, 'MS': 1, 'S': 2}.get(cfg_of[s]['ft'], 3),
                                cfg_of[s]['seed'], cfg_of[s]['model_id']))
        for s in sub:
            c = cfg_of[s]
            got = by_setting[s]
            mse = mae = '–'
            for r in got.values():
                if r.get('mse') is not None:
                    mse, mae = f'{r["mse"]:.4f}', f'{r["mae"]:.4f}'
                    break
            row = [c['model'], c['ne'], variant(c), c['ft'], c['seed'], c['model_id'], mse, mae]
            for m in pl_methods:
                r = got.get(m)
                if not r or not r.get('coverage'):
                    row.append('–')
                else:
                    cell = (f'{float(r["coverage"]):.4f} / {float(r["width"]):.4f}'
                           if r.get('width') else f'{float(r["coverage"]):.4f} / –')
                    # Coverage counts every timestep; width, unavoidably, only the finite-
                    # width ones (an infinite interval can't enter a mean). Flag it inline so
                    # the two numbers are never read as describing the same population.
                    unb = float(r['unbounded']) if r.get('unbounded') else 0.0
                    if unb > 0:
                        cell += f' ({unb:.0%} ∞)'
                    row.append(cell)
            A('| ' + ' | '.join(row) + ' |')
        A('')

A('## 8. What is *not* here')
A('')
A('Every `result_calibration_*.txt` in the repo, and whether it holds any time-series rows. '
  'Counted fresh each run, so a file that starts producing `long_term_forecast` results moves '
  'out of the tabular-only group on its own:')
A('')
A('| result file | long_term_forecast rows | tabular_regression rows |')
A('|---|---|---|')
_tab_only = []
for _p in sorted(glob.glob(os.path.join(ROOT, 'result_calibration_*.txt'))):
    _txt = open(_p, errors='ignore').read()
    _n_ts = len(re.findall(r'(?m)^long_term_forecast\S+ \(', _txt))
    _n_tab = len(re.findall(r'(?m)^tabular_regression\S+ \(', _txt))
    _fn = os.path.basename(_p)
    A(f'| `{_fn}` | {_n_ts if _n_ts else "–"} | {_n_tab if _n_tab else "–"} |')
    if not _n_ts and _n_tab:
        _tab_only.append(_fn)
A('')
A(f'So {len(_tab_only)} of these are still tabular-only — the HPD family (MoG/STA/SETA/MELD/EASE), '
  'MoCE, plain CQR, the static/aleatoric CP-VS variants and the tabular copies of Standard CP and '
  'MoECP have no long-term-forecast run recorded yet.')
A('')
A('Structural gaps in the tables above:')
A('')
A('* **CPVS / Aleatoric-* on MOE rows** — these calibrators consume the predicted MoG variance, '
  'which only exists with `--prob_expert`. MOE configs can only be calibrated with Standard CP.')
A('* **MOGU + `num_experts=1`** — refused by `run.py` as degenerate.')
A('* **CQR rows** — only exist where a `model_id=cqr` (pinball-loss) model was trained.')
A('* **MoECP rows** — need `--prob_expert` (it consumes the gating distribution), and only '
  'the `tau=1.0` entries are kept.')
A('')

A('## 9. Separate offline benchmark: `results_mog_calibration.csv`')
A('')
A('[scripts/benchmark_mog_calibration.py](../scripts/benchmark_mog_calibration.py) runs its own '
  'split-conformal sweep from cached predictions and writes '
  '[results_mog_calibration.csv](../results_mog_calibration.csv) / '
  '[results_mog_calibration.md](../results_mog_calibration.md). It is **not** merged into the '
  'tables above because the protocol differs (fit on validation, evaluate on test, quantile taken '
  'per horizon-step and channel; CP-DVS additionally splits validation 50/50).')
A('')
A('| field | values |')
A('|---|---|')
A('| datasets | ETTh1, ETTh2, ETTm1, ETTm2 |')
A('| backbone | iTransformer |')
A('| num_experts / variant | 3 / MOG (`unc_gating=0`) |')
A('| pred_len | 96, 192, 336, 720 |')
A('| seeds | 4021, 4022, 4023 |')
A('| alpha | 0.05 and 0.1 (both) |')
A('| methods | standard_cp, cpvs, cpvs_within, aleatoric_only, cp_dvs |')
A('| rows | 480 (48 runs x 2 alphas x 5 methods) |')
A('| extra metrics | interval score, pinball, width CV, SSC min-coverage, worst-horizon / '
  'worst-channel coverage, block coverage, violation clustering |')
A('')
A('`cpvs_within` and `aleatoric_only` are algebraically the same estimator there, which is why '
  'their columns match.')
A('')

open(OUT_MD, 'w').write('\n'.join(L) + '\n')

# ------------------------------------------------------------------ changelog
# Newest entry first, so the top of the file is always the most recent day.
stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
E = []
render_delta(E.append, delta,
             f'## {stamp} — {len(recs)} results / {len(by_setting)} configs')
entry = '\n'.join(E).rstrip() + '\n'

HEADER = ('# Calibration results changelog\n\n'
          'What each run of `scripts/collect_calibration_results.py` picked up, newest first. '
          'The run rescans every file under `logs/` and every `result_calibration_*.txt`, so new '
          'log files are found on their own; this file records what was new that day.\n')
old = ''
if os.path.exists(OUT_CHANGELOG):
    old = open(OUT_CHANGELOG).read()
    if old.startswith(HEADER):
        old = old[len(HEADER):]
    else:  # older/hand-edited file: keep everything below its first heading
        parts = old.split('\n## ', 1)
        old = ('\n## ' + parts[1]) if len(parts) > 1 else ''
open(OUT_CHANGELOG, 'w').write(HEADER + '\n' + entry + '\n' + old.lstrip('\n'))

json.dump({'generated': stamp, 'files': now_files, 'results': now_results},
          open(STATE, 'w'))

print('wrote', OUT_MD, len(L), 'lines')
print('wrote', OUT_CSV)
print('wrote', OUT_CHANGELOG)
if delta['first_run']:
    print('delta: first run, baseline snapshot')
else:
    print(f'delta: +{len(delta["new_results"])} results, '
          f'~{len(delta["changed_results"])} changed, '
          f'+{len(delta["new_files"])} new source files, '
          f'{len(delta["grown_files"])} appended to')
