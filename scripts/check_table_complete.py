"""Report which cells of the seed-4021 headline calibration table are still missing.

Prints one "MISSING <dataset> <method>" line per empty cell and exits 1, or prints
"COMPLETE" and exits 0.

Reads docs/calibration_results_tsf.csv, regenerating it first via
collect_calibration_results.py. That collector parses BOTH the result_calibration_*.txt
files and logs/**/*.log -- which matters: weather, electricity and traffic have their CP /
CPVS / aleatoric-only results only in logs/run_mog_cp_all_datasets2.log, and a txt-only
check reports them missing forever.

Config: iTransformer, num_experts=3, features=M, unc_gating=0, seed 4021, pred_len 96
(national-illness 24). MoG methods are the MOG variant; CQR is the separate MOE model.
"""
import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'docs', 'calibration_results_tsf.csv')

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity',
            'traffic', 'exchange-rate', 'national-illness']
PL = {'national-illness': 24}

METHODS = [  # (label, collector method slug, variant)
    ('CP',             'standard_cp',               'MOG'),
    ('CP+ACI',         'aci_cp',                    'MOG'),
    ('CPVS',           'cpvs',                      'MOG'),
    ('CPVS+ACI',       'aci_cpvs',                  'MOG'),
    ('CPVS-aleatoric', 'aleatoric_only',            'MOG'),
    ('CPVS-aleat+ACI', 'aci_aleatoric_only',        'MOG'),
    ('CP-MOG',         'cp_aleatoric_scale',        'MOG'),
    ('ACI g=0.01',     'aci_aleatoric_scale',       'MOG'),
    ('ACI g=0.001',    'aci_aleatoric_scale_g001',  'MOG'),
    ('MoECP',          'moecp',                     'MOG'),
    ('MoECP+ACI',      'aci_moecp',                 'MOG'),
    ('CQR quantile',   'cqr_quantile',              'MOE'),
    ('CQR quant+ACI',  'aci_cqr_quantile',          'MOE'),
    ('CQR retrain+ACI', 'aci_cqr_retrain',          'MOE'),
]

subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'collect_calibration_results.py')],
               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

have = set()
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        if (r['model'] == 'iTransformer' and r['num_experts'] == '3' and r['seed'] == '4021'
                and r['features'] == 'M' and r['unc_gating'] == '0' and r['coverage']):
            have.add((r['dataset'], int(r['pred_len']), r['method'], r['variant']))

missing = [(d, label) for d in DATASETS for label, slug, variant in METHODS
           if (d, PL.get(d, 96), slug, variant) not in have]

for d, label in missing:
    print(f'MISSING {d} {label}')
total = len(DATASETS) * len(METHODS)
print('COMPLETE' if not missing else f'{total - len(missing)}/{total} cells filled')
sys.exit(1 if missing else 0)
