"""Sanity-check SynthMoE3 against the three properties it was built to have.

Run before spending GPU time: if any of these fail, the dataset does not pose the problem
the MoE calibration methods are meant to solve, and the numbers off it would mean nothing.

1. Every split (Dataset_Custom's fixed 70/10/20) sees all three regimes. Calibrators fit
   on val and are evaluated on test, so a regime missing from either makes the comparison
   about extrapolation rather than about calibration.
2. The active regime is recoverable from a seq_len window alone. If it is not, the gate
   cannot route and the experts cannot specialize.
3. Per-horizon irreducible error differs by regime. This is the heteroscedasticity that
   variance-scaling calibrators convert into narrower intervals; without it, standard CP
   is already optimal and every method ties.
"""

import numpy as np
import pandas as pd

CSV = './data/long_term_forecast/synthetic/synth_moe3.csv'
NPZ = './data/long_term_forecast/synthetic/synth_moe3_latent.npz'
SEQ_LEN, PRED_LEN = 96, 96

df = pd.read_csv(CSV)
lat = np.load(NPZ)
regime, sigma, clean = lat['regime'], lat['sigma'], lat['clean']
n = len(df)
vals = df.drop(columns=['date']).values

num_train = int(n * 0.7)
num_test = int(n * 0.2)
num_vali = n - num_train - num_test
bounds = {'train': (0, num_train),
          'val': (num_train, num_train + num_vali),
          'test': (n - num_test, n)}

print('=' * 70)
print('1. Regime occupancy per split')
print('=' * 70)
ok1 = True
for name, (a, b) in bounds.items():
    occ = [np.mean(regime[a:b] == k) for k in range(3)]
    n_sw = int((np.diff(regime[a:b]) != 0).sum())
    flag = '' if min(occ) > 0.05 else '   <-- REGIME UNDER-REPRESENTED'
    if min(occ) <= 0.05:
        ok1 = False
    print(f'  {name:5s} [{a:6d},{b:6d})  n={b - a:6d}  switches={n_sw:3d}  '
          f'R0={occ[0]:.3f} R1={occ[1]:.3f} R2={occ[2]:.3f}{flag}')

print()
print('=' * 70)
print('2. Is the regime identifiable from a 96-step input window?')
print('=' * 70)
# Windows fully inside one regime, described by cheap frequency/scale features. A linear
# model on these is a deliberately weak stand-in for the gate: if it separates the
# regimes, a trained gate certainly can.
starts = np.arange(0, n - SEQ_LEN)
win_reg = regime[starts[:, None] + np.arange(SEQ_LEN)]
pure = (win_reg == win_reg[:, :1]).all(axis=1)
starts_pure, lab = starts[pure], win_reg[pure, 0]

feats = []
for s in starts_pure:
    w = vals[s:s + SEQ_LEN]
    d1 = np.diff(w, axis=0)
    feats.append([w.std(axis=0).mean(),           # level scale
                  np.abs(d1).mean(),              # roughness -> dominant frequency
                  np.abs(np.diff(d1, axis=0)).mean()])
feats = np.array(feats)

rng = np.random.default_rng(0)
perm = rng.permutation(len(feats))
cut = int(0.7 * len(perm))
tr, te = perm[:cut], perm[cut:]

mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-9
Xtr, Xte = (feats[tr] - mu) / sd, (feats[te] - mu) / sd
# Nearest class-centroid: no fitting beyond three means, so this is a floor on separability.
cent = np.stack([Xtr[lab[tr] == k].mean(0) for k in range(3)])
pred = np.argmin(((Xte[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
acc = np.mean(pred == lab[te])
print(f'  pure-regime windows: {len(feats)} of {len(starts)}  '
      f'({len(feats) / len(starts):.1%} sit inside a single regime)')
print(f'  3-class nearest-centroid accuracy on 3 hand features: {acc:.3f}  (chance = 0.333)')
for k in range(3):
    print(f'    R{k} feature centroid (level, roughness, curvature): '
          f'{cent[k][0]:+.2f} {cent[k][1]:+.2f} {cent[k][2]:+.2f}')
ok2 = acc > 0.9

print()
print('=' * 70)
print('3. Irreducible (oracle) error by regime and horizon')
print('=' * 70)
# Exact, not probed. The DGP is known, so the h-step floor can be written down: R0 and R1
# are deterministic given the regime and phase, so a forecaster that has identified the
# regime faces only the observation noise, flat in h. R2 adds an AR(1) whose optimal
# h-step predictor is rho^h * (current state), leaving variance (1 - rho^2h) that nothing
# can remove -- so R2's floor climbs with h and saturates near its marginal std. Probing
# this with a short linear regression instead would confound the floor with the probe's
# own weakness (an 8-lag probe cannot extrapolate a 168-period sinusoid, and reports R0 as
# ~6x worse than it is).
amp, rho, ar_amp = lat['amp'], float(lat['ar_rho']), float(lat['ar_amp'])
noise_scale = lat['noise_scale']
print(f'  {"horizon":>8s} | ' + ' | '.join(f'{"R" + str(k) + " floor":>9s}' for k in range(3)))
for h in (1, 24, 48, 96):
    row = []
    for k in range(3):
        obs = float((noise_scale[k] * amp).mean())          # observation noise
        if k == 2:
            ar_var = (ar_amp * amp) ** 2 * (1 - rho ** (2 * h))
            row.append(float(np.sqrt((noise_scale[k] * amp) ** 2 + ar_var).mean()))
        else:
            row.append(obs)
    print(f'  {h:8d} | ' + ' | '.join(f'{v:9.3f}' for v in row))
ratio = sigma[regime == 2].mean() / sigma[regime == 0].mean()
print(f'  sigma ratio R2/R0 = {ratio:.2f}x  '
      f'(a constant-width CP interval must pay the R2 width on every R0 step)')
ok3 = ratio > 3

print()
print('=' * 70)
print('4. Can one expert fit all three regimes? (per-regime best linear AR(96) fit)')
print('=' * 70)
# Fit a single least-squares AR map from a 96-step window to the next step on each regime
# separately, then cross-apply. Large off-diagonal error = the regimes need different
# maps = there is something for three experts to divide.
ch = 0
X, Y, L = [], [], []
for s in starts_pure[::7]:
    X.append(vals[s:s + SEQ_LEN, ch])
    Y.append(vals[s + SEQ_LEN, ch])
    L.append(regime[s])
X, Y, L = np.array(X), np.array(Y), np.array(L)
W = {}
for k in range(3):
    m = L == k
    W[k] = np.linalg.lstsq(X[m], Y[m], rcond=None)[0]
print(f'  rows = expert fitted on regime i, cols = evaluated on regime j (MSE, channel {ch})')
print(f'  {"":10s}' + ''.join(f'{"eval R" + str(j):>12s}' for j in range(3)))
for i in range(3):
    errs = [np.mean((X[L == j] @ W[i] - Y[L == j]) ** 2) for j in range(3)]
    print(f'  fit R{i}    ' + ''.join(f'{e:12.4f}' for e in errs))
Wall = np.linalg.lstsq(X, Y, rcond=None)[0]
shared = [np.mean((X[L == j] @ Wall - Y[L == j]) ** 2) for j in range(3)]
spec = [np.mean((X[L == j] @ W[j] - Y[L == j]) ** 2) for j in range(3)]
print(f'  {"shared":10s}' + ''.join(f'{e:12.4f}' for e in shared))
print(f'  {"specialist":10s}' + ''.join(f'{e:12.4f}' for e in spec))
gain = (np.mean(shared) - np.mean(spec)) / np.mean(shared)
print(f'  specializing cuts one-step MSE by {gain:.1%} over a single shared linear map')
ok4 = gain > 0.05

print()
print('=' * 70)
print(f'  1 all regimes in every split ....... {"PASS" if ok1 else "FAIL"}')
print(f'  2 regime readable from window ...... {"PASS" if ok2 else "FAIL"}')
print(f'  3 heteroscedastic across regimes ... {"PASS" if ok3 else "FAIL"}')
print(f'  4 experts beat a shared map ........ {"PASS" if ok4 else "FAIL"}')
print('=' * 70)
