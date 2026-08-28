"""Generate SynthMoE3: a 3-regime switching time series built for a 3-expert MoE.

The tabular side of this repo already has a synthetic benchmark whose whole point is a
mixture structure the gate can recover -- `Dataset_Synthetic` draws a domain label in
{linear, quadratic, cubic} and each expert can own one branch. There is no time-series
analogue, so every long-term-forecast number in the repo comes from real data where the
"true" number of experts and the true aleatoric/epistemic split are both unknown.

This script builds that analogue. The generative process is a hidden semi-Markov chain
over three regimes, and each regime differs in **two** ways that matter for this codebase:

1. *Dynamics* -- regimes have different functional form and different dominant periods,
   so a 96-step input window identifies the active regime (the gate has something to
   route on) and no single expert can fit all three well (the experts have a reason to
   specialize).

2. *Noise scale* -- sigma differs by ~5x across regimes, so the aleatoric component is
   genuinely heteroscedastic and, crucially, *predictable from the input window*. Methods
   that scale their interval by a predicted sigma (Aleatoric Scale, Aleatoric Only,
   CP-DVS, Adaptive Variance) have real signal to exploit; standard CP, which spends one
   constant width everywhere, must pay the worst regime's width everywhere.

Regime switches are what make the epistemic component non-trivial: a window sitting near
a switch cannot tell when -- or whether -- the horizon crosses into the next regime, so
the between-expert (epistemic) variance spikes there while within-expert (aleatoric)
variance does not. That is the exact decomposition the MoG variants and MoECP consume.

Regime dwell time is drawn so that most horizons stay inside one regime and a minority
straddle a boundary; if switches were frequent, no expert could specialize and the task
would collapse to "predict the marginal".

Writes a CSV in `Dataset_Custom` format (a `date` column, then channels, then the `OT`
target) plus a companion .npz holding the latent regime path and the per-step noise
scale, which the loader never reads -- it exists so the oracle uncertainty is available
for analysis.
"""

import argparse
import os

import numpy as np
import pandas as pd


# Mean dwell in steps. At hourly freq with seq_len=96 and pred_len=96, a 400-step mean
# dwell puts ~2 switches in every 1000 steps: most (input, horizon) pairs of total span
# 192 sit inside one regime, and roughly 192/400 ~ 40% of them touch a switch somewhere,
# which is the minority that carries the epistemic signal.
MEAN_DWELL = 400

# Dwell lengths are Gamma-distributed rather than geometric. A geometric dwell (the plain
# Markov chain) has CV=1, so over the ~75 switches in a series this long the three regimes
# land in the fixed 70/10/20 split very unevenly -- an early draft put 55% of the test
# split in one regime. Shape 8 gives CV=1/sqrt(8)~0.35, which evens out occupancy across
# splits while keeping both the switch times and the regime order random.
DWELL_SHAPE = 8

# Per-regime noise scale, in units of the channel amplitude. The 5x spread between R0 and
# R2 is what a variance-scaling calibrator can convert into width savings.
NOISE_SCALE = {0: 0.10, 1: 0.25, 2: 0.50}

# Persistence of R2's AR(1) term, and its amplitude relative to the channel amplitude.
# rho < 1 is what makes R2's forecast floor grow with the horizon: the best predictor of
# the AR state h steps out is rho^h times the current state, leaving variance
# (1 - rho^2h) that no model can remove. R0 and R1 are deterministic given the regime and
# phase, so their floor is flat in h. Named here so the verifier can state the exact
# floor rather than probe for it.
AR_RHO = 0.95
AR_AMP = 1.1

# Steps over which a switch is cross-faded. Non-zero so the series has no infinite
# derivative (which would let a model detect switches from a single differenced spike),
# short enough that the regime change is still abrupt relative to the 96-step horizon.
CROSSFADE = 12


def _regime_path(n, n_regimes, rng):
    """Semi-Markov regime path: Gamma dwell lengths, next regime uniform over the others.

    Excluding the current regime from the next draw means a switch is always a real
    change of dynamics, and keeps the chain from silently lengthening a dwell.
    """
    path = np.empty(n, dtype=np.int64)
    cur = int(rng.integers(n_regimes))
    t = 0
    while t < n:
        dwell = max(1, int(round(rng.gamma(DWELL_SHAPE, MEAN_DWELL / DWELL_SHAPE))))
        path[t:t + dwell] = cur
        t += dwell
        others = [k for k in range(n_regimes) if k != cur]
        cur = int(rng.choice(others))
    return path


def _regime_signals(n, n_channels, rng):
    """Noise-free signal for every regime at every step: [n_regimes, n, n_channels].

    Each regime is evaluated over the whole time axis rather than only where it is
    active, so a switch can be cross-faded between two already-defined signals instead of
    restarting a process from an arbitrary state.
    """
    t = np.arange(n, dtype=np.float64)

    # Channel-specific amplitude and phase: channels are driven by the same latent regime
    # (one gate, one regime) but are not copies of each other, so `features=M` is a real
    # multivariate problem.
    amp = rng.uniform(0.7, 1.4, size=n_channels)
    phase = rng.uniform(0, 2 * np.pi, size=(3, n_channels))

    sig = np.empty((3, n, n_channels))

    # R0 "calm seasonal": weekly carrier plus a daily harmonic. Long periods relative to
    # the 96-step window, smooth, and by far the most predictable regime.
    sig[0] = (amp * np.sin(2 * np.pi * t[:, None] / 168.0 + phase[0])
              + 0.6 * amp * np.sin(2 * np.pi * t[:, None] / 24.0 + phase[0][::-1]))

    # R1 "fast oscillatory": two short periods (12 and 8 steps) that are resolvable inside
    # a 96-step window but demand a different frequency response than R0 -- this is the
    # pair that a single shared expert fits worst.
    sig[1] = (amp * np.sin(2 * np.pi * t[:, None] / 12.0 + phase[1])
              + 0.5 * amp * np.cos(2 * np.pi * t[:, None] / 8.0 + phase[1][::-1]))

    # R2 "drifting volatile": a very slow carrier plus a persistent AR(1). rho=0.95 makes
    # the AR term predictable a few steps out but not 96, so the irreducible error grows
    # with the horizon -- the per-horizon structure the calibrators bin on.
    innov = rng.normal(0.0, 1.0, size=(n, n_channels))
    ar = np.empty((n, n_channels))
    ar[0] = innov[0]
    for i in range(1, n):
        ar[i] = AR_RHO * ar[i - 1] + np.sqrt(1 - AR_RHO ** 2) * innov[i]
    sig[2] = (0.8 * amp * np.sin(2 * np.pi * t[:, None] / 336.0 + phase[2])
              + AR_AMP * amp * ar)

    return sig, amp


def _crossfade_weights(path, n_regimes):
    """One-hot regime indicator, linearly ramped over CROSSFADE steps after each switch."""
    n = len(path)
    w = np.zeros((n, n_regimes))
    w[np.arange(n), path] = 1.0
    if CROSSFADE <= 1:
        return w
    kernel = np.ones(CROSSFADE) / CROSSFADE
    smoothed = np.empty_like(w)
    for k in range(n_regimes):
        # 'same'-mode box filter; the edges see a slightly light kernel, which the
        # renormalization below repairs.
        smoothed[:, k] = np.convolve(w[:, k], kernel, mode='same')
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def generate(n=20000, n_channels=7, seed=2026, start='2016-07-01 00:00:00', freq='h'):
    rng = np.random.default_rng(seed)

    path = _regime_path(n, 3, rng)
    sig, amp = _regime_signals(n, n_channels, rng)
    w = _crossfade_weights(path, 3)

    # Mixed noise-free signal, and the noise scale that the active regime implies. Both
    # are blended by the same weights, so sigma ramps across a switch exactly as the
    # signal does.
    clean = np.einsum('kr,rkc->kc', w, sig)
    scale = np.array([NOISE_SCALE[r] for r in range(3)])
    sigma = (w @ scale)[:, None] * amp[None, :]

    values = clean + rng.normal(0.0, 1.0, size=(n, n_channels)) * sigma

    dates = pd.date_range(start=start, periods=n, freq=freq)
    cols = [f'ch{i}' for i in range(n_channels - 1)] + ['OT']
    df = pd.DataFrame(values, columns=cols)
    df.insert(0, 'date', dates)
    return df, path, sigma, clean, amp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=20000, help='number of time steps')
    p.add_argument('--n_channels', type=int, default=7,
                   help='channels including the OT target (7 matches ETT, so enc_in/dec_in/c_out defaults apply)')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--out_dir', type=str, default='./data/long_term_forecast/synthetic/')
    p.add_argument('--name', type=str, default='synth_moe3')
    args = p.parse_args()

    df, path, sigma, clean, amp = generate(n=args.n, n_channels=args.n_channels, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f'{args.name}.csv')
    df.to_csv(csv_path, index=False)

    npz_path = os.path.join(args.out_dir, f'{args.name}_latent.npz')
    np.savez_compressed(npz_path, regime=path, sigma=sigma, clean=clean, amp=amp,
                        ar_rho=AR_RHO, ar_amp=AR_AMP,
                        noise_scale=np.array([NOISE_SCALE[k] for k in range(3)]))

    n_switch = int((np.diff(path) != 0).sum())
    print(f'wrote {csv_path}  shape={df.shape}')
    print(f'wrote {npz_path}  (latent regime path, per-step sigma, noise-free signal)')
    print(f'regime occupancy: ' + ', '.join(
        f'R{k}={np.mean(path == k):.3f}' for k in range(3)))
    print(f'switches: {n_switch} (mean dwell {len(path) / (n_switch + 1):.0f} steps)')
    print(f'per-regime sigma (channel-mean): ' + ', '.join(
        f'R{k}={sigma[path == k].mean():.3f}' for k in range(3)))
    print(f'per-regime signal std (channel-mean): ' + ', '.join(
        f'R{k}={clean[path == k].std():.3f}' for k in range(3)))


if __name__ == '__main__':
    main()
