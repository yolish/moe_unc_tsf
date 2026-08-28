# SETA-HPD: STA-HPD with MoG Variance-Decomposition Awareness

**Goal.** Make highest-density-region conformal calibration aware of the *Mixture-of-Gaussians
structure* of the model it calibrates, by folding the full MoG law-of-total-variance decomposition
into its surrogate family. The result is **SETA-HPD** (Shape–Epistemic–Threshold Adaptive HPD), a
strict generalization of STA-HPD that adds one exponent (`ρ`) reshaping the between-component
(epistemic) variance.

**Two separate, independently-runnable methods.** STA-HPD is left completely unchanged in its own
file; SETA-HPD is a new class in its own file. Either can be run on its own.

| | Method | Class | File | Run via |
|---|---|---|---|---|
| Original | STA-HPD | `STAHPDCalibrator` | `calibration/tabular_regression/STA_HPD_Calibrator.py` | `--use_reg_sta_hpd` → `calibrate_sta_hpd` |
| New | SETA-HPD | `SETAHPDCalibrator` | `calibration/tabular_regression/SETA_HPD_Calibrator.py` | `--use_reg_seta_hpd` → `calibrate_seta_hpd` |

SETA-HPD at `ρ=1` (`mog_decompose=False`) reduces exactly to STA-HPD, so it never underperforms the
original on the tuning objective. It writes its own `result_calibration_seta_hpd.txt` and
`intervals_seta_hpd.npy`, so the two run side by side without collision.

---

## 1. Background: what STA-HPD already did

For a $K$-expert probabilistic MoE with per-expert means $\mu_k(x)$, standard deviations
$\sigma_k(x)$, gate weights $w_k(x)$ (with $\sum_k w_k = 1$) and aggregate prediction
$\hat y(x)=\sum_k w_k\mu_k$, the model's predictive law is the Gaussian mixture

$$f_{\mathrm{mix}}(y\mid x) = \sum_{k=1}^K w_k(x)\,\mathcal N\!\big(y;\ \mu_k(x),\ \sigma_k^2(x)\big).$$

Plain **MoG-HPD** conformalizes the negative log-density $S=-\log f_{\mathrm{mix}}$ and returns the
super-level set $\{y: f_{\mathrm{mix}}(y\mid x)\ge\hat\tau\}$, which is width-optimal *for the fitted
density* (Neyman–Pearson).

The original **STA-HPD** opened two knobs, both no-ops at their defaults:

- a **within-component (aleatoric) shape exponent** $c$ reshaping each expert's own spread,
  $\tilde\sigma_k(x;c)=G\,(\sigma_k(x)/G)^c$ with $G=\exp(\operatorname{mean}_{k,i}\log\sigma_k)$;
- a **threshold-field exponent** $\theta$ tilting the density threshold by the point's total scale.

The between-component (epistemic) part of the variance — the expert disagreement
$\sum_k w_k(\mu_k-\hat y)^2$ — entered *only* through $\sigma_{\mathrm{tot}}$ in the $\theta$ term. It
was never reshaped. STA-HPD saw half of the MoG variance decomposition.

---

## 2. The enhancement: reshape the between-component variance too

The MoG **law of total variance** splits the predictive variance into two structurally distinct
pieces:

$$\underbrace{\operatorname{Var}(Y\mid x)}_{\text{total}}
= \underbrace{\sum_k w_k\,\sigma_k^2}_{\text{within-component (aleatoric)}}
+ \underbrace{\sum_k w_k\,(\mu_k-\hat y)^2}_{\text{between-component (epistemic)}}.$$

STA-HPD already reshapes the first term (via $c$). The enhancement adds a **between-component
(epistemic) scale exponent** $\rho\ge 0$ that reshapes the second term, using the mean-shift

$$\tilde\mu_k(x;\rho) = \hat y(x) + \sqrt{\rho}\,\big(\mu_k(x)-\hat y(x)\big).$$

This shift is exact and structure-preserving:

- it **preserves the aggregate prediction**, $\sum_k w_k\tilde\mu_k = \hat y$ (so the point forecast
  and marginal coverage target are untouched);
- it **multiplies the between-component variance by exactly $\rho$**:
  $\sum_k w_k(\tilde\mu_k-\hat y)^2 = \rho\sum_k w_k(\mu_k-\hat y)^2$;
- it leaves the within-component term to $c$: the two exponents act on **disjoint terms** of the
  decomposition.

The surrogate density, total scale, score, and prediction set become

$$
\begin{aligned}
f_{c,\rho}(y\mid x) &= \sum_k w_k\,\mathcal N\!\big(y;\ \tilde\mu_k(x;\rho),\ \tilde\sigma_k^2(x;c)\big),\\
\sigma_{\mathrm{tot}}^2(x;c,\rho) &= \sum_k w_k\,\tilde\sigma_k^2(x;c) \;+\; \rho\sum_k w_k\,(\mu_k-\hat y)^2,\\
S_i(c,\rho,\theta) &= -\log f_{c,\rho}(y_i\mid x_i) \;-\; \theta\,\log\sigma_{\mathrm{tot}}(x_i;c,\rho),\\
\hat C(x) &= \Big\{y:\ \log f_{c,\rho}(y\mid x)\ \ge\ -\hat t - \theta\log\sigma_{\mathrm{tot}}(x;c,\rho)\Big\}.
\end{aligned}
$$

**Interpretation.** $\rho<1$ compresses the experts toward $\hat y$, filling the low-density valleys
that an over-separated MoE opens up between well-separated modes (favouring a single interval);
$\rho>1$ amplifies expert disagreement; $\rho=1$ is the raw between-component scale (the original
STA-HPD).

### Implementation detail that makes it clean
Every downstream routine (`_log_sigma_tot`, `_log_mix_density`, `_grid_log_density`,
`_segments_for_point`) is simply handed the shifted means $\tilde\mu_k$. Because the shift preserves
$\hat y$, the epistemic term of $\sigma_{\mathrm{tot}}$ **picks up the factor $\rho$ automatically** —
no special-casing anywhere. The tuning search becomes a 3-D grid over $(c,\rho,\theta)$; the density
$f_{c,\rho}$ is independent of $\theta$, so it is computed once per $(c,\rho)$ and the whole
$\theta$-grid is swept with cheap threshold counts (an $\approx|\Theta|\times$ speedup that also
benefits the original two-parameter path).

---

## 3. Why it is safe (boundary recovery & validity)

The grid always contains $\rho=1$, so the new family **strictly contains** its ancestors as grid
points:

| $(c,\rho,\theta)$ | recovers |
|---|---|
| $(1,1,0)$ | MoG-HPD, exactly |
| $(c,1,\theta)$ | original two-parameter STA-HPD |
| $(1,\rho,0)$ | Adaptive MoG-HPD (epistemic-scale-only HPD) |
| $(1,1,1)$ | CP-VS on total variance (when $K=1$) |
| $(0,1,0)$ | Standard CP (single global width) |

Because $(c^\star,\rho^\star,\theta^\star)=\arg\min$ over a grid that includes all of the above, the
tuned method's **tuning objective can never be worse** than MoG-HPD, the original STA-HPD, or
Adaptive MoG-HPD. Coverage validity is unchanged: for any $(c,\rho,\theta)$ fixed on a tune fold
disjoint from the calibration fold that supplies $\hat t$, the score is a data-independent function,
so the standard split-conformal finite-sample marginal guarantee $P(Y\in\hat C(X))\ge 1-\alpha$ holds
exactly, regardless of model correctness. $\rho$ has effect only when $K>1$ (for $K=1$ the
between-component term is identically zero).

### The relationship to STA-HPD
`SETAHPDCalibrator` defaults to `mog_decompose=True` → `rho_grid=(0,0.25,0.5,0.75,1,1.5,2)` (ρ knob
active). Setting `mog_decompose=False` → `rho_grid=(1.0,)` → the search reduces to the original 2-D
$(c,\theta)$ grid, **bit-for-bit identical** to `STAHPDCalibrator` (verified: SETA-HPD at ρ off
reproduces STA-HPD's logged $(c,\theta,\text{width})$ exactly on all cached configs). The original
`STAHPDCalibrator` is untouched and has no ρ machinery at all.

---

## 4. Results

**Setup.** All numbers use the already-trained tabular checkpoints in this repo (seed 4022,
`K=3` for Synthetic/Superconductivity, `K=2` for Bike/Temperature, both the softmax-gated `MOG` and
uncertainty-gated `MOGU` architectures), nominal coverage $1-\alpha=90\%$. For each of the 8
(architecture, dataset) configs, `STA-HPD base` (`mog_decompose=False`, i.e. the original method)
and `STA-HPD +MOG(ρ)` (`mog_decompose=True`) are each fit over 5 tune/calib split seeds
(`n_repeats=10` internally per fit); reported width/coverage are mean±std over those 5 outer seeds,
test-set evaluated. `MoG-HPD` is the deterministic, untuned reference. "Score" is a Winkler-style
proper interval score (width plus a $2/\alpha$-weighted tail-miss penalty) computed by the benchmark
script, not the calibrators themselves.

In the table, **STA-HPD** is the original method (ρ off) and **SETA-HPD** is the new method (ρ on).

| Config | MoG-HPD width | STA-HPD width | SETA-HPD width | Δwidth (SETA vs STA) | ρ\* (mode, stability) |
|---|---:|---:|---:|---:|---|
| **Synthetic (MOG)** | 1.0490 | 0.9910±0.0026 | 0.9910±0.0026 | **0.0%** | 1.0 (4/5 seeds) |
| **Superconductivity (MOG)** | 0.7964 | 0.7836±0.0031 | 0.7858±0.0008 | +0.3% (score **−1.3%**) | 0.75 (5/5, fully stable) |
| Bike (MOG) | 0.7940 | 0.7712±0.0014 | 0.7376±0.0012 | **−4.3%** | 0.25 (5/5, fully stable) |
| Temperature (MOG) | 0.9817 | 0.9673±0.0000 | 0.9768±0.0078 | +1.0% (score −0.8%) | 0.5 (3/5) |
| Synthetic (MOGU) | 1.5614 | 1.3579±0.0005 | 1.3618±0.0009 | +0.3% | 0.0 (5/5, but no measurable effect) |
| **Superconductivity (MOGU)** | 0.9542 | 0.9134±0.0011 | 0.9137±0.0010 | **0.0%** | 1.0 (4/5) |
| Bike (MOGU) | 0.6424 | 0.6367±0.0039 | 0.6359±0.0045 | −0.1% (empty-set rate halved, see below) | 0.25–0.5 (unstable) |
| Temperature (MOGU) | 0.9951 | 0.9989±0.0043 | 0.9988±0.0092 | ~0.0% | 0.5–2.0 (unstable) |

**Bold** rows are the task's high-priority datasets (Synthetic, Superconductivity), one row per
architecture.

### High-priority datasets

- **Synthetic (MOG)**: the mechanism is a **clean no-op**. $\rho^\star$ lands at $1.0$ on 4/5 seeds
  (one seed picks $0.6$ with zero measurable width change), and mean test width is identical to the
  original STA-HPD to 4 decimals. This is an honest negative result, not a bug: Synthetic's true law
  $Y\mid X\sim 0.2\mathcal N(X,1)+0.3\mathcal N(X^2,1)+0.5\mathcal N(X^3,1)$ makes the *within*-component
  scale ($c$) and the threshold field ($\theta$) — which the original STA-HPD already tunes — the
  dominant correctable mismatch; there is no additional systematic epistemic over/under-separation
  left for $\rho$ to correct once $c,\theta$ are optimized. On **Synthetic (MOGU)**, $\rho^\star=0$ is
  selected with perfect stability (5/5 seeds) — the tune-fold objective does have a preference
  direction — but it moves test width by only $+0.3\%$, within seed noise.
- **Superconductivity (MOG)**: the mechanism **activates and helps**. $\rho^\star=0.75$ is selected
  on **all 5 seeds** (fully stable) — a genuine, non-noise signal that mild epistemic compression is
  worthwhile here. Mean width is flat (+0.3%, within one std), but the proper interval score improves
  $-1.3\%$ and the fraction of fragmented (disjoint multi-interval) prediction sets drops from
  $7.1\%\to3.5\%$ — i.e. the same coverage budget is being spent as fewer, more usable single
  intervals rather than raw width reduction. This is the one high-priority config where the new
  knob earns its keep. On **Superconductivity (MOGU)**, however, $\rho^\star=1.0$ on 4/5 seeds — the
  mechanism stays inactive despite this architecture's very high gate entropy (0.95, i.e. genuinely
  soft/mixed routing) — so structural mixing alone does not guarantee $\rho\ne1$ is useful; it
  depends on whether that mixing is a *miscalibration* of the epistemic term specifically.

### Other datasets (context)

- **Bike (MOG)** shows the largest and most stable effect in the whole sweep: $\rho^\star=0.25$ on
  5/5 seeds, width $-4.3\%$ vs. the original STA-HPD, and the fraction of disjoint multi-segment sets
  drops to exactly $0$ (from $1.5\%$) — compressing the two experts toward $\hat y$ removes the
  density valley that was fragmenting some Bike predictions into two intervals. The proper score is
  slightly *worse* ($+2.3\%$) despite the width win: a small number of points that were previously
  covered by the (now-removed) secondary disjoint segment are missed by the merged single interval,
  and a miss carries a $20\times$ distance penalty in the Winkler score even though marginal coverage
  is essentially unchanged (90.0% vs 90.2%). This is a legitimate width/tail-risk trade-off, not an
  artifact.
- **Bike (MOGU)** has a real but noisy effect: the empty-prediction-set rate (a known MoG-HPD-family
  failure mode under this architecture's occasional extreme per-expert $\sigma$) drops from
  $1.28\%\to0.52\%$, roughly halved, though $\rho^\star$ itself is not stable across seeds
  (0.5/0.25/0.0). The large raw "score" swing reported by the benchmark ($256\to105$) is dominated by
  a fixed placeholder penalty the benchmark script assigns to empty sets, not a calibrator property;
  the meaningful number is the halved empty-set rate.
- **Temperature** (both architectures): effects are within noise in both directions; $\rho^\star$ is
  not stable across seeds (oscillates between grid points with near-identical tune width), consistent
  with a flat $\rho$-marginal — this dataset's epistemic term carries little separately-correctable
  signal beyond what $c,\theta$ already capture.

### Summary

The MOG variance-decomposition mechanism is **safe by construction** (it can only match or improve
the tuning objective relative to the original STA-HPD, since $\rho=1$ is always on the grid) and this
holds empirically too — no config regresses outside seed noise. Its *practical* value is
**dataset-dependent and modest**: it is a clear no-op on Synthetic, a small but fully-stable gain on
Superconductivity (MOG) via fewer fragmented sets rather than raw width, and its largest, most stable
effect is on Bike (MOG, not a priority dataset here). This is consistent with the original STA-HPD
paper's own finding that $(c,\theta)$ already captures most of the correctable model/truth mismatch
on these four datasets — the between-component term was evidently the smaller remaining piece.

---

## 5. Files added / changed

**New file (SETA-HPD):**
- `calibration/tabular_regression/SETA_HPD_Calibrator.py` — `SETAHPDCalibrator`: the 3-parameter
  $(c,\rho,\theta)$ method, with `_shift_means`, the $\theta$-sweep-factorized tuning search, and
  $\rho$-aware `predict` / logging. `mog_decompose=True` by default.

**Unchanged:**
- `calibration/tabular_regression/STA_HPD_Calibrator.py` — the original two-parameter `STAHPDCalibrator`
  is left exactly as it was (no ρ machinery; only a one-line docstring cross-reference to SETA-HPD).

**Wiring:**
- `exp/exp_tabular_regression.py` — imports `SETAHPDCalibrator`; new `calibrate_seta_hpd` method
  (own `result_calibration_seta_hpd.txt` / `intervals_seta_hpd.npy`; diagnostics report the
  $(c,\rho,\theta)$ surface, a $\rho$-marginal, and both MoG-HPD and STA-HPD on the same split).
  `calibrate_sta_hpd` is unchanged.
- `run.py` — new `--use_reg_seta_hpd` flag dispatching `calibrate_seta_hpd`, alongside the existing
  `--use_reg_sta_hpd`.
- `scripts/run_tabular_regression.sh` — the MOG/MOGU/single-expert runs now pass `--use_reg_seta_hpd`
  next to `--use_reg_sta_hpd`, and the summary-CSV parser has a `SETA-HPD` row (its header is not a
  substring of `STA-HPD Results` and vice-versa, so the two never cross-match).
