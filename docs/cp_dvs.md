# CP-DVS: Conformal Prediction via Decomposed Variance Scaling

Implementation: [`calibration/cp_dvs_calibration.py`](../calibration/cp_dvs_calibration.py)
Benchmark: [`scripts/benchmark_mog_calibration.py`](../scripts/benchmark_mog_calibration.py)

## Setting

A MoG forecast head emits, per input `x` and per (horizon step `h`, channel `c`), a
mixture `{pi_k(x), mu_k(x), sigma_k^2(x)}_{k=1..K}`. In this repo that is exactly the
tuple `(expert_weights, expert_out, expert_unc)` returned by `models/MoE.py`. The law of
total variance splits the predictive variance in two:

```
mu_bar(x)          = sum_k pi_k mu_k
sigma^2_within(x)  = sum_k pi_k sigma_k^2                    <- aleatoric
sigma^2_between(x) = sum_k pi_k (mu_k - mu_bar)^2            <- expert disagreement
sigma^2_total(x)   = sigma^2_within(x) + sigma^2_between(x)
```

`Exp_Long_Term_Forecast.calc_aleatoric_epistermic_uncertainty` already computes these
three quantities; CP-DVS consumes them directly.

## The gap CP-DVS targets

CP-VS fixes the non-conformity score to `|y - mu_bar| / sigma_total`. That is width-optimal
only if `sigma_total` is proportional to the conditional `(1-alpha)` quantile of the
absolute residual. Two reasons it is not, on MoG time-series forecasts:

1. **Wrong functional form.** `sigma_total` is a conditional *standard deviation*. The
   quantity that sets the oracle half-width is a conditional *quantile*. Their ratio is
   constant only if the standardized residual distribution is the same at every `x`; when
   `sigma` is a noisy scale estimate the map from `sigma` to the right half-width is
   systematically flatter than linear, and no single scalar `q_hat` can fix a shape error.
2. **Wrong mixing of the components.** Summing within and between with weight 1:1 is what
   the law of total variance prescribes for the *variance*, not for the residual quantile.
   Expert disagreement and aleatoric noise carry different amounts of information about
   `|y - mu_bar|`, and their optimal relative weight is a property of the fitted model, not
   a constant.

## Method

Score against a *learned* scale field, log-linear in the decomposition:

```
log u(x) = b0 + b1*l(x) + b2*l(x)^2 + b3*d(x) + b4*rho(x)

l(x)   = log sigma_total(x)                          (centered on the fit split)
d(x)   = log sigma_within(x) - log sigma_total(x)    in (-inf, 0]
rho(x) = sigma^2_between(x) / sigma^2_total(x)       in [0, 1]
```

`b` is fit by **quantile regression at the target level** `tau = 1 - alpha` on
`log|y - mu_bar|`. Because `log` is monotone, the `tau`-quantile of `log|r|` exponentiates
to the `tau`-quantile of `|r|`, so `exp(z'b)` estimates the conditional `(1-alpha)`
quantile of the absolute residual — precisely the oracle half-width. The pinball objective
is convex in `b`, so this is a well-posed convex fit, not a grid search over exponents.

Split conformal then restores finite-sample validity on a **disjoint** block:

```
s_i   = |y_i - mu_bar_i| / u(x_i)
q_hat = Quantile_{ceil((n+1)(1-alpha))/n}({s_i})     per (horizon step, channel)
C(x)  = mu_bar(x) +/- q_hat * u(x)
```

Coverage therefore does not depend on the scale model being correct; only the width does.

## Relationship to the existing calibrators

Two baselines are exact points of the `b` family:

| `b` | `u(x)` | method |
|---|---|---|
| `(c, 0, 0, 0, 0)` | `const` | Standard CP |
| `(c, 1, 0, 0, 0)` | `sigma_total` | CP-VS |

`_fit_scale_model` evaluates both closed forms explicitly, runs the optimizer from the
better of the two, and returns whichever of the three has the lowest pinball loss. So
**CP-DVS is never worse than either baseline on the fitting objective, by construction**.
The gap between that and "narrower on test" is generalization, which the benchmark
measures rather than assumes.

Against the other MoG calibrators in this repo:

- `AleatoricOnlyCalibrator` is CP-VS on `sigma_within` (the squared-residual/variance-ratio
  form is algebraically the same estimator). CP-DVS contains it as `b = (c, 1, 0, 1, 0)`.
- The HPD family (`MoG_HPD`, `STA_HPD`, `SETA_HPD`, ...) also opens the shape of the
  `sigma -> width` map, but by grid-searching 2-4 exponents against a tune-fold width
  objective. CP-DVS differs in that the scale is fit by a convex program directly against
  the pinball loss *at the target quantile level*, and that it is defined for the interval
  (not level-set) geometry the time-series pipeline uses.

## Protocol notes

- Conformal quantiles are taken per `(h, c)` (axis 0), matching `AdaptiveCPVS` and the
  other time-series calibrators. A per-`(h, c)` quantile absorbs any factor of `u` constant
  within a cell, so the scale model only has to explain *within-cell* heteroscedasticity.
- `fit_frac=0.5` splits the calibration block in time order: the leading half fits `b`, the
  trailing half (closer to the test period) supplies `q_hat`. The two must be disjoint or
  `q_hat` is optimistically biased.
- CP-DVS is *static* split conformal, unlike the online delayed-update protocol used by
  `calibrate_cpvs`, because the scale fit needs a held-out block.
- Exchangeability does not hold for time series, so — as for every calibrator here —
  coverage is empirical, not guaranteed. The benchmark reports it directly.

## Usage

```bash
python run.py ... --prob_expert --num_experts 3 --do_cp_dvs_calibration --cp_dvs_alpha 0.1
```

Standalone:

```python
from calibration.cp_dvs_calibration import CPDVSCalibrator, MoGPrediction

cal = MoGPrediction(pi_val, mu_val, var_val)        # [N, K, H, C]
c = CPDVSCalibrator(alpha=0.1).calibrate(cal, y_val)
lower, upper = c.predict_intervals(MoGPrediction(pi_test, mu_test, var_test))
```

`MoGPrediction.from_decomposition(mu_bar, v_within, v_between)` avoids materializing the
per-expert tensors when only the decomposition is available.
