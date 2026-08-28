# MoG conformal calibration on ETT

Split-conformal protocol, identical for every method: fit on the validation split, evaluate on the test split, conformal quantile taken per (horizon step, channel). CP-DVS additionally splits validation 50/50 into scale-model fit and conformal quantile, so its quantile sees **half** the calibration data the baselines get.

`cpvs_within` and `aleatoric_only` are algebraically the same estimator (variance-ratio CP on the within-component); both are listed so the equivalence is visible in the numbers.

Time series are not exchangeable, so coverage here is empirical, not guaranteed -- for every method alike.

Aggregated over 48 (dataset, horizon, seed) runs.

## alpha = 0.05  (target coverage 0.95)

### By dataset

| dataset | metric | standard_cp | cpvs | cpvs_within | aleatoric_only | cp_dvs |
|---|---|---|---|---|---|---|
| ETTh1 | coverage | 0.9715 | 0.9875 | 0.9877 | 0.9877 | 0.9701 |
| ETTh1 | avg width | 4.2904 | 4.3331 | 4.3469 | 4.3469 | 3.8075 |
| ETTh2 | coverage | 0.9307 | 0.9301 | 0.9301 | 0.9301 | 0.9355 |
| ETTh2 | avg width | 2.2628 | 2.4318 | 2.4354 | 2.4354 | 2.4108 |
| ETTm1 | coverage | 0.9710 | 0.9812 | 0.9812 | 0.9812 | 0.9623 |
| ETTm1 | avg width | 3.2720 | 3.3275 | 3.3333 | 3.3333 | 2.8633 |
| ETTm2 | coverage | 0.9188 | 0.9229 | 0.9229 | 0.9229 | 0.9323 |
| ETTm2 | avg width | 1.7150 | 1.9121 | 1.9142 | 1.9142 | 1.8962 |

### By horizon (avg width)

| pred_len | standard_cp | cpvs | cpvs_within | aleatoric_only | cp_dvs | CP-DVS vs CP-VS |
|---|---|---|---|---|---|---|
| 96 | 2.2765 | 2.3061 | 2.3116 | 2.3116 | 2.1673 | -6.02% |
| 192 | 2.7077 | 2.8053 | 2.8094 | 2.8094 | 2.5977 | -7.40% |
| 336 | 3.0539 | 3.1587 | 3.1640 | 3.1640 | 2.8974 | -8.27% |
| 720 | 3.5020 | 3.7344 | 3.7448 | 3.7448 | 3.3153 | -11.22% |

### Overall -- headline

| method | coverage | avg width | width vs CP-VS | interval score (norm) | IS vs CP-VS |
|---|---|---|---|---|---|
| standard_cp | 0.9480 | 2.8850 | -3.87% | 4.3052 | +3.00% |
| cpvs | 0.9554 | 3.0011 | +0.00% | 4.1797 | +0.00% |
| cpvs_within | 0.9555 | 3.0074 | +0.21% | 4.1863 | +0.16% |
| aleatoric_only | 0.9555 | 3.0074 | +0.21% | 4.1863 | +0.16% |
| cp_dvs | 0.9501 | 2.7444 | -8.55% | 3.9919 | -4.49% |

### Overall -- conditional coverage and robustness

(nominal = target coverage; closer to nominal is better for the coverage columns, lower is better for gaps and clustering)

| method | SSC min cov | SSC max gap | worst horizon cov | horizon cov MAE | worst channel cov | min block cov | violation clustering | width CV |
|---|---|---|---|---|---|---|---|---|
| standard_cp | 0.9168 | 0.0505 | 0.9348 | 0.0243 | 0.8945 | 0.8945 | 18.810 | 0.4225 |
| cpvs | 0.9397 | 0.0460 | 0.9393 | 0.0296 | 0.9234 | 0.9013 | 29.318 | 0.5992 |
| cpvs_within | 0.9398 | 0.0461 | 0.9393 | 0.0297 | 0.9235 | 0.9013 | 29.587 | 0.6003 |
| aleatoric_only | 0.9398 | 0.0461 | 0.9393 | 0.0297 | 0.9235 | 0.9013 | 29.587 | 0.6003 |
| cp_dvs | 0.9274 | 0.0390 | 0.9306 | 0.0166 | 0.8934 | 0.8970 | 16.861 | 0.5006 |

CP-DVS narrower than CP-VS on **35/48** runs; CP-DVS held nominal coverage on **24/48** runs.

## alpha = 0.10  (target coverage 0.90)

### By dataset

| dataset | metric | standard_cp | cpvs | cpvs_within | aleatoric_only | cp_dvs |
|---|---|---|---|---|---|---|
| ETTh1 | coverage | 0.9502 | 0.9684 | 0.9688 | 0.9688 | 0.9408 |
| ETTh1 | avg width | 3.3833 | 3.3594 | 3.3704 | 3.3704 | 2.9409 |
| ETTh2 | coverage | 0.8825 | 0.8834 | 0.8832 | 0.8832 | 0.8869 |
| ETTh2 | avg width | 1.8059 | 1.9091 | 1.9114 | 1.9114 | 1.9656 |
| ETTm1 | coverage | 0.9409 | 0.9538 | 0.9539 | 0.9539 | 0.9223 |
| ETTm1 | avg width | 2.3877 | 2.4521 | 2.4557 | 2.4557 | 2.0759 |
| ETTm2 | coverage | 0.8738 | 0.8755 | 0.8756 | 0.8756 | 0.8856 |
| ETTm2 | avg width | 1.3425 | 1.4538 | 1.4552 | 1.4552 | 1.4692 |

### By horizon (avg width)

| pred_len | standard_cp | cpvs | cpvs_within | aleatoric_only | cp_dvs | CP-DVS vs CP-VS |
|---|---|---|---|---|---|---|
| 96 | 1.7058 | 1.7277 | 1.7310 | 1.7310 | 1.5900 | -7.97% |
| 192 | 2.0363 | 2.1011 | 2.1041 | 2.1041 | 1.9448 | -7.44% |
| 336 | 2.3682 | 2.4146 | 2.4184 | 2.4184 | 2.2432 | -7.10% |
| 720 | 2.8092 | 2.9311 | 2.9392 | 2.9392 | 2.6737 | -8.78% |

### Overall -- headline

| method | coverage | avg width | width vs CP-VS | interval score (norm) | IS vs CP-VS |
|---|---|---|---|---|---|
| standard_cp | 0.9118 | 2.2299 | -2.78% | 3.4452 | +4.68% |
| cpvs | 0.9203 | 2.2936 | +0.00% | 3.2913 | +0.00% |
| cpvs_within | 0.9204 | 2.2982 | +0.20% | 3.2957 | +0.13% |
| aleatoric_only | 0.9204 | 2.2982 | +0.20% | 3.2957 | +0.13% |
| cp_dvs | 0.9089 | 2.1129 | -7.88% | 3.1994 | -2.79% |

### Overall -- conditional coverage and robustness

(nominal = target coverage; closer to nominal is better for the coverage columns, lower is better for gaps and clustering)

| method | SSC min cov | SSC max gap | worst horizon cov | horizon cov MAE | worst channel cov | min block cov | violation clustering | width CV |
|---|---|---|---|---|---|---|---|---|
| standard_cp | 0.8702 | 0.0741 | 0.8896 | 0.0363 | 0.8440 | 0.8327 | 10.478 | 0.4195 |
| cpvs | 0.8969 | 0.0660 | 0.8919 | 0.0427 | 0.8744 | 0.8440 | 12.827 | 0.5931 |
| cpvs_within | 0.8970 | 0.0661 | 0.8920 | 0.0429 | 0.8746 | 0.8441 | 12.915 | 0.5941 |
| aleatoric_only | 0.8970 | 0.0661 | 0.8920 | 0.0429 | 0.8746 | 0.8441 | 12.915 | 0.5941 |
| cp_dvs | 0.8776 | 0.0583 | 0.8776 | 0.0241 | 0.8228 | 0.8310 | 8.942 | 0.5168 |

CP-DVS narrower than CP-VS on **32/48** runs; CP-DVS held nominal coverage on **27/48** runs.

