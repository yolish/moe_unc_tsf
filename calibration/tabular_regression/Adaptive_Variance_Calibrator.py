import numpy as np
import torch

class AdaptiveVarianceCalibrator:
    def __init__(self, alpha=0.1, r_grid=None, nested_tuning=True,
                 tune_frac=0.5, min_nested_n=200, split_seed=0):
        self.alpha = alpha
        self.r_grid = r_grid if r_grid is not None else \
            np.unique(np.concatenate(([0.0], np.geomspace(1e-2, 1e2, 25))))
        self.nested_tuning = nested_tuning
        self.tune_frac = tune_frac
        self.min_nested_n = min_nested_n
        self.split_seed = split_seed

        self.r = None
        self.q_sq = None
        self.n_tune_ = None
        self.n_calib_ = None
        self.tuning_widths_ = {}

    def _q_level(self, n):
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        return min(max(q_level, 0.0), 1.0)

    def _q_sq_for(self, resid_sq, aleat_var, epist_var, r):
        scores = torch.clamp((resid_sq - r * epist_var) / aleat_var, min=0.0).numpy()
        q_level = self._q_level(len(scores))
        return np.quantile(scores, q_level, method='higher')

    def fit(self, cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist):
        """
        Learn a variance-ratio r and scaling factor q^2 from the calibration set.
        """
        resid_sq = (cal_trues.squeeze() - cal_preds.squeeze()) ** 2
        aleat_var = cal_stds_aleat.squeeze() ** 2 + 1e-8
        epist_var = cal_stds_epist.squeeze() ** 2
        n_total = len(resid_sq)

        use_nested = self.nested_tuning and n_total >= self.min_nested_n
        if use_nested:
            rng = np.random.RandomState(self.split_seed)
            perm = rng.permutation(n_total)
            n_tune = int(round(self.tune_frac * n_total))
            tune_idx, calib_idx = perm[:n_tune], perm[n_tune:]
        else:
            tune_idx = calib_idx = np.arange(n_total)

        best_r, best_width = None, np.inf
        for r in self.r_grid:
            q_sq_tune = self._q_sq_for(resid_sq[tune_idx], aleat_var[tune_idx],
                                        epist_var[tune_idx], r)
            width_r = torch.sqrt(
                q_sq_tune * aleat_var[tune_idx] + r * epist_var[tune_idx]
            ).mean().item()
            self.tuning_widths_[float(r)] = width_r
            if width_r < best_width - 1e-12:
                best_width, best_r = width_r, r

        self.r = float(best_r)
        self.q_sq = float(self._q_sq_for(resid_sq[calib_idx], aleat_var[calib_idx],
                                          epist_var[calib_idx], self.r))
        self.n_tune_ = len(tune_idx) if use_nested else None
        self.n_calib_ = len(calib_idx) if use_nested else None

    def predict(self, test_preds, test_stds_aleat, test_stds_epist):
        """
        Construct prediction intervals using the learned ratio r and scaling factor q^2.
        """
        if self.q_sq is None:
            raise ValueError("Calibrator must be fitted before calling predict.")

        test_aleat_var = test_stds_aleat.squeeze() ** 2
        test_epist_var = test_stds_epist.squeeze() ** 2

        calibrated_std = torch.sqrt(
            self.q_sq * test_aleat_var + self.r * test_epist_var
        )

        lower_bound = test_preds.squeeze() - calibrated_std
        upper_bound = test_preds.squeeze() + calibrated_std

        intervals = torch.stack([lower_bound, upper_bound], dim=1)
        return intervals
