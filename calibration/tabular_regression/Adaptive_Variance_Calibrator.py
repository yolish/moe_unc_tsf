import numpy as np
import torch

class AdaptiveVarianceCalibrator:
    def __init__(self, alpha=0.1, r_grid=None, nested_tuning=True,
                 tune_frac=0.5, min_nested_n=200, split_seed=0, n_repeats=20):
        self.alpha = alpha
        self.r_grid = r_grid if r_grid is not None else \
            np.unique(np.concatenate(([0.0], np.geomspace(1e-2, 1e2, 25))))
        self.nested_tuning = nested_tuning
        self.tune_frac = tune_frac
        self.min_nested_n = min_nested_n
        self.split_seed = split_seed
        self.n_repeats = n_repeats

        self.r = None
        self.q_sq = None
        self.n_tune_ = None
        self.n_calib_ = None
        self.n_repeats_ = None
        self.tuning_widths_ = {}
        self.tuning_widths_std_ = {}

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
            # A single random tune/calib split makes the width-minimizing r sensitive to which
            # points happen to land in the tune fold (observed empirically: the same trained
            # checkpoint re-calibrated with a different split lands on materially different r).
            # Draw n_repeats independent splits and average the tune-fold width curve across
            # them before picking r* -- a cross-conformal-style repeated-splitting scheme (cf.
            # Vovk's cross-conformal predictors / Barber et al.'s CV+) that trades a bit of
            # compute for a much more stable r choice at a fixed model.
            n_tune = int(round(self.tune_frac * n_total))
            width_curves = np.empty((self.n_repeats, len(self.r_grid)))
            splits = []
            for b in range(self.n_repeats):
                rng = np.random.RandomState(self.split_seed + b)
                perm = rng.permutation(n_total)
                tune_idx, calib_idx = perm[:n_tune], perm[n_tune:]
                splits.append((tune_idx, calib_idx))
                for j, r in enumerate(self.r_grid):
                    q_sq_tune = self._q_sq_for(resid_sq[tune_idx], aleat_var[tune_idx],
                                                epist_var[tune_idx], r)
                    width_curves[b, j] = torch.sqrt(
                        q_sq_tune * aleat_var[tune_idx] + r * epist_var[tune_idx]
                    ).mean().item()

            avg_width_curve = width_curves.mean(axis=0)
            self.tuning_widths_ = {float(r): float(w) for r, w in zip(self.r_grid, avg_width_curve)}
            self.tuning_widths_std_ = {float(r): float(s)
                                        for r, s in zip(self.r_grid, width_curves.std(axis=0))}

            best_j, best_width = 0, avg_width_curve[0]
            for j in range(1, len(self.r_grid)):
                if avg_width_curve[j] < best_width - 1e-12:
                    best_j, best_width = j, avg_width_curve[j]
            self.r = float(self.r_grid[best_j])

            # q^2 at the now-fixed r*: median (not mean) of each repeat's calib-fold quantile,
            # more robust to any single fold's extreme quantile estimate (standard
            # cross-conformal aggregation).
            q_sq_repeats = [
                self._q_sq_for(resid_sq[calib_idx], aleat_var[calib_idx], epist_var[calib_idx], self.r)
                for _, calib_idx in splits
            ]
            self.q_sq = float(np.median(q_sq_repeats))
            self.n_tune_ = n_tune
            self.n_calib_ = n_total - n_tune
            self.n_repeats_ = self.n_repeats
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
            self.n_tune_ = None
            self.n_calib_ = None
            self.n_repeats_ = None

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
