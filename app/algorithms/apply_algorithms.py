from app.algorithms.damp.damp import damp_algorithm
from app.algorithms.dumb_univariate_algorithm.dumb import dumb_algorithm
from app.algorithms.multidim_ad.multidimensional_ad import multidimensional_ad_post_sorting, \
    multidimensional_ad_pre_sorting
import numpy as np


class ApplyAnomalyDetectionAlgorithms:
    def __init__(self):
        self.algorithms_1d = {"dumb": self._dumb, "damp": self._damp}
        self.algorithms_nd = {"post_sorting": self._post_sorting, "pre_sorting": self._pre_sorting}

    def _dumb(self, time_series, threshold, m):
        return dumb_algorithm(time_series, threshold, m)

    def _damp(self, time_series, threshold, m):
        return damp_algorithm(np.array(time_series), threshold)

    def _post_sorting(self, time_series, threshold, m):
        return multidimensional_ad_post_sorting(time_series, threshold, m)

    def _pre_sorting(self, time_series, threshold, m):
        return multidimensional_ad_pre_sorting(time_series, threshold, m)

    def apply_algorithm_1d(self, algorithm_type, time_series, threshold, m):
        return self.algorithms_1d[algorithm_type](time_series, threshold, m)

    def apply_algorithm_nd(self, algorithm_type, time_series, threshold, m):
        return self.algorithms_nd[algorithm_type](time_series, threshold, m)
