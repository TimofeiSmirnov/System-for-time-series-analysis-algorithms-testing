from app.algorithms.damp.damp import damp_algorithm
from app.algorithms.classic.classic import classic_algorithm
from app.algorithms.multidim_ad.multidimensional_ad import multidimensional_ad_post_sorting, \
    multidimensional_ad_pre_sorting, multidimensional_ad_mstump
import numpy as np


class ApplyAnomalyDetectionAlgorithms:
    def __init__(self):
        self.algorithms_1d = {"dumb": self._classic, "damp": self._damp}
        self.algorithms_nd = {"post_sorting": self._post_sorting, "pre_sorting": self._pre_sorting, "mstump": self._mstump}
        self.algorithms_for_1d_ad = ["dumb", "damp"]
        self.algorithms_for_nd_ad = ["pre_sorting", "post_sorting"]

    def _classic(self, time_series, threshold, m):
        return classic_algorithm(time_series, threshold, m)

    def _damp(self, time_series, threshold, args):
        return damp_algorithm(np.array(time_series), threshold, args)

    def _post_sorting(self, time_series, threshold, m):
        return multidimensional_ad_post_sorting(time_series, threshold, m)

    def _pre_sorting(self, time_series, threshold, m):
        return multidimensional_ad_pre_sorting(time_series, threshold, m)

    def _mstump(self, time_series, threshold, m):
        return multidimensional_ad_mstump(time_series, threshold, m)

    def apply_algorithm_1d(self, algorithm_type, time_series, threshold, m, args: [None, None]):
        if algorithm_type == "damp":
            return self.algorithms_1d[algorithm_type](time_series, threshold, args)
        return self.algorithms_1d[algorithm_type](time_series, threshold, m)

    def apply_algorithm_nd(self, algorithm_type, time_series, threshold, m):
        return self.algorithms_nd[algorithm_type](time_series, threshold, m)
