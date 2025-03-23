from app.algorithms.damp.damp import damp_algorithm
from app.algorithms.dumb_univariate_algorithm.dumb import dumb_algorithm
from app.algorithms.multidim_ad.multidimensional_ad import multidimensional_ad
import numpy as np


def dumb(time_series, threshold):
    return dumb_algorithm(time_series, threshold)


def damp(time_series, threshold):
    return damp_algorithm(np.array(time_series), threshold)


def post_sorting(time_series, threshold):
    return multidimensional_ad(time_series, threshold)


algorithms_1d = {"dumb": dumb, "damp": damp}

algorithms_nd = {"post_sorting": post_sorting}


def apply_algorithm_1d(algorithm_type, time_series, threshold):
    return algorithms_1d[algorithm_type](time_series, threshold)


def apply_algorithm_nd(algorithm_type, time_series, threshold):
    return algorithms_nd[algorithm_type](time_series, threshold)

