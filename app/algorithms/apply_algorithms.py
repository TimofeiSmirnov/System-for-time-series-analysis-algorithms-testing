from app.algorithms.damp.damp import damp_algorithm
from app.algorithms.dumb_univariate_algorithm.dumb import dumb_algorithm
from app.algorithms.multidim_ad.multidimensional_ad import multidimensional_ad_post_sorting, \
    multidimensional_ad_pre_sorting, multidimensional_ad_with_kdps
import numpy as np


def dumb(time_series, threshold, m):
    return dumb_algorithm(time_series, threshold, m)


def damp(time_series, threshold, m):
    return damp_algorithm(np.array(time_series), threshold)


def post_sorting(time_series, threshold, m):
    multidimensional_ad_with_kdps(time_series, threshold, m)
    return multidimensional_ad_post_sorting(time_series, threshold, m)


def pre_sorting(time_series, threshold, m):
    return multidimensional_ad_pre_sorting(time_series, threshold, m)


# def anomaly_detection_with_kdps(time_series, threshold, m):
#     return multidimensional_ad_with_kdps(time_series, threshold, m)


algorithms_1d = {"dumb": dumb, "damp": damp}

algorithms_nd = {"post_sorting": post_sorting, "pre_sorting": pre_sorting}


def apply_algorithm_1d(algorithm_type, time_series, threshold, m):
    return algorithms_1d[algorithm_type](time_series, threshold, m)


def apply_algorithm_nd(algorithm_type, time_series, threshold, m):
    return algorithms_nd[algorithm_type](time_series, threshold, m)

