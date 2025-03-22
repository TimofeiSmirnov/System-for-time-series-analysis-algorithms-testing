from app.algorithms.damp.damp import damp_algorithm
from app.algorithms.dumb_univariate_algorithm.dumb import dumb_algorithm
import numpy as np


def dumb(time_series, threshold):
    return dumb_algorithm(time_series, threshold)


def damp(time_series, threshold):
    dp = damp_algorithm(np.array(time_series), threshold)
    return dp


algorithms = {"dumb": dumb, "damp": damp}


def apply_algorithm(algorithm_type, time_series, threshold):
    return algorithms[algorithm_type](time_series, threshold)
