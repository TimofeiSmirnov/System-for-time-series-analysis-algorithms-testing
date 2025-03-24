import stumpy
import pandas as pd
import statistics
import numpy as np


def dumb_algorithm(time_series, threshold_number, m):
    matrix_profile = stumpy.stump(time_series, m=m)
    mp_data = matrix_profile[:, 0]
    threshold = np.percentile(mp_data, threshold_number)
    above_threshold = mp_data > threshold
    anomaly_indices = np.where(above_threshold)[0]

    return anomaly_indices
