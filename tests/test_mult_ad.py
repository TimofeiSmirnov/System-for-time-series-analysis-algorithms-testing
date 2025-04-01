import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.algorithms.multidim_ad.multidimensional_ad import (
    z_normalized_euclidean_distance,
    multidimensional_ad_post_sorting_for_test,
    multidimensional_ad_mstump_for_test
)


def test_z_normalized_euclidean_distance():
    list1 = [1, 2, 3, 4, 5]
    list2 = [5, 4, 3, 2, 1]
    distance = z_normalized_euclidean_distance(list1, list2)
    assert isinstance(distance, float)
    assert distance >= 0


def test_multidimensional_ad_post_sorting():
    time_series = [np.random.rand(100).tolist() for _ in range(3)]
    threshold = 90
    window_length = 10

    anomaly_indices = multidimensional_ad_post_sorting_for_test(time_series, threshold, window_length)
    assert isinstance(anomaly_indices, np.ndarray)


def test_multidimensional_ad_mstump():
    time_series = [np.random.rand(100).tolist() for _ in range(3)]
    threshold = 90
    window_length = 10

    anomaly_indices = multidimensional_ad_mstump_for_test(time_series, threshold, window_length)
    assert isinstance(anomaly_indices, np.ndarray)
