import stumpy
from scipy.stats import zscore
from scipy.spatial.distance import euclidean
import numpy as np


def z_normalized_euclidean_distance(first_list, second_list):
    """Z-нормализованное Евклидово расстояние"""
    first_list_z = zscore(first_list)
    second_list_z = zscore(second_list)
    return euclidean(first_list_z, second_list_z)


def multidimensional_ad(time_series, threshold):
    """Детектинг аномалий в многомерных временных рядах"""
    number_of_dimensions = len(time_series)
    list_of_matrix_profiles = []

    for d in range(number_of_dimensions):
        matrix_profile = stumpy.stump(time_series[d], 50)[:, 0]
        list_of_matrix_profiles.append(matrix_profile)

    sorted_list_of_matrix_profiles = sorted(list_of_matrix_profiles, key=lambda profile: np.min(profile))
    threshold = np.percentile(sorted_list_of_matrix_profiles[0], threshold)
    above_threshold = sorted_list_of_matrix_profiles[0] > threshold
    anomaly_indices = np.where(above_threshold)[0]

    return anomaly_indices, sorted_list_of_matrix_profiles
