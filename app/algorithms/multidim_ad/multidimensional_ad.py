import stumpy
from scipy.stats import zscore
from scipy.spatial.distance import euclidean
import numpy as np


def z_normalized_euclidean_distance(first_list, second_list):
    """Z-нормализованное Евклидово расстояние"""
    first_list_z = zscore(first_list)
    second_list_z = zscore(second_list)
    return euclidean(first_list_z, second_list_z)


def multidimensional_ad_post_sorting(time_series, threshold, m):
    """Детектинг аномалий в многомерных временных рядах post sorting алгоритм"""
    number_of_dimensions = len(time_series)
    list_of_matrix_profiles = []

    for d in range(number_of_dimensions):
        matrix_profile = stumpy.stump(time_series[d], m)[:, 0]
        list_of_matrix_profiles.append(matrix_profile)

    sorted_list_of_matrix_profiles = sorted(list_of_matrix_profiles, key=lambda profile: np.min(profile))
    threshold = np.percentile(sorted_list_of_matrix_profiles[0], threshold)
    above_threshold = sorted_list_of_matrix_profiles[0] > threshold
    anomaly_indices = np.where(above_threshold)[0]

    return anomaly_indices, sorted_list_of_matrix_profiles


def multidimensional_ad_post_sorting_for_test(time_series, threshold, m):
    """Детектинг аномалий в многомерных временных рядах post sorting алгоритм"""
    number_of_dimensions = len(time_series)
    list_of_matrix_profiles = []

    for d in range(number_of_dimensions):
        matrix_profile = stumpy.stump(time_series[d], m)[:, 0]
        list_of_matrix_profiles.append(matrix_profile)

    sorted_list_of_matrix_profiles = sorted(list_of_matrix_profiles, key=lambda profile: np.min(profile))
    threshold = np.percentile(sorted_list_of_matrix_profiles[0], threshold)
    above_threshold = sorted_list_of_matrix_profiles[0] > threshold
    anomaly_indices = np.where(above_threshold)[0]

    return anomaly_indices


def multidimensional_ad_pre_sorting(time_series, threshold, m):
    """Детектинг аномалий в многомерных временных рядах pre sorting алгоритм"""
    number_of_dimensions = len(time_series)
    len_time_series = len(time_series[0])

    tensor_of_euclidian_distances = np.full((len_time_series - m + 1, len_time_series - m + 1, number_of_dimensions), np.inf)

    for d in range(number_of_dimensions):
        print(d)
        for i in range(len_time_series - m + 1):
            print("-", i)
            for j in range(len_time_series - m + 1):
                if i == j:
                    continue
                tensor_of_euclidian_distances[i, j, d] = z_normalized_euclidean_distance(
                    time_series[d][i:i + m], time_series[d][j:j + m]
                )

    matrix_profile_matrix = np.min(tensor_of_euclidian_distances, axis=2)
    threshold_value = np.percentile(matrix_profile_matrix, threshold)
    anomaly_indices = np.where(matrix_profile_matrix > threshold_value)

    return anomaly_indices, matrix_profile_matrix


def multidimensional_ad_pre_sorting_for_test(time_series, threshold, m):
    """Детектинг аномалий в многомерных временных рядах pre sorting алгоритм"""
    number_of_dimensions = len(time_series)
    len_time_series = len(time_series[0])

    tensor_of_euclidian_distances = np.full((len_time_series - m + 1, len_time_series - m + 1, number_of_dimensions), np.inf)

    for d in range(number_of_dimensions):
        print(d)
        for i in range(len_time_series - m + 1):
            print("-", i)
            for j in range(len_time_series - m + 1):
                if i == j:
                    continue
                tensor_of_euclidian_distances[i, j, d] = z_normalized_euclidean_distance(
                    time_series[d][i:i + m], time_series[d][j:j + m]
                )

    matrix_profile_matrix = np.min(tensor_of_euclidian_distances, axis=2)
    threshold_value = np.percentile(matrix_profile_matrix, threshold)
    anomaly_indices = np.where(matrix_profile_matrix > threshold_value)

    return anomaly_indices


# def multidimensional_ad_with_kdps(time_series, threshold, m):
#     """Вычисляет все матричные профили и сортирует их в убывающем порядке."""
#
#     time_series_length = len(time_series)
#     matrix_profiles = []
#     for i in range(time_series_length):
#         matrix_profiles.append([])
#
#     for i, ts in enumerate(matrix_profiles):
#         matrix_profiles[i] = stumpy.stump(time_series[i], m)[:, 0]
#
#     list_of_matrix_profiles = np.array(matrix_profiles)
#     KDP_idx = np.argsort(list_of_matrix_profiles, axis=1)[:, ::-1]
#     KDPs = np.take_along_axis(list_of_matrix_profiles, KDP_idx, axis=1)
#
#     threshold = np.percentile(KDPs[0], threshold)
#     above_threshold = KDPs[0] > threshold
#     anomaly_indices = np.where(above_threshold)[0]
#
#     return anomaly_indices, KDPs

