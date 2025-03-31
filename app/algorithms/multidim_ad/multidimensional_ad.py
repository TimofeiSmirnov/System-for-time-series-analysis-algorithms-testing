import stumpy
from scipy.stats import zscore
from scipy.spatial.distance import euclidean
import numpy as np


def z_normalized_euclidean_distance(first_list: list[float], second_list: list[float]) -> float:
    """
    Z-нормализованное Евклидово расстояние.
    :param first_list: (list[float]) первый массив
    :param second_list: (list[float]) второй массив
    :return:
        - (float) z-нормализованное Евклидово расстояние между двумя массивами
    """
    first_list_z = zscore(first_list)
    second_list_z = zscore(second_list)
    return euclidean(first_list_z, second_list_z)


def multidimensional_ad_post_sorting(
    time_series: list[list[float]],
    threshold: float,
    window_length: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Поиск аномалий в многомерных временных рядах. Post-sorting алгоритм.

    :param time_series: (list[list[float]])
    :param threshold: (float) пороговое значение для определения аномальности точки
    :param window_length: (int) длина окна алгоритма
    :return:
        - anomaly_indices (numpy.ndarray): индексы аномальных точек
        - sorted_list_of_matrix_profiles (list[numpy.ndarray]): список матричных профилей после
    """
    number_of_dimensions = len(time_series)
    list_of_matrix_profiles = []

    for d in range(number_of_dimensions):
        matrix_profile = stumpy.stump(np.array(time_series[d]), window_length)[:, 0]
        list_of_matrix_profiles.append(matrix_profile)

    sorted_list_of_matrix_profiles = sorted(list_of_matrix_profiles, key=lambda profile: np.min(profile))
    threshold = np.percentile(sorted_list_of_matrix_profiles[0], threshold)
    above_threshold = sorted_list_of_matrix_profiles[0] > threshold
    anomaly_indices = np.where(above_threshold)[0]

    return anomaly_indices, sorted_list_of_matrix_profiles


def multidimensional_ad_post_sorting_for_test(
    time_series: list[list[float]],
    threshold: float,
    window_length: int
) -> np.ndarray:
    """
    Поиск аномалий в многомерных временных рядах. Post-sorting алгоритм для тестирования

    :param time_series: (list[list[float]])
    :param threshold: (float) пороговое значение для определения аномальности точки
    :param window_length: (int) длина окна алгоритма
    :return:
        - anomaly_indices (numpy.ndarray): индексы аномальных точек
    """
    anomaly_indices, sorted_list_of_matrix_profiles = multidimensional_ad_post_sorting(time_series, threshold, window_length)
    return anomaly_indices


def multidimensional_ad_pre_sorting(
    time_series: list[list[float]],
    threshold: float,
    window_length: int
) -> tuple[np.array, np.ndarray]:
    """
    Поиск аномалий в многомерных временных рядах. Pre-sorting алгоритм.

    :param time_series: (list[list[float]])
    :param threshold: (float) пороговое значение для определения аномальности точки
    :param window_length: (int) длина окна алгоритма
    :return:
        - anomaly_indices (numpy.array): индексы аномальных точек
        - matrix_profile_matrix (list[numpy.ndarray]): результирующая матрица pre-sorting алгоритма
    """
    number_of_dimensions = len(time_series)
    len_time_series = len(time_series[0])

    tensor_of_euclidian_distances = np.full((
        len_time_series - window_length + 1, len_time_series - window_length + 1, number_of_dimensions),
        np.inf
    )

    for d in range(number_of_dimensions):
        for i in range(len_time_series - window_length + 1):
            for j in range(len_time_series - window_length + 1):
                if i == j:
                    continue
                tensor_of_euclidian_distances[i, j, d] = z_normalized_euclidean_distance(
                    time_series[d][i:i + window_length], time_series[d][j:j + window_length]
                )

    matrix_profile_matrix = np.min(tensor_of_euclidian_distances, axis=2)
    threshold_value = np.percentile(matrix_profile_matrix, threshold)
    anomaly_indices = np.where(matrix_profile_matrix > threshold_value)

    return anomaly_indices, matrix_profile_matrix


def multidimensional_ad_pre_sorting_for_test(
    time_series: list[list[float]],
    threshold: float,
    window_length: int
) -> np.array:
    """
    Поиск аномалий в многомерных временных рядах. Pre-sorting алгоритм для тестирования

    :param time_series: (list[list[float]])
    :param threshold: (float) пороговое значение для определения аномальности точки
    :param window_length: (int) длина окна алгоритма
    :return:
        - anomaly_indices (numpy.array): индексы аномальных точек
    """
    anomaly_indices, matrix_profile_matrix = multidimensional_ad_pre_sorting(time_series, threshold, window_length)
    return anomaly_indices


def multidimensional_ad_mstump(
    time_series: list[list[float]],
    threshold: float,
    window_length: int
) -> tuple[np.array, np.ndarray]:
    """
    Поиск аномалий в многомерных временных рядах с MSTUMP

    :param time_series: (list[list[float]]) многомерный матричный профиль
    :param threshold: (float) пороговое значение для определения аномальности точки
    :param window_length:
    :return:
        - anomaly_indices (np.array): список аномальных точек
        - matrix_profile (np.ndarray) результат работы MSTUMP
    """
    matrix_profile = stumpy.mstump(np.array(time_series), window_length)[0]
    last_mp = matrix_profile[-1]
    threshold = np.percentile(last_mp, threshold)
    above_threshold = last_mp > threshold
    anomaly_indices = np.where(above_threshold)[0]
    return anomaly_indices, matrix_profile


def multidimensional_ad_mstump_for_test(
    time_series: list[list[float]],
    threshold: float,
    window_length: int
) -> np.array:
    """
    Поиск аномалий в многомерных временных рядах с MSTUMP

    :param time_series: (list[list[float]]) многомерный матричный профиль
    :param threshold: (float) пороговое значение для определения аномальности точки
    :param window_length:
    :return:
        - anomaly_indices (np.array): список аномальных точек
    """
    anomaly_indices, matrix_profile = multidimensional_ad_mstump(time_series, threshold, window_length)
    return anomaly_indices
