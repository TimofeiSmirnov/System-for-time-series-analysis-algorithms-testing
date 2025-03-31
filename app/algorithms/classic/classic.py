import stumpy
import numpy as np


def classic_algorithm(
    time_series: np.ndarray,
    threshold: float,
    window_length: int
) -> np.ndarray:
    """
    :param time_series: (numpy.ndarray) одномерный временной ряд
    :param threshold: (float) пороговое значение для определения аномальности точки
    :param window_length: (int) длина окна матричного профиля
    :return:
        - anomaly_indices (numpy.ndarray): индексы аномальных точек во временном ряду
    """
    matrix_profile = stumpy.stump(time_series, m=window_length)
    matrix_profile_data = matrix_profile[:, 0]
    filtered_array = np.percentile(matrix_profile_data, threshold)
    above_threshold = matrix_profile_data > filtered_array
    anomaly_indices = np.where(above_threshold)[0]
    return anomaly_indices
