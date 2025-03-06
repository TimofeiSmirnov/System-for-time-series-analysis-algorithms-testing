import stumpy
from matrixprofile import matrixProfile


def generate_mp(algorithm_type, time_series, m):
    """Вычисляет матричный профиль в зависимости от запроса пользователя"""
    if algorithm_type == "stomp":
        return matrixProfile.stomp(time_series, m)
    elif algorithm_type == "scrimp++":
        return matrixProfile.scrimp_plus_plus(time_series, m)
