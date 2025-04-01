# В работе часть кода была взята с репозитория:
# https://github.com/tirthajyoti/Synthetic-data-gen/blob/master/Notebooks/Synth_Time_series.ipynb

import numpy as np
import random
import math


class TimeSeriesGenerator:
    """
    Класс, отвечающий за генерацию временных рядов
    """
    def __init__(self, n, k, avg_pattern_length=50, avg_amplitude=1, default_variance=0.1,
                 variance_pattern_length=50, variance_amplitude=1, generators=None, include_negatives=False):
        self.n = n
        self.k = k
        self.avg_pattern_length = avg_pattern_length
        self.avg_amplitude = avg_amplitude
        self.default_variance = default_variance
        self.variance_pattern_length = variance_pattern_length
        self.variance_amplitude = variance_amplitude
        self.include_negatives = include_negatives

        if generators is None:
            self.generators = [self.generate_bell, self.generate_funnel, self.generate_cylinder]
        else:
            self.generators = generators

    def generate_bell(self, length, amplitude, default_variance):
        """Генерирует восходящий паттерн"""
        bell = np.random.normal(0, default_variance, length) + amplitude * np.arange(length) / length
        return bell

    def generate_funnel(self, length, amplitude, default_variance):
        """Генерирует нисходящий паттерн"""
        funnel = np.random.normal(0, default_variance, length) + amplitude * np.arange(length)[::-1] / length
        return funnel

    def generate_cylinder(self, length, amplitude, default_variance):
        """Генерирует горизонтальный паттерн"""
        cylinder = np.random.normal(0, default_variance, length) + amplitude
        return cylinder

    def generate_ts_data(self):
        """Генерирует временной ряд"""
        data = np.random.normal(0, self.default_variance, self.n)
        current_start = random.randint(0, self.avg_pattern_length)
        current_length = max(1, math.ceil(random.gauss(self.avg_pattern_length, self.variance_pattern_length)))

        while current_start + current_length < self.n:
            generator = random.choice(self.generators)
            current_amplitude = random.gauss(self.avg_amplitude, self.variance_amplitude)

            while current_length <= 0:
                current_length = -(current_length - 1)

            pattern = generator(current_length, current_amplitude, self.default_variance)

            data[current_start: current_start + current_length] = pattern

            current_start = current_start + current_length + random.randint(0, self.avg_pattern_length)
            current_length = max(1, math.ceil(random.gauss(self.avg_pattern_length, self.variance_pattern_length)))

        return np.array(data)

    def generate_timestamps(self):
        """Создает последовательность временных меток"""
        return np.arange(self.n).astype(np.float64)

    def generate_time_series(self):
        """Генерирует временные ряды"""
        values = []
        for i in range(self.k):
            values.append(self.generate_ts_data())
        return self.generate_timestamps(), values
