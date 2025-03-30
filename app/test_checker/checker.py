import os
import time
import numpy as np
import pandas as pd
import stumpy

from app.algorithms.damp.damp import damp_algorithm
from app.algorithms.multidim_ad.multidimensional_ad import (multidimensional_ad_post_sorting_for_test,
                                                            multidimensional_ad_pre_sorting_for_test,
                                                            multidimensional_ad_mstump_for_test)
from app.algorithms.classic.classic import classic_algorithm


class Checker:
    def __init__(self):
        self.algorithms_ad = {
            "damp": self._damp,
            "post_sorting": self._multidimensional_post_sorting,
            "pre_sorting": self._multidimensional_pre_sorting,
            "mstump": self.multidimensional_ad_mstump,
            "dumb": self._classic
        }

    def _damp(self, time_series, threshold, window_length, learn_window_length):
        return damp_algorithm(time_series[0], threshold, [window_length, learn_window_length])

    def _multidimensional_post_sorting(self, time_series, threshold, window_length, learn_window_length):
        return multidimensional_ad_post_sorting_for_test(time_series, threshold, window_length)

    def _multidimensional_pre_sorting(self, time_series, threshold, window_length, learn_window_length):
        return multidimensional_ad_pre_sorting_for_test(time_series, threshold, window_length)

    def _classic(self, time_series, threshold, window_length, learn_window_length):
        return classic_algorithm(time_series[0], threshold, window_length)

    def multidimensional_ad_mstump(self, time_series, threshold, window_length, learn_window_length):
        return multidimensional_ad_mstump_for_test(time_series, threshold, window_length)

    def _calculate_metrics_ad(self, file_path, anomaly_indices):
        time_series_file = pd.read_csv(file_path)
        anomaly_real_indices = time_series_file["anomaly"].tolist()
        tp, tn, fp, fn = 0, 0, 0, 0

        for i in range(len(anomaly_real_indices)):
            if (i in anomaly_indices) and int(anomaly_real_indices[i]) == 1:
                tp += 1
            elif (i in anomaly_indices) and int(anomaly_real_indices[i]) == 0:
                fp += 1
            elif (i not in anomaly_indices) and int(anomaly_real_indices[i]) == 1:
                fn += 1
            elif (i not in anomaly_indices) and int(anomaly_real_indices[i]) == 0:
                tn += 1

        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        if tn + fp != 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0

        if precision + recall != 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0

        accuracy = (tp + tn) / (tp + tn + fp + fn)  if tp + tn + fp + fn != 0 else 0

        if tp + fp != 0:
            fdr = fp / (tp + fp)
        else:
            fdr = 0

        if tn + fn != 0:
            for_rate = fn / (tn + fn)
        else:
            for_rate = 0

        result = {
            "True Positive Rate (Recall)": recall,
            "Missed Alarm Rate": fn / (tp + fn) if tp + fn != 0 else 0,
            "Specificity": specificity,
            "False Positive Rate": fp / (tn + fp) if tn + fp != 0 else 0,
            "G-mean": ((tp / (tp + fn)) * (tn / (tn + fp))) ** 0.5 if tp + fn != 0 and tn + fp != 0 else 0,
            "Accuracy": accuracy,
            "Precision": precision,
            "F1-Score": f1_score,
            "False Discovery Rate": fdr,
            "False Omission Rate": for_rate,
        }

        return result

    def _calculate_metrics_cpd(self, file_path, cpd_indices):
        time_series_file = pd.read_csv(file_path)
        cp_real_indices = time_series_file["Label"].tolist()

        cpd_indices.sort()

        real_cp_indices = [i for i, label in enumerate(cp_real_indices) if label == 1]

        predicted_cp_indices = [int(i) for i in cpd_indices if i < len(cp_real_indices)]

        real_cp_indices = real_cp_indices[:len(predicted_cp_indices)]

        predicted_cp_indices = predicted_cp_indices[:len(real_cp_indices)]

        absolute_errors = [abs(pred - real) for pred, real in zip(predicted_cp_indices, real_cp_indices)]
        squared_errors = [error ** 2 for error in absolute_errors]
        signed_errors = [pred - real for pred, real in zip(predicted_cp_indices, real_cp_indices)]

        if len(absolute_errors) > 0:
            mae = np.mean(absolute_errors)
            mse = np.mean(squared_errors)
            rmse = np.sqrt(mse)
            msd = np.mean(signed_errors)
        else:
            mae = mse = rmse = msd = 0

        result = {
            "Mean Absolute Error (MAE)": mae,
            "Mean Squared Error (MSE)": mse,
            "Mean Signed Difference (MSD)": msd,
            "Root Mean Squared Error (RMSE)": rmse,
        }

        return result

    def check_ad(self, algorithm_type, threshold, window_length, anomaly_part_of_window=0.5, learn_window_length=None):
        """Тестирование алгоритмов AD по запросу. Возвращаем время и список результатов по метрикам."""
        if not (algorithm_type in self.algorithms_ad.keys()):
            raise ValueError("Такого алгоритма нет")

        if algorithm_type in ("damp", "dumb"):
            data_path_index = "oneDim"
        else:
            data_path_index = "multDim"

        data_path = os.path.join(os.path.dirname(__file__), "data", data_path_index)
        csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

        counter_of_files = 0
        results = {}
        summarised_results = {
            "True Positive Rate (Recall)": 0,
            "Missed Alarm Rate": 0,
            "Specificity": 0,
            "False Positive Rate": 0,
            "G-mean": 0,
            "Accuracy": 0,
            "Precision": 0,
            "F1-Score": 0,
            "False Discovery Rate": 0,
            "False Omission Rate": 0,
        }

        for file_name in csv_files:
            file_path = os.path.join(data_path, file_name)
            path_to_check = os.path.join(os.path.dirname(__file__), "data", "check_" + data_path_index, file_name)
            try:
                time_series_file = pd.read_csv(file_path)
                print("Checking:", file_name)

                time_series_filtered = time_series_file.drop(time_series_file.columns[0], axis=1)
                time_series = [time_series_filtered[col].map(float).tolist() for col in time_series_filtered]
                time_series = np.array(time_series)

                start_time = time.time()
                anomaly_indices = self.algorithms_ad[algorithm_type](
                    time_series=time_series,
                    threshold=threshold,
                    window_length=window_length,
                    learn_window_length=learn_window_length
                )
                execution_time = time.time() - start_time

                expanded_anomaly_indices = set()
                for id in anomaly_indices:
                    expanded_anomaly_indices.update(range(id, id + int(window_length * anomaly_part_of_window) + 1))
                anomaly_indices = list(expanded_anomaly_indices)

                results[file_name] = {
                    "time": execution_time,
                    "metrics": self._calculate_metrics_ad(path_to_check, anomaly_indices)
                }

                for metric in results[file_name]["metrics"].keys():
                    summarised_results[metric] += results[file_name]["metrics"][metric]

                print("Finish checking:", file_name)
                counter_of_files += 1
            except Exception as e:
                raise ValueError(f"Не получилось открыть файл {file_name}")

        for metric in summarised_results.keys():
            summarised_results[metric] /= counter_of_files

        return summarised_results, results

    def check_cpd(self, window_length):
        """Тестирование алгоритмов CPD по запросу. Возвращаем время и список результатов по метрикам."""
        data_path_index = "cpd"

        data_path = os.path.join(os.path.dirname(__file__), "data", data_path_index)
        csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

        counter_of_files = 0
        results = {}
        summarised_results = {
            "Mean Absolute Error (MAE)": 0,
            "Mean Squared Error (MSE)": 0,
            "Mean Signed Difference (MSD)": 0,
            "Root Mean Squared Error (RMSE)": 0,
            "Normalized RMSE (NRMSE)": 0
        }

        for file_name in csv_files:
            file_path = os.path.join(data_path, file_name)
            path_to_check = os.path.join(os.path.dirname(__file__), "data", "check_" + data_path_index, file_name)
            try:
                time_series_file = pd.read_csv(file_path)
                print("Checking:", file_name)

                time_series = time_series_file["value"].map(float).tolist()
                time_series = np.array(time_series)

                start_time = time.time()
                matrix_profile = stumpy.stump(time_series, m=window_length)
                cac, anomaly_indices = stumpy.fluss(matrix_profile[:, 1],
                                                    L=window_length, n_regimes=6, excl_factor=1)
                execution_time = time.time() - start_time
                results[file_name] = {
                    "time": execution_time,
                    "metrics": self._calculate_metrics_cpd(path_to_check, anomaly_indices)
                }

                for metric in results[file_name]["metrics"].keys():
                    summarised_results[metric] += results[file_name]["metrics"][metric]

                print("Finish checking:", file_name)
                counter_of_files += 1
            except Exception as e:
                raise ValueError(f"Не получилось открыть файл {file_name}")

        for metric in summarised_results.keys():
            summarised_results[metric] /= counter_of_files

        return summarised_results, results
