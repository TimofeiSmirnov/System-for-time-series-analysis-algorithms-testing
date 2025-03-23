import os
from flask import Flask, render_template, request
from app.algorithms.apply_algorithms import apply_algorithm_1d, apply_algorithm_nd
from app.algorithms.multidim_ad.multidimensional_ad import multidimensional_ad
from app.data_controller.data_controller import DataController
from app.utils.mp_calculator import generate_mp, arc_curve_calculator
from app.utils.visualization import visualize_time_series_with_matrix_profile, visualize_time_series_ad, \
    visualize_time_series_with_fluss, visualize_time_series_ad_multidim
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'app', 'data', 'userTS')


@app.route("/", methods=["GET", "POST"])
def index():
    """Главная страница с возможностью выбора ряда и генерации матричного профиля по нему"""
    plot_html = None
    data_controller = DataController()
    available_series = data_controller.get_available_series()
    timestamps = []
    time_series = []
    matrix_profile_algorithm = "stomp"
    window_length = 1
    matrix_profile = []
    execution_time = 999
    time_series_length = 0
    time_series_dim = 0

    if request.method == "POST":
        if 'n' in request.form and 'k' in request.form:
            time_series_length = int(request.form["n"])
            time_series_dim = int(request.form["k"])

            timestamps, time_series = data_controller.generate_series(time_series_length, time_series_dim)

            if "algorithm_for_gen" in request.form and "m_for_gen" in request.form:
                matrix_profile_algorithm = request.form["algorithm_for_gen"]
                window_length = int(request.form["m_for_gen"])

                matrix_profile, execution_time = generate_mp(matrix_profile_algorithm, time_series, window_length)
        elif 'series' in request.form:
            requested_time_series_to_load = request.form["series"]
            timestamps, time_series = data_controller.get_series(requested_time_series_to_load)
            time_series_length = len(time_series)
            time_series_dim = 1
            if "algorithm_for_choose" in request.form and "m_for_choose" in request.form:
                matrix_profile_algorithm = request.form["algorithm_for_choose"]
                window_length = int(request.form["m_for_choose"])

                matrix_profile, execution_time = generate_mp(matrix_profile_algorithm, time_series, window_length)
        elif 'file' in request.files:
            uploaded_file = request.files['file']

            if uploaded_file:
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)

                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)

                uploaded_file.save(file_path)

                timestamps, time_series = data_controller.get_series(filename)
                time_series_length = len(time_series)
                time_series_dim = 1

            if "algorithm_for_load" in request.form and "m_for_load" in request.form:
                matrix_profile_algorithm = request.form["algorithm_for_load"]
                window_length = int(request.form["m_for_load"])

                matrix_profile, execution_time = generate_mp(matrix_profile_algorithm, time_series, window_length)

        plot_html = visualize_time_series_with_matrix_profile(timestamps, time_series, matrix_profile)
        return render_template("index.html", plot_html=plot_html, available_series=available_series)

    return render_template("index.html", plot_html=plot_html, available_series=available_series)


@app.route("/ad_analysis", methods=["GET", "POST"])
def ad_analysis():
    """Страница для анализа алгоритмов Anomaly Detection (AD)"""
    plot_html = None
    data_controller = DataController()
    available_series = data_controller.get_available_series()
    timestamps = []
    time_series = []
    matrix_profile_algorithm = "stomp"
    threshold = 95.0
    matrix_profile = []
    execution_time = 999
    anomaly_indexes = []
    matrix_profile_for_multidimensional_ts = []

    if request.method == "POST":
        if 'n' in request.form and 'k' in request.form:
            time_series_length = int(request.form["n"])
            time_series_dim = int(request.form["k"])

            timestamps, time_series = data_controller.generate_series(time_series_length, time_series_dim)

            if "algorithm_for_gen" in request.form and "threshold_for_gen" in request.form:
                matrix_profile_algorithm = request.form["algorithm_for_gen"]
                threshold = float(request.form["threshold_for_gen"])
                # matrix_profile_for_multidimensional_ts = multidimensional_ad(time_series, 50)
                anomaly_indexes = apply_algorithm_1d(matrix_profile_algorithm, time_series[0], threshold)
        elif 'series' in request.form:
            requested_time_series_to_load = request.form["series"]
            timestamps, time_series = data_controller.get_series(requested_time_series_to_load)

            if "algorithm_for_choose" in request.form and "threshold_for_choose" in request.form:
                matrix_profile_algorithm = request.form["algorithm_for_choose"]
                threshold = float(request.form["threshold_for_choose"])
                anomaly_indexes = apply_algorithm_1d(matrix_profile_algorithm, time_series[0], threshold)
        elif 'file' in request.files:
            uploaded_file = request.files['file']

            if uploaded_file:
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)

                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)

                uploaded_file.save(file_path)

                timestamps, time_series = data_controller.get_series(filename)

            if "algorithm_for_load" in request.form and "threshold_for_load" in request.form:
                matrix_profile_algorithm = request.form["algorithm_for_load"]
                threshold = float(request.form["threshold_for_load"])

                anomaly_indexes = apply_algorithm_1d(matrix_profile_algorithm, time_series[0], threshold)

        plot_html = visualize_time_series_ad(timestamps, time_series[0], anomaly_indexes)
        return render_template("ad_analysis.html", plot_html=plot_html, available_series=available_series)

    return render_template("ad_analysis.html", plot_html=plot_html, available_series=available_series)


@app.route("/cpd_analysis", methods=["GET", "POST"])
def cpd_analysis():
    """Страница для анализа алгоритмов Change Point Detection (CPD)"""
    plot_html = None
    data_controller = DataController()
    available_series = data_controller.get_available_series()
    timestamps = []
    time_series = []
    number_of_regimes = 2
    window_size = 100

    if request.method == "POST":
        if 'n' in request.form and 'k' in request.form:
            time_series_length = int(request.form["n"])
            time_series_dim = int(request.form["k"])
            window_size = int(request.form["m_gen"])
            timestamps, time_series = data_controller.generate_series(time_series_length, time_series_dim)
        elif 'series' in request.form:
            requested_time_series_to_load = request.form["series"]
            window_size = int(request.form["m_choose"])
            timestamps, time_series = data_controller.get_series(requested_time_series_to_load)
        elif 'file' in request.files:
            uploaded_file = request.files['file']
            window_size = int(request.form["m_load"])
            if uploaded_file:
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)

                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)

                uploaded_file.save(file_path)
                timestamps, time_series = data_controller.get_series(filename)

        arc_curve = arc_curve_calculator(time_series[0], window_size, number_of_regimes)
        plot_html = visualize_time_series_with_fluss(timestamps, time_series, arc_curve)
        return render_template("cpd_analysis.html", plot_html=plot_html, available_series=available_series)

    return render_template("cpd_analysis.html", plot_html=plot_html, available_series=available_series)


@app.route("/multidim_ad_analysis", methods=["GET", "POST"])
def multidim_ad_analysis():
    """Страница для анализа алгоритмов Anomaly Detection (AD)"""
    plot_html = None
    data_controller = DataController()
    available_series = data_controller.get_available_series()
    timestamps = []
    time_series = []
    matrix_profile_algorithm = "stomp"
    threshold = 95.0
    matrix_profile = []
    execution_time = 999
    anomaly_indexes = []
    matrix_profile_for_multidimensional_ts = []

    if request.method == "POST":
        if 'n' in request.form and 'k' in request.form:
            time_series_length = int(request.form["n"])
            time_series_dim = int(request.form["k"])

            timestamps, time_series = data_controller.generate_series(time_series_length, time_series_dim)

            if "algorithm_for_gen" in request.form and "threshold_for_gen" in request.form:
                matrix_profile_algorithm = request.form["algorithm_for_gen"]
                threshold = float(request.form["threshold_for_gen"])
        elif 'series' in request.form:
            requested_time_series_to_load = request.form["series"]
            timestamps, time_series = data_controller.get_series(requested_time_series_to_load)

            if "algorithm_for_choose" in request.form and "threshold_for_choose" in request.form:
                matrix_profile_algorithm = request.form["algorithm_for_choose"]
                threshold = float(request.form["threshold_for_choose"])
        elif 'file' in request.files:
            uploaded_file = request.files['file']

            if uploaded_file:
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)

                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)

                uploaded_file.save(file_path)

                timestamps, time_series = data_controller.get_series(filename)

            if "algorithm_for_load" in request.form and "threshold_for_load" in request.form:
                matrix_profile_algorithm = request.form["algorithm_for_load"]
                threshold = float(request.form["threshold_for_load"])

        anomaly_indexes, matrix_profile = apply_algorithm_nd(matrix_profile_algorithm, time_series, threshold)
        plot_html = visualize_time_series_ad_multidim(timestamps, time_series, anomaly_indexes, matrix_profile)
        return render_template("multidim_ad_analysis.html", plot_html=plot_html, available_series=available_series)

    return render_template("multidim_ad_analysis.html", plot_html=plot_html, available_series=available_series)
