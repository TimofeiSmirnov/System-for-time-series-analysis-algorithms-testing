from flask import Flask, render_template, request

from app.utils.data_loader import load_time_series
from app.utils.mp_calculator import generate_mp
from app.utils.visualization import visualize_time_series

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Главная страница с возможностью выборов параметра ряда для генерации"""
    plot_html = None
    if request.method == "POST":
        time_series_length = int(request.form['n'])
        try:
            time_series_dim = int(request.form['k'])
        except KeyError:
            time_series_dim = 1
        window_length = int(request.form['m'])
        algorithms = request.form.getlist('algorithm')
        task = request.form['task']
        timestamps, time_series = load_time_series(time_series_length, time_series_dim)

        array_of_mp = []
        print(algorithms)
        for algorithm in algorithms:
            mp, execution_time = generate_mp(algorithm, time_series, window_length)
            for matrix_profile in mp:
                array_of_mp.append(matrix_profile)
            print(algorithm + ": ", execution_time)
        plot_html = visualize_time_series(timestamps, time_series, array_of_mp)

    return render_template("index.html", plot_html=plot_html)


@app.route("/ad_analysis", methods=["GET", "POST"])
def ad_analysis():
    """Страница для анализа алгоритмов Anomaly Detection (AD)"""
    plot_html = None

    return render_template("ad_analysis.html", plot_html=plot_html)


@app.route("/cpd_analysis", methods=["GET", "POST"])
def cpd_analysis():
    """Страница для анализа алгоритмов Change Point Detection (CPD)"""
    plot_html = None

    return render_template("cpd_analysis.html", plot_html=plot_html)
