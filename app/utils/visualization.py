import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_time_series_with_matrix_profile(timestamps, values, values_mp, file_name=""):
    """Визуализирует многомерный временной ряд и несколько матричных профилей"""
    num_dimensions = len(values)
    num_mp = len(values_mp)
    total_rows = num_dimensions + num_mp

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Измерение {i + 1}" for i in range(num_dimensions)] +
                       [f"Матричный профиль {i + 1}" for i in range(num_mp)]
    )

    for i in range(num_dimensions):
        fig.add_trace(
            go.Scatter(x=timestamps, y=values[i], mode='lines', name=f"Измерение {i + 1}"),
            row=i + 1, col=1
        )

    for i in range(num_mp):
        fig.add_trace(
            go.Scatter(x=timestamps, y=values_mp[i], mode='lines', name=f"Матричный профиль {i + 1}"),
            row=num_dimensions + i + 1, col=1
        )

    if file_name != "":
        title = "Визуализация многомерного временного ряда и матричных профилей в файле " + file_name
    else:
        title = "Визуализация многомерного временного ряда и матричных профилей"
    fig.update_layout(
        title=title,
        height=300 * total_rows,
        showlegend=True,
    )

    return fig.to_html()


def visualize_time_series_ad(timestamps, values, ad_values, file_name=""):
    """Визуализирует одномерный временной ряд и выделяет аномалии красными точками"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            line=dict(color='blue'),
            name="Временной ряд"
        )
    )

    anomaly_x = [timestamps[i] for i in ad_values]
    anomaly_y = [values[i] for i in ad_values]

    fig.add_trace(
        go.Scatter(
            x=anomaly_x,
            y=anomaly_y,
            mode='markers',
            marker=dict(color='red', size=8),
            name="Аномалии"
        )
    )

    if file_name != "":
        title = "Визуализация временного ряда с аномалиями " + file_name
    else:
        title = "Визуализация временного ряда с аномалиями"


    fig.update_layout(
        title=title,
        xaxis_title="Время",
        yaxis_title="Значение",
        showlegend=True
    )

    return fig.to_html()


def visualize_time_series_ad_multidim(timestamps, values, ad_values, matrix_profile, file_name=""):
    """Визуализирует многомерный временной ряд с аномалиями и матричные профили на отдельных графиках"""

    num_dimensions = len(values)

    fig = make_subplots(
        rows=2 * num_dimensions, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Временной ряд {dim + 1}" for dim in range(num_dimensions)] +
                       [f"Матричный профиль {dim + 1}" for dim in range(num_dimensions)]
    )

    for dim in range(num_dimensions):
        series_values = values[dim]
        anomaly_x = [timestamps[i] for i in ad_values]
        anomaly_y = [series_values[i] for i in ad_values]

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=series_values,
                mode='lines',
                line=dict(color=f'rgba(0, 0, 255, {1 - dim * 0.1})'),
                name=f"Временной ряд {dim + 1}"
            ),
            row=dim + 1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=anomaly_x,
                y=anomaly_y,
                mode='markers',
                marker=dict(color='red', size=8),
                name=f"Аномалии {dim + 1}"
            ),
            row=dim + 1, col=1
        )

    for dim in range(num_dimensions):
        profile_values = matrix_profile[dim]

        fig.add_trace(
            go.Scatter(
                x=timestamps[:len(profile_values)],
                y=profile_values,
                mode='lines',
                line=dict(color='green'),
                name=f"Матричный профиль {dim + 1}"
            ),
            row=num_dimensions + dim + 1, col=1
        )

    if file_name != "":
        title = "Визуализация многомерного временного ряда с аномалиями и матричными профилями " + file_name
    else:
        title = "Визуализация многомерного временного ряда с аномалиями и матричными профилями"


    fig.update_layout(
        title=title,
        xaxis_title="Время",
        yaxis_title="Значение",
        showlegend=True,
        height=600 * num_dimensions
    )

    return fig.to_html()


def visualize_time_series_with_fluss(timestamps, values, values_fluss, file_name=""):
    """Визуализирует многомерный временной ряд и несколько арочных кривых"""
    num_dimensions = len(values)
    num_of_arc_curves = len(values_fluss)

    total_rows = num_dimensions + num_of_arc_curves

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f"Измерение {i + 1}" for i in range(num_dimensions)] +
                       [f"Арочная кривая {i + 1}" for i in range(num_of_arc_curves)]
    )

    for i in range(num_dimensions):
        fig.add_trace(
            go.Scatter(x=timestamps, y=values[i], mode='lines', name=f"Измерение {i + 1}"),
            row=i + 1, col=1
        )

    for i in range(num_of_arc_curves):
        fig.add_trace(
            go.Scatter(x=timestamps, y=values_fluss[i], mode='lines', name=f"Арочная кривая {i + 1}"),
            row=num_dimensions + i + 1, col=1
        )

    if file_name != "":
        title = "Визуализация многомерного временного ряда и арочной кривой " + file_name
    else:
        title = "Визуализация многомерного временного ряда и арочной кривой"

    fig.update_layout(
        title=title,
        height=300 * total_rows,
        showlegend=True,
    )

    return fig.to_html()
