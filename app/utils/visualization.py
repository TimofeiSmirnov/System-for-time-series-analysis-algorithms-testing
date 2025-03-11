import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_time_series(timestamps, values, values_mp):
    """Визуализирует многомерный временной ряд и несколько матричных профилей"""
    num_dimensions = len(values)  # Количество измерений временного ряда
    num_mp = len(values_mp)  # Количество матричных профилей
    print(values_mp)

    # Общее количество подграфиков: одно для каждого измерения + для каждого матричного профиля
    total_rows = num_dimensions + num_mp

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f"Измерение {i + 1}" for i in range(num_dimensions)] +
                       [f"Матричный профиль {i + 1}" for i in range(num_mp)]
    )

    # Добавляем измерения временного ряда
    for i in range(num_dimensions):
        fig.add_trace(
            go.Scatter(x=timestamps, y=values[i], mode='lines', name=f"Измерение {i + 1}"),
            row=i + 1, col=1
        )

    # Добавляем матричные профили
    for i in range(num_mp):
        fig.add_trace(
            go.Scatter(x=timestamps, y=values_mp[i], mode='lines', name=f"Матричный профиль {i + 1}"),
            row=num_dimensions + i + 1, col=1
        )

    # Обновляем макет
    fig.update_layout(
        title="Визуализация многомерного временного ряда и матричных профилей",
        height=300 * total_rows,  # Высота зависит от количества подграфиков
        showlegend=True,
    )

    return fig.to_html()