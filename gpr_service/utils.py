import pandas as pd
import dash
import plotly.graph_objects as go
from dash import dcc, html


def update_table(dataframe, selected_indicador, table_id):
    filtered_df = dataframe.copy()

    if selected_indicador:
        filtered_df = filtered_df[filtered_df['INDICADOR_CORTO'].isin(selected_indicador)]

    return filtered_df.to_dict('records')


def download_excel(n_clicks, table_data, table_id):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_excel, filename="datos_descargados.xlsx", index=False)


# En utils.py, modifica create_pie_charts temporalmente para retornar solo un gráfico.
def create_pie_charts(df, calculators, selected_indicadores=None):
    pie_charts = []
    row = []

    # Define cuáles indicadores usar: todos de calculators o solo los seleccionados
    indicadores_a_usar = selected_indicadores if selected_indicadores else calculators.keys()

    for indicador in indicadores_a_usar:
        # Encuentra la cantidad de informes para el indicador, si no hay, usa 0
        nro_informes = df[df['INDICADOR_CORTO'] == indicador]['Nro. INFORME'].count() if not df.empty else 0
        calculador = calculators.get(indicador)
        porcentaje = calculador(nro_informes) if calculador else 0  # Asegúrate de que hay una función calculador

        # Define los colores para cada parte del pie chart
        colores = ['#007BFF', '#D62828']  # Azul para 'Avance', rojo para 'Restante'

        # Nota que hemos cambiado el orden en los labels y values aquí para que 'Avance' sea primero
        fig = go.Figure(data=[go.Pie(labels=['Avance', 'Restante'], values=[porcentaje, 100 - porcentaje], hole=0.3,
                                     marker=dict(colors=colores), sort=False)])
        fig.update_layout(title_text=f"Avance indicador {indicador} ({nro_informes} informes)", title_x=0.5,
                          legend=dict(traceorder='normal'))
        row.append(dcc.Graph(figure=fig))

        # Añadir los gráficos a pie_charts en filas de tres
        if len(row) == 3:
            pie_charts.append(html.Div(row, style={'display': 'flex', 'justify-content': 'space-around'}))
            row = []

    # Añadir la última fila si tiene menos de tres gráficos
    if row:
        pie_charts.append(html.Div(row, style={'display': 'flex', 'justify-content': 'space-around'}))

    return pie_charts


def ccde01(nro_informes):
    return (nro_informes / 60) * 100 if nro_informes else 0


def ccde02(nro_informes):
    return (nro_informes / 2) * 100 if nro_informes else 0


def ccde03(nro_informes):
    return (nro_informes / 6) * 100 if nro_informes else 0


def ccde04(nro_informes):
    return (nro_informes / 140) * 100 if nro_informes else 0


def ccde11(nro_informes):
    return (nro_informes / 2) * 100 if nro_informes else 0


def ccdh01(nro_informes):
    return (nro_informes / 12) * 100 if nro_informes else 0


def ccds01(nro_informes):
    return (nro_informes / 51) * 100 if nro_informes else 0


def ccds03(nro_informes):
    return (nro_informes / 2) * 100 if nro_informes else 0


def ccds05(nro_informes):
    return (nro_informes / 2) * 100 if nro_informes else 0


def ccds08(nro_informes):
    return (nro_informes / 3) * 100 if nro_informes else 0


def ccds11(nro_informes):
    return (nro_informes / 4) * 100 if nro_informes else 0


def ccds12(nro_informes):
    return (nro_informes / 1) * 100 if nro_informes else 0


def ccds13(nro_informes):
    return (nro_informes / 10) * 100 if nro_informes else 0


def ccds16(nro_informes):
    return (nro_informes / 15) * 100 if nro_informes else 0


def ccds17(nro_informes):
    return (nro_informes / 37) * 100 if nro_informes else 0


def ccds18(nro_informes):
    return (nro_informes / 8) * 100 if nro_informes else 0


def ccds30(nro_informes):
    return (nro_informes / 7) * 100 if nro_informes else 0


def ccds31(nro_informes):
    return (nro_informes / 1) * 100 if nro_informes else 0


def ccds32(nro_informes):
    return (nro_informes / 12) * 100 if nro_informes else 0


def ccdr04(nro_informes):
    return (nro_informes / 23) * 100 if nro_informes else 0
