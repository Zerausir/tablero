import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.exceptions import PreventUpdate
from datetime import date, timedelta


def calculate_disabled_days_for_year(year):
    # Calcula el último día de cada mes en el año especificado
    def last_day_of_month(year, month):
        if month == 12:
            return date(year, month, 31)
        return date(year, month + 1, 1) - timedelta(days=1)

    # Lista de los últimos días de cada mes en el año especificado
    last_days = [last_day_of_month(year, month) for month in range(1, 13)]

    # Genera una lista de todos los días del año especificado
    all_days = [date(year, month, day) for month in range(1, 13)
                for day in range(1, (last_day_of_month(year, month).day + 1))]

    # Filtra los días que no son el último día de cada mes
    disabled_days = [day for day in all_days if day not in last_days]

    return disabled_days


def update_table(dataframe, selected_indicador, table_id):
    filtered_df = dataframe.copy()

    if selected_indicador:
        filtered_df = filtered_df[filtered_df['INDICADOR_CORTO'].isin(selected_indicador)]

    return filtered_df.to_dict('records')


def download_excel(n_clicks, dataframe):
    if n_clicks is None:
        raise PreventUpdate
    # Ensure that dataframe is a DataFrame
    if isinstance(dataframe, pd.DataFrame):
        return dcc.send_data_frame(dataframe.to_excel, filename="datos_descargados.xlsx", index=False)
    else:
        raise ValueError("The provided data for download is not a pandas DataFrame.")


def create_pie_charts(df, calculators, selected_indicadores=None):
    pie_charts = []
    for indicador in (selected_indicadores or calculators.keys()):
        nro_informes = df[df['INDICADOR_CORTO'] == indicador].shape[0] if 'INDICADOR_CORTO' in df.columns else 0
        porcentaje = calculators[indicador](nro_informes) if calculators.get(indicador) else 0
        fig = go.Figure(data=[
            go.Pie(
                labels=['Avance', 'Restante'],
                values=[porcentaje, 100 - porcentaje],
                hole=.3,
                marker=dict(colors=['#007BFF', '#D62828']),
            )
        ])
        fig.update_layout(
            title_text=f"Avance indicador {indicador} ({nro_informes} informes)",
            title_x=0.5,
            legend=dict(traceorder='normal', orientation="h", x=0.5, xanchor="center"),
            margin=dict(l=20, r=20, t=45, b=20),  # Establece márgenes internos
            uniformtext_minsize=12,
            uniformtext_mode='hide'
        )

        pie_chart_div = html.Div([
            dcc.Graph(
                figure=fig,
                id={
                    'type': 'dynamic-pie-chart',
                    'index': indicador
                },
            ),
            html.Div(id={'type': 'details', 'index': indicador}, className='verifiable-details')
            # Div para los detalles
        ], className='six columns')

        pie_charts.append(pie_chart_div)

    return html.Div(pie_charts, className='row')


def ccde01(nro_informes):
    return min((nro_informes / 60) * 100, 100) if nro_informes else 0


def ccde02(nro_informes):
    return min((nro_informes / 2) * 100, 100) if nro_informes else 0


def ccde03(nro_informes):
    return min((nro_informes / 6) * 100, 100) if nro_informes else 0


def ccde04(nro_informes):
    return min((nro_informes / 140) * 100, 100) if nro_informes else 0


def ccde11(nro_informes):
    return min((nro_informes / 2) * 100, 100) if nro_informes else 0


def ccdh01(nro_informes):
    return min((nro_informes / 12) * 100, 100) if nro_informes else 0


def ccds01(nro_informes):
    return min((nro_informes / 51) * 100, 100) if nro_informes else 0


def ccds03(nro_informes):
    return min((nro_informes / 2) * 100, 100) if nro_informes else 0


def ccds05(nro_informes):
    return min((nro_informes / 2) * 100, 100) if nro_informes else 0


def ccds08(nro_informes):
    return min((nro_informes / 3) * 100, 100) if nro_informes else 0


def ccds11(nro_informes):
    return min((nro_informes / 4) * 100, 100) if nro_informes else 0


def ccds12(nro_informes):
    return min((nro_informes / 1) * 100, 100) if nro_informes else 0


def ccds13(nro_informes):
    return min((nro_informes / 10) * 100, 100) if nro_informes else 0


def ccds16(nro_informes):
    return min((nro_informes / 15) * 100, 100) if nro_informes else 0


def ccds17(nro_informes):
    return min((nro_informes / 37) * 100, 100) if nro_informes else 0


def ccds18(nro_informes):
    return min((nro_informes / 8) * 100, 100) if nro_informes else 0


def ccds30(nro_informes):
    return min((nro_informes / 7) * 100, 100) if nro_informes else 0


def ccds31(nro_informes):
    return min((nro_informes / 1) * 100, 100) if nro_informes else 0


def ccds32(nro_informes):
    return min((nro_informes / 12) * 100, 100) if nro_informes else 0


def ccdr04(nro_informes):
    return min((nro_informes / 23) * 100, 100) if nro_informes else 0
