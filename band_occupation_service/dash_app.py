import io
import json
import pandas as pd
import asyncio
from dash import dcc, html, Input, Output
from django.conf import settings
from django.http import HttpRequest, QueryDict
from django_plotly_dash import DjangoDash
from dash.dependencies import State
import plotly.graph_objs as go

from .utils import convert_timestamps_to_strings, create_heatmap_layout, create_heatmap_data, \
    calculate_occupation_percentage, create_scatter_plot
from .services import customize_data

app = DjangoDash(
    name='BandOccupationApp',
    add_bootstrap_links=True,
    external_stylesheets=["/static/css/inner.css"]
)


def define_app_layout():
    return html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date_placeholder_text="Fecha inicial",
            end_date_placeholder_text="Fecha final",
            minimum_nights=0,
            style={'margin': '10px'}
        ),
        dcc.Dropdown(
            id='city-dropdown',
            options=[{'label': ciudad, 'value': ciudad} for ciudad in json.loads(settings.CITIES)],
            placeholder="Selecciona una ciudad",
            style={'margin': '10px'}
        ),
        dcc.Input(
            id='start-freq-input',
            type='number',
            placeholder='Frecuencia inicial (MHz)',
            style={'margin': '10px'}
        ),
        dcc.Input(
            id='end-freq-input',
            type='number',
            placeholder='Frecuencia final (MHz)',
            style={'margin': '10px'}
        ),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=[
                html.Div(
                    id='data-container',  # This is the placeholder for the heatmaps and tables.
                    style={'height': '70vh', 'overflowY': 'auto'}
                )
            ],
            style={'margin': '10px'}
        ),
        dcc.Store(id='store-df-original'),
        dcc.Store(id='store-df-clean'),
    ], style={
        'display': 'flex',
        'flex-direction': 'column',
        'justify-content': 'flex-start',  # Aligns children to the start of the container
        'align-items': 'stretch',  # Stretches children to fill the container width
        'min-height': '100vh',  # Ensures at least the height of the viewport
        'height': 'auto',  # Allows the container to grow beyond the viewport height if needed
    })


app.layout = define_app_layout()


def register_callbacks():
    @app.callback(
        [Output('store-df-original', 'data'),
         Output('store-df-clean', 'data'),
         Output('data-container', 'children')],
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('city-dropdown', 'value'),
         Input('start-freq-input', 'value'),
         Input('end-freq-input', 'value')]
    )
    def update_content_wrapper(fecha_inicio, fecha_fin, ciudad, start_freq, end_freq):
        request = HttpRequest()
        request.GET = QueryDict(mutable=True)
        request.GET['start_date'] = fecha_inicio
        request.GET['end_date'] = fecha_fin
        request.GET['city'] = ciudad
        request.GET['start_freq'] = start_freq
        request.GET['end_freq'] = end_freq
        return update_content(request)

    @app.callback(
        Output('heatmap', 'figure'),
        [Input('store-df-original', 'data')]
    )
    def update_heatmap(data):
        df = pd.DataFrame(data)
        return create_heatmap_data(df)

    @app.callback(
        [Output('scatter', 'figure'),
         Output('table', 'data')],  # Add the table container as an output
        [Input('threshold-slider', 'value')],
        [State('store-df-original', 'data')]
    )
    def update_scatter(threshold, data):
        df = pd.DataFrame(data)
        scatter_df = calculate_occupation_percentage(df, threshold)
        x_range = [scatter_df['frecuencia_hz'].min(), scatter_df['frecuencia_hz'].max()]
        scatter_fig = create_scatter_plot(scatter_df, x_range, threshold)
        scatter_df.columns = ['Frecuencia (Hz)', 'Ocupación (%)']
        table_data = scatter_df.to_dict('records')  # Convert the dataframe to a dictionary
        return scatter_fig, table_data  # Return the scatter plot figure and the table data

    @app.callback(
        [Output('table-container', 'style'),
         Output('download-excel', 'style')],
        [Input('toggle-table', 'n_clicks'),
         Input('threshold-slider', 'value')],
        [State('table-container', 'style'),
         State('download-excel', 'style')]
    )
    def toggle_table(n_clicks, threshold, current_table_style, current_download_style):
        if n_clicks and threshold is not None:
            if current_table_style.get('display') == 'none':
                return {'overflowX': 'auto', 'maxHeight': '300px'}, {'display': 'inline-block'}
            else:
                return {'display': 'none'}, current_download_style
        return current_table_style, current_download_style

    @app.callback(
        Output("download-data", "data"),
        [Input("download-excel", "n_clicks")],
        [State('table', 'data')]
    )
    def download_excel(n_clicks, data):
        if n_clicks:
            df = pd.DataFrame(data)
            excel_file = io.BytesIO()
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data')
            excel_file.seek(0)
            return dcc.send_bytes(excel_file.read(), filename="Data.xlsx")

    @app.callback(
        Output('parameter-heatmap', 'figure'),
        [Input('parameter-dropdown', 'value'),
         Input('store-df-original', 'data')]
    )
    def update_parameter_heatmap(parameter, data):
        if not data or not parameter:
            return go.Figure()

        df = pd.DataFrame(data)
        titles = {
            'level_dbuv_m': 'Nivel de Intensidad de Campo Eléctrico (dBµV/m)',
            'offset_hz': 'Offset (Hz)',
            'modulation_value': 'Valor de Modulación',
            'bandwidth_hz': 'Ancho de Banda (Hz)'
        }

        return create_heatmap_data(df, parameter, titles.get(parameter, ''))


async def customize_data_async(request):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, customize_data, request)
    return data


def update_content(request):
    fecha_inicio = request.GET.get('start_date')
    fecha_fin = request.GET.get('end_date')
    ciudad = request.GET.get('city')
    start_freq = request.GET.get('start_freq')
    end_freq = request.GET.get('end_freq')

    if not all([fecha_inicio, fecha_fin, ciudad, start_freq, end_freq]):
        return {}, {}, "Selecciona una fecha inicial, una fecha final, una ciudad, una frecuencia inicial y una frecuencia final"

    data = asyncio.run(customize_data_async(request))

    df_original, df_clean = data

    df_original = convert_timestamps_to_strings(df_original)

    tabs_layout = create_heatmap_layout(df_original)

    return df_original.to_dict('records'), df_clean.to_dict('records'), tabs_layout


register_callbacks()

if __name__ == '__main__':
    app.run_server(debug=True)
