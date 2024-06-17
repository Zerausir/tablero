import io
import json
import pandas as pd
import asyncio
from dash import dcc, html, Input, Output
from django.conf import settings
from django.http import HttpRequest, QueryDict
from django_plotly_dash import DjangoDash
from dash.dependencies import State

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
        dcc.Store(id='store-df-original1'),
        dcc.Store(id='store-df-original2'),
        dcc.Store(id='store-df-original3'),
        dcc.Store(id='store-df-original4'),
        dcc.Store(id='store-df-original5'),
        dcc.Store(id='store-df-original6'),
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
        [Output('store-df-original1', 'data'),
         Output('store-df-original2', 'data'),
         Output('store-df-original3', 'data'),
         Output('store-df-original4', 'data'),
         Output('store-df-original5', 'data'),
         Output('store-df-original6', 'data'),
         Output('data-container', 'children')],
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('city-dropdown', 'value')]
    )
    def update_content_wrapper(fecha_inicio, fecha_fin, ciudad):
        request = HttpRequest()
        request.GET = QueryDict(mutable=True)
        request.GET['start_date'] = fecha_inicio
        request.GET['end_date'] = fecha_fin
        request.GET['city'] = ciudad
        return update_content(request)

    @app.callback(
        Output('heatmap1', 'figure'),
        [Input('store-df-original1', 'data')]
    )
    def update_heatmap1(data):
        df = pd.DataFrame(data)
        return create_heatmap_data(df)

    @app.callback(
        Output('heatmap2', 'figure'),
        [Input('store-df-original2', 'data')]
    )
    def update_heatmap2(data):
        df = pd.DataFrame(data)
        return create_heatmap_data(df)

    @app.callback(
        Output('heatmap3', 'figure'),
        [Input('store-df-original3', 'data')]
    )
    def update_heatmap3(data):
        df = pd.DataFrame(data)
        return create_heatmap_data(df)

    @app.callback(
        [Output('scatter1', 'figure'),
         Output('table1', 'data')],  # Add the table container as an output
        [Input('threshold-slider1', 'value')],
        [State('store-df-original1', 'data')]
    )
    def update_scatter1(threshold, data):
        df = pd.DataFrame(data)
        scatter1_df = calculate_occupation_percentage(df, threshold)
        x_range1 = [scatter1_df['frecuencia_hz'].min(), scatter1_df['frecuencia_hz'].max()]
        scatter1_fig = create_scatter_plot(scatter1_df, x_range1, threshold)
        scatter1_df.columns = ['Frecuencia (Hz)', 'Ocupación (%)']
        table1_data = scatter1_df.to_dict('records')  # Convert the dataframe to a dictionary
        return scatter1_fig, table1_data  # Return the scatter plot figure and the table data

    @app.callback(
        [Output('scatter2', 'figure'),
         Output('table2', 'data')],  # Add the table container as an output
        [Input('threshold-slider2', 'value')],
        [State('store-df-original2', 'data')]
    )
    def update_scatter2(threshold, data):
        df = pd.DataFrame(data)
        scatter2_df = calculate_occupation_percentage(df, threshold)
        x_range2 = [scatter2_df['frecuencia_hz'].min(), scatter2_df['frecuencia_hz'].max()]
        scatter2_fig = create_scatter_plot(scatter2_df, x_range2, threshold)
        scatter2_df.columns = ['Frecuencia (Hz)', 'Ocupación (%)']
        table2_data = scatter2_df.to_dict('records')  # Convert the dataframe to a dictionary
        return scatter2_fig, table2_data  # Return the scatter plot figure and the table data

    @app.callback(
        [Output('scatter3', 'figure'),
         Output('table3', 'data')],  # Add the table container as an output
        [Input('threshold-slider3', 'value')],
        [State('store-df-original3', 'data')]
    )
    def update_scatter3(threshold, data):
        df = pd.DataFrame(data)
        scatter3_df = calculate_occupation_percentage(df, threshold)
        x_range3 = [scatter3_df['frecuencia_hz'].min(), scatter3_df['frecuencia_hz'].max()]
        scatter3_fig = create_scatter_plot(scatter3_df, x_range3, threshold)
        scatter3_df.columns = ['Frecuencia (Hz)', 'Ocupación (%)']
        table3_data = scatter3_df.to_dict('records')  # Convert the dataframe to a dictionary
        return scatter3_fig, table3_data  # Return the scatter plot figure and the table data

    @app.callback(
        [Output('table1-container', 'style'),
         Output('download-excel1', 'style')],
        [Input('toggle-table1', 'n_clicks'),
         Input('threshold-slider1', 'value')],
        [State('table1-container', 'style'),
         State('download-excel1', 'style')]
    )
    def toggle_table1(n_clicks, threshold, current_table_style, current_download_style):
        if n_clicks and threshold is not None:
            if current_table_style.get('display') == 'none':
                return {'overflowX': 'auto', 'maxHeight': '300px'}, {'display': 'inline-block'}
            else:
                return {'display': 'none'}, current_download_style
        return current_table_style, current_download_style

    @app.callback(
        Output("download-data1", "data"),
        [Input("download-excel1", "n_clicks")],
        [State('table1', 'data')]
    )
    def download_excel1(n_clicks, data):
        if n_clicks:
            df = pd.DataFrame(data)
            excel_file = io.BytesIO()
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='703-733MHz')
            excel_file.seek(0)
            return dcc.send_bytes(excel_file.read(), filename="Data_703-733MHz.xlsx")

    @app.callback(
        [Output('table2-container', 'style'),
         Output('download-excel2', 'style')],
        [Input('toggle-table2', 'n_clicks'),
         Input('threshold-slider2', 'value')],
        [State('table2-container', 'style'),
         State('download-excel2', 'style')]
    )
    def toggle_table2(n_clicks, threshold, current_table_style, current_download_style):
        if n_clicks and threshold is not None:
            if current_table_style.get('display') == 'none':
                return {'overflowX': 'auto', 'maxHeight': '300px'}, {'display': 'inline-block'}
            else:
                return {'display': 'none'}, current_download_style
        return current_table_style, current_download_style

    @app.callback(
        Output("download-data2", "data"),
        [Input("download-excel2", "n_clicks")],
        [State('table2', 'data')]
    )
    def download_excel2(n_clicks, data):
        if n_clicks:
            df = pd.DataFrame(data)
            excel_file = io.BytesIO()
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='758-788MHz')
            excel_file.seek(0)
            return dcc.send_bytes(excel_file.read(), filename="Data_758-788MHz.xlsx")

    @app.callback(
        [Output('table3-container', 'style'),
         Output('download-excel3', 'style')],
        [Input('toggle-table3', 'n_clicks'),
         Input('threshold-slider3', 'value')],
        [State('table3-container', 'style'),
         State('download-excel3', 'style')]
    )
    def toggle_table3(n_clicks, threshold, current_table_style, current_download_style):
        if n_clicks and threshold is not None:
            if current_table_style.get('display') == 'none':
                return {'overflowX': 'auto', 'maxHeight': '300px'}, {'display': 'inline-block'}
            else:
                return {'display': 'none'}, current_download_style
        return current_table_style, current_download_style

    @app.callback(
        Output("download-data3", "data"),
        [Input("download-excel3", "n_clicks")],
        [State('table3', 'data')]
    )
    def download_excel3(n_clicks, data):
        if n_clicks:
            df = pd.DataFrame(data)
            excel_file = io.BytesIO()
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='2500-2690MHz')
            excel_file.seek(0)
            return dcc.send_bytes(excel_file.read(), filename="Data_2500-2690MHz.xlsx")


async def customize_data_async(request):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, customize_data, request)
    return data


def update_content(request):
    fecha_inicio = request.GET.get('start_date')
    fecha_fin = request.GET.get('end_date')
    ciudad = request.GET.get('city')

    if not all([fecha_inicio, fecha_fin, ciudad]):
        return {}, {}, {}, {}, {}, {}, "Selecciona una fecha inicial, una fecha final, una ciudad y un umbral"

    data = asyncio.run(customize_data_async(request))

    df_original1, df_original2, df_original3 = data[:3]
    df_clean1, df_clean2, df_clean3 = data[3:]

    df_original1 = convert_timestamps_to_strings(df_original1)
    df_original2 = convert_timestamps_to_strings(df_original2)
    df_original3 = convert_timestamps_to_strings(df_original3)

    tabs_layout = create_heatmap_layout(df_original1, df_original2, df_original3)

    return df_original1.to_dict('records'), df_original2.to_dict('records'), df_original3.to_dict('records'), \
        df_clean1.to_dict('records'), df_clean2.to_dict('records'), df_clean3.to_dict('records'), tabs_layout


register_callbacks()

if __name__ == '__main__':
    app.run_server(debug=True)
