import io
import json
import pandas as pd
import asyncio
from dash import dcc, html, Input, Output, dash_table
from django.conf import settings
from django.http import HttpRequest, QueryDict
from django_plotly_dash import DjangoDash
from dash.dependencies import State, ALL
import plotly.graph_objs as go

from .utils import convert_timestamps_to_strings, create_heatmap_layout, create_heatmap_data, \
    calculate_occupation_percentage, create_scatter_plot, calculate_intermodulation_products, create_intermod_heatmap, \
    analyze_specific_frequencies, create_frequency_timeline
from .services import customize_data

app = DjangoDash(
    name='BandOccupationApp',
    add_bootstrap_links=True,
    external_stylesheets=["/static/css/inner.css"]
)

# ── Estilos de sidebar ────────────────────────────────────────────────────────
_SIDEBAR_LABEL = {
    'fontSize': '10px',
    'fontWeight': '700',
    'textTransform': 'uppercase',
    'letterSpacing': '0.09em',
    'color': '#9ca3af',
    'marginBottom': '8px',
    'marginTop': '16px',
}
_SIDEBAR_LABEL_FIRST = {**_SIDEBAR_LABEL, 'marginTop': '0'}

_INPUT_STYLE = {
    'width': '100%',
    'height': '36px',
    'border': '1px solid rgba(0,0,0,.18)',
    'borderRadius': '8px',
    'padding': '0 10px',
    'fontSize': '13px',
    'background': '#f7f8fa',
    'color': '#111827',
    'marginBottom': '4px',
    'boxSizing': 'border-box',
}

_BTN_PRIMARY = {
    'width': '100%',
    'padding': '10px 0',
    'background': '#185FA5',
    'color': '#fff',
    'border': 'none',
    'borderRadius': '8px',
    'fontSize': '13px',
    'fontWeight': '600',
    'cursor': 'pointer',
    'marginTop': '16px',
}

_BTN_SECONDARY = {
    'width': '100%',
    'padding': '9px 0',
    'background': 'transparent',
    'color': '#185FA5',
    'border': '1px solid #185FA5',
    'borderRadius': '8px',
    'fontSize': '13px',
    'fontWeight': '600',
    'cursor': 'pointer',
    'marginTop': '8px',
}


def define_app_layout():
    return html.Div([

        # ── Sidebar de filtros ─────────────────────────────────────────────────
        html.Div([

            html.Div('Período de análisis', style=_SIDEBAR_LABEL_FIRST),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date_placeholder_text="Fecha inicial",
                end_date_placeholder_text="Fecha final",
                minimum_nights=0,
                display_format='DD/MM/YYYY',
                style={'width': '100%', 'marginBottom': '12px'},
            ),

            html.Div('Ciudad', style=_SIDEBAR_LABEL),
            dcc.Dropdown(
                id='city-dropdown',
                options=[{'label': c, 'value': c} for c in json.loads(settings.CITIES)],
                placeholder="Selecciona una ciudad",
                style={'marginBottom': '4px'},
            ),

            html.Div('Rango de frecuencias (MHz)', style=_SIDEBAR_LABEL),
            dcc.Input(
                id='start-freq-input',
                type='number',
                placeholder='Frecuencia inicial',
                style=_INPUT_STYLE,
            ),
            dcc.Input(
                id='end-freq-input',
                type='number',
                placeholder='Frecuencia final',
                style=_INPUT_STYLE,
            ),

        ], style={
            'width': '260px',
            'flexShrink': '0',
            'background': '#ffffff',
            'borderRight': '1px solid rgba(0,0,0,.10)',
            'padding': '20px 18px',
            'overflowY': 'auto',
            'display': 'flex',
            'flexDirection': 'column',
        }),

        # ── Área principal ─────────────────────────────────────────────────────
        html.Div([

            dcc.Loading(
                id="loading-1",
                type="circle",
                color='#185FA5',
                children=[
                    html.Div(
                        id='data-container',
                        style={'minHeight': '400px'},
                    )
                ],
            ),

            # Stores
            dcc.Store(id='store-df-original'),
            dcc.Store(id='store-df-clean'),

        ], style={
            'flex': '1',
            'minWidth': '0',
            'padding': '20px',
            'overflowY': 'auto',
            'background': '#f0f2f5',
        }),

    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'minHeight': '100%',
        'height': 'auto',
        'background': '#f0f2f5',
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
         Output('table', 'data')],
        [Input('threshold-slider', 'value')],
        [State('store-df-original', 'data')]
    )
    def update_scatter(threshold, data):
        df = pd.DataFrame(data)
        scatter_df = calculate_occupation_percentage(df, threshold)
        x_range = [scatter_df['frecuencia_hz'].min(), scatter_df['frecuencia_hz'].max()]
        scatter_fig = create_scatter_plot(scatter_df, x_range, threshold)
        scatter_df.columns = ['Frecuencia (Hz)', 'Ocupación (%)']
        table_data = scatter_df.to_dict('records')
        return scatter_fig, table_data

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
            'bandwidth_hz': 'Ancho de Banda (Hz)',
        }
        return create_heatmap_data(df, parameter, titles.get(parameter, ''))

    register_intermod_callbacks()


def register_intermod_callbacks():
    @app.callback(
        Output('source-ranges-container', 'children'),
        [Input('add-range-button', 'n_clicks')],
        [State('source-ranges-container', 'children')]
    )
    def add_range_input(n_clicks, children):
        if n_clicks is None:
            return children

        new_range = html.Div([
            dcc.Input(
                id={'type': 'source-start', 'index': len(children)},
                type='number',
                placeholder=f'Inicio rango {len(children) + 1}',
                style={**_INPUT_STYLE, 'width': '45%', 'display': 'inline-block', 'marginRight': '4px'},
            ),
            html.Span(' - ', style={'color': '#9ca3af', 'margin': '0 4px'}),
            dcc.Input(
                id={'type': 'source-end', 'index': len(children)},
                type='number',
                placeholder=f'Fin rango {len(children) + 1}',
                style={**_INPUT_STYLE, 'width': '45%', 'display': 'inline-block'},
            ),
        ], style={'marginBottom': '6px'})

        return children + [new_range]

    @app.callback(
        [Output('intermod-heatmap', 'figure'),
         Output('intermod-table', 'data')],
        [Input('execute-analysis-button', 'n_clicks')],
        [State('analysis-start-freq', 'value'),
         State('analysis-end-freq', 'value'),
         State('intermod-type', 'value'),
         State('source-ranges-container', 'children'),
         State('store-df-original', 'data')]
    )
    def update_intermod_analysis(n_clicks, analysis_start, analysis_end, intermod_types,
                                 source_ranges_children, data):
        if n_clicks is None:
            return go.Figure(), []

        if not all([analysis_start, analysis_end, data, intermod_types]):
            return go.Figure(), []

        df = pd.DataFrame(data)
        if df.empty:
            return go.Figure(), []

        source_ranges = []
        for child in source_ranges_children:
            try:
                start = float(child['props']['children'][0]['props']['value'])
                end = float(child['props']['children'][2]['props']['value'])
                if start and end:
                    source_ranges.append((start, end))
            except (TypeError, KeyError):
                continue

        analysis_range = (analysis_start * 1e6, analysis_end * 1e6)

        products_df = calculate_intermodulation_products(
            df,
            intermod_types,
            [(start * 1e6, end * 1e6) for start, end in source_ranges]
        )

        fig = create_intermod_heatmap(df, products_df, analysis_range, source_ranges)
        return fig, products_df.to_dict('records')

    @app.callback(
        [Output('frequency-analysis-results', 'children')],
        [Input('execute-analysis-button', 'n_clicks')],
        [State('specific-frequencies', 'value'),
         State('store-df-original', 'data'),
         State('analysis-start-freq', 'value'),
         State('analysis-end-freq', 'value'),
         State('intermod-type', 'value'),
         State('source-ranges-container', 'children')]
    )
    def update_frequency_analysis(n_clicks, specific_freqs, data, start_freq, end_freq, intermod_types,
                                  source_ranges_children):
        if not n_clicks or not specific_freqs or not data:
            return [html.Div()]

        try:
            selected_frequencies = [float(f.strip()) for f in specific_freqs.split(',')]
            df = pd.DataFrame(data)

            source_ranges = []
            for child in source_ranges_children:
                try:
                    start = float(child['props']['children'][0]['props']['value'])
                    end = float(child['props']['children'][2]['props']['value'])
                    if start and end:
                        source_ranges.append((start, end))
                except (TypeError, KeyError):
                    continue

            products_df = calculate_intermodulation_products(df, intermod_types, source_ranges)
            results = analyze_specific_frequencies(df, products_df, selected_frequencies)

            return [html.Div([
                html.Div([
                    html.H4(
                        f'Frecuencia: {freq} MHz',
                        style={'fontSize': '15px', 'fontWeight': '600', 'marginBottom': '8px', 'color': '#111827'}
                    ),
                    html.Div([
                        html.P(f'Nivel promedio medido: {info["average_level"]:.2f} dBµV/m',
                               style={'fontSize': '13px', 'color': '#374151'}),
                        html.P(f'Número de productos que afectan: {info["num_products"]}',
                               style={'fontSize': '13px', 'color': '#374151'}),
                        html.P(f'Contribución total de productos: {info["total_contribution"]:.2f} dBµV/m',
                               style={'fontSize': '13px', 'color': '#374151'}),
                    ]),
                    html.Div([
                        html.H5('Productos que contribuyen:',
                                style={'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '6px',
                                       'marginTop': '10px'}),
                        dash_table.DataTable(
                            data=info['contributing_products'],
                            columns=[
                                {'name': 'Tipo', 'id': 'type'},
                                {'name': 'Ecuación', 'id': 'equation'},
                                {'name': 'Nivel', 'id': 'level'},
                            ],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'fontSize': '13px'},
                            style_header={'fontWeight': '700', 'backgroundColor': '#f7f8fa'},
                        ) if info['contributing_products'] else html.P(
                            'No hay productos que afecten a esta frecuencia',
                            style={'fontSize': '13px', 'color': '#9ca3af'}
                        ),
                    ]),
                    dcc.Graph(figure=create_frequency_timeline(info['measurements'], freq)),
                ], style={
                    'background': '#fff',
                    'border': '1px solid rgba(0,0,0,.10)',
                    'borderRadius': '10px',
                    'padding': '16px',
                    'marginBottom': '16px',
                }),
            ]) for freq, info in results.items()]
        except Exception as e:
            return [html.Div(f'Error en el análisis: {str(e)}',
                             style={'color': '#b91c1c', 'fontSize': '13px'})]


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
        return {}, {}, html.Div(
            'Selecciona una fecha inicial, una fecha final, una ciudad y un rango de frecuencias.',
            style={
                'padding': '20px',
                'background': '#eff6ff',
                'border': '1px solid #bfdbfe',
                'borderRadius': '8px',
                'fontSize': '13px',
                'color': '#1e40af',
            }
        )

    data = asyncio.run(customize_data_async(request))
    df_original, df_clean = data
    df_original = convert_timestamps_to_strings(df_original)
    tabs_layout = create_heatmap_layout(df_original)
    return df_original.to_dict('records'), df_clean.to_dict('records'), tabs_layout


register_callbacks()

if __name__ == '__main__':
    app.run_server(debug=True)
