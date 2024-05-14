import io
import pandas as pd

from django_plotly_dash import DjangoDash
from django.conf import settings
from datetime import date, datetime
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .services import process_data, verificables, procesar_mes_con_fecha, actualizar_planificada, pact_2024, \
    pac_verificables
from .utils import calculate_disabled_days_for_year, create_pie_charts_for_indicators, create_summary_pie_charts

# Inicializa df_pact2024 al comienzo de tu script, justo después de importar tus módulos y funciones necesarias
df_pact2024 = pact_2024()

# Inicializa estos DataFrames con un valor por defecto (vacío) y actualiza solo con callbacks
df_final = None
df_verificables = None
df_pact2024_verificables = None

app = DjangoDash(
    name='GprApp',
    add_bootstrap_links=True,
    external_stylesheets=["/static/css/inner.css"]
)

app.layout = html.Div(children=[
    # Filtros Dropdown
    html.Div([
        html.Label("Fecha de corte:"),
        dcc.DatePickerSingle(
            id='fecha-seleccionada',
            min_date_allowed=date(2024, 1, 1),
            max_date_allowed=date(2024, 12, 31),
            initial_visible_month=date(2024, 1, 1),
            date=date(2024, 1, 31),  # Asegúrate de que esta sea una fecha válida según tu lógica
            disabled_days=calculate_disabled_days_for_year(2024)
        ),
    ], style={'marginBottom': 10, 'marginTop': 10}),

    # Filtro de indicadores
    html.Div([
        html.Label("Filtrar por indicador:"),
        dcc.Dropdown(
            id='filter-indicador',
            options=[{'label': i, 'value': i} for i in settings.INDICADORES_GPR],
            multi=True,
            placeholder='Seleccionar indicadores',
        ),
    ], id='filter-indicador-container', style={'display': 'none', 'marginBottom': 10}),

    # Botones para mostrar/ocultar datos
    html.Div([
        html.Button("Mostrar/Ocultar Datos PACT2024", id="toggle-data-btn1", n_clicks=0),
        html.Button("Mostrar/Ocultar Datos CZO2", id="toggle-data-btn", n_clicks=0),
    ], style={'marginBottom': 10}),

    # Botones para mostrar/ocultar detalles del indicador
    html.Div([
        html.Button("Detalle del indicador", id="toggle-detail-btn", n_clicks=0),
    ], style={'marginBottom': 10}),

    # Contenedores para DataTables
    html.Div([
        html.Div(id="datatable-container1", style={'display': 'none'}),
        html.Div([
            html.Button('Descargar Datos PACT2024', id='btn_excel1'),
            dcc.Download(id='download-excel1')
        ], id='download-container1', style={'display': 'none'})
    ], style={'marginBottom': 10}),

    html.Div([
        html.Div(id="datatable-container", style={'display': 'none'}),
        html.Div([
            html.Button('Descargar Datos CZO2', id='btn_excel'),
            dcc.Download(id='download-excel')
        ], id='download-container', style={'display': 'none'})
    ], style={'marginBottom': 10}),

    # Stores for keeping the dataframes in the client side for downloading
    dcc.Store(id='stored-df-final'),
    dcc.Store(id='stored-df-pact2024-verificables'),
    dcc.Store(id='stored-df-final-filtered', data={}),
    dcc.Store(id='stored-df-pact2024-verificables-filtered', data={}),

    # Container for Pie Charts
    html.Div(id='pie-chart-container', style={'marginBottom': 10})
])


@app.callback(
    [
        Output('datatable-container1', 'children'),
        Output('datatable-container', 'children'),
        Output('stored-df-final', 'data'),
        Output('stored-df-pact2024-verificables', 'data'),
        Output('stored-df-final-filtered', 'data'),
        Output('stored-df-pact2024-verificables-filtered', 'data')
    ],
    [
        Input('fecha-seleccionada', 'date'),
        Input('filter-indicador', 'value')
    ]
)
def update_data_on_date_and_indicator_selection(selected_date, selected_indicators):
    global df_final, df_verificables, df_pact2024_verificables

    if selected_date:
        df_final = process_data(selected_date)
        df_verificables = verificables(df_final, selected_date)
        df_pact2024_actualizado = actualizar_planificada(procesar_mes_con_fecha(df_pact2024, selected_date))
        df_pact2024_verificables = pac_verificables(df_pact2024_actualizado, df_verificables)

    if selected_indicators:
        df_pact2024_verificables_filtered = df_pact2024_verificables[
            df_pact2024_verificables['INDICADOR_CORTO'].isin(selected_indicators)]
        df_final_filtered = df_final[df_final['INDICADOR_CORTO'].isin(selected_indicators)]
    else:
        df_pact2024_verificables_filtered = df_pact2024_verificables
        df_final_filtered = df_final

    table1 = create_data_table(df_pact2024_verificables_filtered)
    table2 = create_data_table(df_final_filtered)

    return [
        table1,
        table2,
        df_final.to_json(date_format='iso', orient='split'),
        df_pact2024_verificables.to_json(date_format='iso', orient='split'),
        df_final_filtered.to_json(date_format='iso', orient='split'),
        df_pact2024_verificables_filtered.to_json(date_format='iso', orient='split')
    ]


def create_data_table(dataframe):
    return html.Div([
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in dataframe.columns],
            data=dataframe.to_dict('records'),
            style_table={'overflowX': 'auto', 'maxHeight': '1000px'},
            style_cell={'minWidth': '100px', 'width': '150px', 'maxWidth': '300px', 'overflow': 'hidden',
                        'textOverflow': 'ellipsis'},
            style_header={'backgroundColor': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'overflow': 'hidden',
                          'textOverflow': 'ellipsis'},
            editable=True,
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            row_deletable=True,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=100,
            fixed_rows={'headers': True}
        )
    ])


@app.callback(
    [
        Output('datatable-container1', 'style'),
        Output('download-container1', 'style')  # Actualiza esto para controlar el nuevo div de descarga
    ],
    [Input('toggle-data-btn1', 'n_clicks')]
)
def toggle_datatable1_visibility(n_clicks):
    if n_clicks % 2 == 0:
        return {'display': 'none'}, {'display': 'none'}
    else:
        return {'display': 'block'}, {'display': 'block'}


@app.callback(
    [
        Output('datatable-container', 'style'),
        Output('download-container', 'style')  # Actualiza esto para controlar el nuevo div de descarga
    ],
    [Input('toggle-data-btn', 'n_clicks')]
)
def toggle_datatable_visibility(n_clicks):
    if n_clicks % 2 == 0:
        return {'display': 'none'}, {'display': 'none'}
    else:
        return {'display': 'block'}, {'display': 'block'}


@app.callback(
    Output('filter-indicador-container', 'style'),
    Input('toggle-detail-btn', 'n_clicks'),
    State('filter-indicador-container', 'style')
)
def toggle_filter_visibility(n_clicks, current_style):
    if n_clicks % 2 == 0:
        return {'display': 'none', 'marginBottom': 10}
    else:
        return {'display': 'block', 'marginBottom': 10}


# Callback para la descarga del Excel
@app.callback(
    Output("download-excel", "data"),
    [Input("btn_excel", "n_clicks")],
    [State('stored-df-final-filtered', 'data'),
     State('fecha-seleccionada', 'date')],  # Use the stored filtered DataFrame data
    prevent_initial_call=True
)
def download_excel_callback(n_clicks, json_data, selected_date):
    if n_clicks is None:
        raise PreventUpdate
    if json_data is not None:
        # Wrap the JSON string in a StringIO object before reading
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')

        # Formatear la fecha de corte para incluirla en el nombre del archivo
        formatted_date = datetime.strptime(selected_date, "%Y-%m-%d").strftime(
            "%Y%m%d") if selected_date else "sin_fecha"
        filename = f"CZO2_corte_{formatted_date}.xlsx"

        return dcc.send_data_frame(df.to_excel, filename=filename, index=False)


# Callback para la descarga del Excel para PACT2024
@app.callback(
    Output("download-excel1", "data"),
    [Input("btn_excel1", "n_clicks")],
    [State('stored-df-pact2024-verificables-filtered', 'data'),
     State('fecha-seleccionada', 'date')],  # Utiliza la fecha de corte como estado
    prevent_initial_call=True
)
def download_excel_callback1(n_clicks, json_data, selected_date):
    if n_clicks is None:
        raise PreventUpdate
    if json_data is not None:
        # Wrap the JSON string in a StringIO object before reading
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')

        # Formatear la fecha de corte para incluirla en el nombre del archivo
        formatted_date = datetime.strptime(selected_date, "%Y-%m-%d").strftime(
            "%Y%m%d") if selected_date else "sin_fecha"
        filename = f"PACT2024_corte_{formatted_date}.xlsx"

        return dcc.send_data_frame(df.to_excel, filename=filename, index=False)


@app.callback(
    Output('pie-chart-container', 'children'),
    [Input('stored-df-pact2024-verificables-filtered', 'data'),
     Input('toggle-detail-btn', 'n_clicks')],
    [State('fecha-seleccionada', 'date')]
)
def update_pie_charts(json_data, n_clicks, selected_date):
    if json_data is None:
        return []

    buffer = io.StringIO(json_data)
    df = pd.read_json(buffer, orient='split')

    if n_clicks % 2 == 0:
        pie_charts = create_summary_pie_charts(df, selected_date)
        summary_charts = [
            html.Div(dcc.Graph(figure=pie_charts['Global Planificado']),
                     style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(figure=pie_charts['Fecha de Corte']), style={'width': '50%', 'display': 'inline-block'})
        ]
        return summary_charts
    else:
        pie_charts = create_pie_charts_for_indicators(df, selected_date)
        children = []
        row_children = []
        # Organizar los gráficos de torta en filas de dos columnas
        for i, (indicador, (pie_global, pie_corte)) in enumerate(pie_charts.items()):
            row_children.append(
                html.Div(dcc.Graph(figure=pie_global), style={'width': '50%', 'display': 'inline-block'}))
            row_children.append(
                html.Div(dcc.Graph(figure=pie_corte), style={'width': '50%', 'display': 'inline-block'}))
            if i % 2 == 1:  # Cada dos gráficos, comenzar una nueva fila
                children.append(html.Div(row_children, style={'display': 'flex', 'flex-wrap': 'wrap'}))
                row_children = []
        if row_children:  # Añadir cualquier fila que tenga menos de dos gráficos
            children.append(html.Div(row_children, style={'display': 'flex', 'flex-wrap': 'wrap'}))

        return children


if __name__ == '__main__':
    app.run_server(debug=True)
