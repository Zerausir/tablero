import io
import pandas as pd

from django_plotly_dash import DjangoDash
from django.conf import settings
from datetime import date
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .services import process_data, verificables, procesar_mes_con_fecha, pact_2024, pac_verificables
from .utils import calculate_disabled_days_for_year, update_table, download_excel, create_pie_charts, ccde01, ccde02, \
    ccde03, ccde04, \
    ccde11, ccdh01, \
    ccds01, ccds03, ccds05, ccds08, ccds11, ccds12, ccds13, ccds16, ccds17, ccds18, ccds30, ccds31, ccds32, ccdr04

# Inicializa df_pact2024 al comienzo de tu script, justo después de importar tus módulos y funciones necesarias
df_pact2024 = pact_2024()
# Inicializa estos DataFrames con un valor por defecto (vacío) y actualiza solo con callbacks
df_final = pd.DataFrame()
df_verificables = pd.DataFrame()
df_pact2024_verificables = pd.DataFrame()

# Define un diccionario para los calculadores de porcentaje
calculators = {
    'CCDE-01': ccde01,
    'CCDE-02': ccde02,
    'CCDE-03': ccde03,
    'CCDE-04': ccde04,
    'CCDE-11': ccde11,
    'CCDH-01': ccdh01,
    'CCDS-01': ccds01,
    'CCDS-03': ccds03,
    'CCDS-05': ccds05,
    'CCDS-08': ccds08,
    'CCDS-11': ccds11,
    'CCDS-12': ccds12,
    'CCDS-13': ccds13,
    'CCDS-16': ccds16,
    'CCDS-17': ccds17,
    'CCDS-18': ccds18,
    'CCDS-30': ccds30,
    'CCDS-31': ccds31,
    'CCDS-32': ccds32,
    'CCDR-04': ccdr04
    # Continúa según sea necesario
}

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
    ], style={'marginBottom': 10}),

    # Botones para mostrar/ocultar datos
    html.Div([
        html.Button("Mostrar/Ocultar Datos PACT2024", id="toggle-data-btn1", n_clicks=0),
        html.Button("Mostrar/Ocultar Datos CZO2", id="toggle-data-btn", n_clicks=0),
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

    html.Div(id='pie-charts-container', children=create_pie_charts(df_final, calculators)),

    # Este contenedor se llenará con la tabla correspondiente al gráfico de pie seleccionado
    html.Div(id='verificables-details-container', style={'display': 'none'}),

    # Stores for keeping the dataframes in the client side for downloading
    dcc.Store(id='stored-df-final'),
    dcc.Store(id='stored-df-pact2024-verificables'),
])


@app.callback(
    [
        Output('datatable-container1', 'children'),
        Output('datatable-container', 'children'),
        Output('stored-df-final', 'data'),
        Output('stored-df-pact2024-verificables', 'data')
    ],
    [
        Input('fecha-seleccionada', 'date'),
        Input('filter-indicador', 'value')
    ]
)
def update_data_on_date_and_indicator_selection(selected_date, selected_indicators):
    global df_final, df_verificables, df_pact2024_verificables

    # Verifica si realmente hay un cambio en la fecha para procesar los datos
    if selected_date:
        df_final = process_data(selected_date)
        df_verificables = verificables(df_final, selected_date)
        df_pact2024_actualizado = procesar_mes_con_fecha(df_pact2024, selected_date)
        df_pact2024_verificables = pac_verificables(df_pact2024_actualizado, df_verificables)

    # Filtra los DataFrames según los indicadores seleccionados
    if selected_indicators:
        df_pact2024_verificables = df_pact2024_verificables[
            df_pact2024_verificables['INDICADOR_CORTO'].isin(selected_indicators)]
        df_final_filtered = df_final[df_final['INDICADOR_CORTO'].isin(selected_indicators)]
    else:
        df_final_filtered = df_final

    # Crea las DataTables para mostrar los DataFrames filtrados
    table1 = create_data_table(df_pact2024_verificables)
    table2 = create_data_table(df_final_filtered)

    return [table1, table2, df_final.to_json(date_format='iso', orient='split'),
            df_pact2024_verificables.to_json(date_format='iso', orient='split')]


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
    Output('pie-charts-container', 'children'),
    [
        Input('fecha-seleccionada', 'date'),
        Input('filter-indicador', 'value')
    ]
)
def update_pie_charts(selected_date, selected_indicators):
    # Recargar los datos basados en la fecha seleccionada
    df_final = process_data(selected_date)

    # Filtrar los datos si se han seleccionado indicadores
    if selected_indicators:
        df_filtered = df_final[df_final['INDICADOR_CORTO'].isin(selected_indicators)]
    else:
        df_filtered = df_final

    # Pasar los datos filtrados a create_pie_charts
    return create_pie_charts(df_filtered, calculators, selected_indicators)


# Callback para actualizar la DataTable basado en los filtros seleccionados
@app.callback(
    Output('table1', 'data'),
    [Input('filter-indicador', 'value')]
)
def update_table1(selected_indicador):
    return update_table(df_final, selected_indicador, 'table1')


@app.callback(
    Output('table2', 'data'),
    [Input('filter-indicador', 'value')]
)
def update_table2(selected_indicador):
    return update_table(df_pact2024_verificables, selected_indicador, 'table2')


# Callback para la descarga del Excel
@app.callback(
    Output("download-excel", "data"),
    [Input("btn_excel", "n_clicks")],
    [State('stored-df-final', 'data')],  # Use the stored DataFrame data
    prevent_initial_call=True
)
def download_excel_callback(n_clicks, json_data):
    if n_clicks is None:
        raise PreventUpdate
    if json_data is not None:
        # Wrap the JSON string in a StringIO object before reading
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')
        return dcc.send_data_frame(df.to_excel, filename="datos_descargados.xlsx", index=False)


@app.callback(
    Output("download-excel1", "data"),
    [Input("btn_excel1", "n_clicks")],
    [State('stored-df-pact2024-verificables', 'data')],  # Use the stored DataFrame data
    prevent_initial_call=True
)
def download_excel_callback1(n_clicks, json_data):
    if n_clicks is None:
        raise PreventUpdate
    if json_data is not None:
        # Wrap the JSON string in a StringIO object before reading
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')
        return dcc.send_data_frame(df.to_excel, filename="datos_descargados.xlsx", index=False)


if __name__ == '__main__':
    app.run_server(debug=True)
