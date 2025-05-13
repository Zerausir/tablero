import io
import pandas as pd
import math
import os

from django_plotly_dash import DjangoDash
from datetime import date, datetime
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .services import process_data, verificables, procesar_mes_con_fecha, actualizar_planificada, pact_data, \
    pac_verificables, load_environment_variables
from .utils import calculate_disabled_days_for_year, create_pie_charts_for_indicators, create_summary_pie_charts

app = DjangoDash(
    name='GprApp',
    add_bootstrap_links=True,
    external_stylesheets=["/static/css/inner.css"]
)

app.layout = html.Div(children=[
    # Selector de año
    html.Div([
        html.Label("Año:"),
        dcc.Dropdown(
            id='year-selector',
            options=[
                {'label': '2024', 'value': 2024},
                {'label': '2025', 'value': 2025}
            ],
            value=2025,  # Valor por defecto
            clearable=False
        ),
    ], style={'marginBottom': 10, 'marginTop': 10}),

    # Filtros Dropdown
    html.Div([
        html.Label("Fecha de corte:"),
        dcc.DatePickerSingle(
            id='fecha-seleccionada',
            min_date_allowed=date(2024, 1, 1),
            max_date_allowed=date(2025, 12, 31),
            initial_visible_month=date(2025, 1, 1),
            date=date(2025, 1, 31),  # Asegúrate de que esta sea una fecha válida según tu lógica
            disabled_days=calculate_disabled_days_for_year(2025)
        ),
    ], style={'marginBottom': 10, 'marginTop': 10}),

    # Filtro de indicadores
    html.Div([
        html.Label("Filtrar por indicador:"),
        dcc.Dropdown(
            id='filter-indicador',
            options=[],  # Vacío inicialmente, se llenará con el callback
            multi=True,
            placeholder='Seleccionar indicadores',
        ),
    ], id='filter-indicador-container', style={'display': 'none', 'marginBottom': 10}),

    # Botones para mostrar/ocultar datos
    html.Div([
        html.Button("Mostrar/Ocultar Datos PACT", id="toggle-data-btn1", n_clicks=0),
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
            html.Button('Descargar Datos PACT', id='btn_excel1'),
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

    html.Div([
        html.Button('Anexo 3.1', id='btn_anexo31'),
        dcc.Download(id='download-anexo31')
    ], id='download-container-anexo31', style={'marginBottom': 10}),

    # Stores for keeping the dataframes in the client side for downloading
    dcc.Store(id='stored-df-final'),
    dcc.Store(id='stored-df-pact-verificables'),
    dcc.Store(id='stored-df-final-filtered', data={}),
    dcc.Store(id='stored-df-pact-verificables-filtered', data={}),
    dcc.Store(id='stored-selected-year', data=2025),  # Almacena el año seleccionado

    # Container for Pie Charts
    html.Div(id='pie-chart-container', style={'marginBottom': 10}),

    # Interval component for periodic updates
    dcc.Interval(
        id='interval-component',
        interval=30 * 1000,  # in milliseconds (e.g., 20*1000 = 20 seconds)
        n_intervals=0
    )
])


# Callback para actualizar el datepicker según el año seleccionado
@app.callback(
    [
        Output('fecha-seleccionada', 'min_date_allowed'),
        Output('fecha-seleccionada', 'max_date_allowed'),
        Output('fecha-seleccionada', 'initial_visible_month'),
        Output('fecha-seleccionada', 'date'),
        Output('fecha-seleccionada', 'disabled_days'),
        Output('stored-selected-year', 'data')
    ],
    [Input('year-selector', 'value')]
)
def update_date_picker(selected_year):
    min_date = date(selected_year, 1, 1)
    max_date = date(selected_year, 12, 31)
    initial_date = date(selected_year, 1, 31)  # Primer mes por defecto
    disabled_days = calculate_disabled_days_for_year(selected_year)

    return min_date, max_date, min_date, initial_date, disabled_days, selected_year


# Callback para actualizar las opciones del dropdown de indicadores
@app.callback(
    Output('filter-indicador', 'options'),
    [Input('stored-df-pact-verificables', 'data')]
)
def update_indicator_dropdown(json_data):
    if json_data:
        # Convertir los datos JSON a DataFrame
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')

        # Obtener los indicadores únicos presentes en los datos
        indicadores_disponibles = df['INDICADOR_CORTO'].unique().tolist()
        indicadores_disponibles.sort()  # Ordenar alfabéticamente

        # Crear las opciones para el dropdown
        options = [{'label': i, 'value': i} for i in indicadores_disponibles]
        return options

    # Si no hay datos disponibles, retornar una lista vacía
    return []


@app.callback(
    [
        Output('datatable-container1', 'children'),
        Output('datatable-container', 'children'),
        Output('stored-df-final', 'data'),
        Output('stored-df-pact-verificables', 'data'),
        Output('stored-df-final-filtered', 'data'),
        Output('stored-df-pact-verificables-filtered', 'data')
    ],
    [
        Input('fecha-seleccionada', 'date'),
        Input('filter-indicador', 'value'),
        Input('interval-component', 'n_intervals'),
        Input('stored-selected-year', 'data')
    ]
)
def update_data_on_date_and_indicator_selection(selected_date, selected_indicators, n_intervals, selected_year):
    # Utilizar el año seleccionado para cargar los datos correspondientes
    df_pact = pact_data(selected_date, selected_year)

    if selected_date:
        df_final = process_data(selected_date, selected_year)
        df_verificables = verificables(df_final, selected_date)
        df_pact_actualizado = actualizar_planificada(procesar_mes_con_fecha(df_pact, selected_date))
        df_pact_verificables = pac_verificables(df_pact_actualizado, df_verificables)

    if selected_indicators:
        df_pact_verificables_filtered = df_pact_verificables[
            df_pact_verificables['INDICADOR_CORTO'].isin(selected_indicators)]
        df_final_filtered = df_final[df_final['INDICADOR_CORTO'].isin(selected_indicators)]
    else:
        df_pact_verificables_filtered = df_pact_verificables
        df_final_filtered = df_final

    table1 = create_data_table(df_pact_verificables_filtered)
    table2 = create_data_table(df_final_filtered)

    return [
        table1,
        table2,
        df_final.to_json(date_format='iso', orient='split'),
        df_pact_verificables.to_json(date_format='iso', orient='split'),
        df_final_filtered.to_json(date_format='iso', orient='split'),
        df_pact_verificables_filtered.to_json(date_format='iso', orient='split')
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
     State('fecha-seleccionada', 'date'),
     State('stored-selected-year', 'data')],  # Añadido el año seleccionado
    prevent_initial_call=True
)
def download_excel_callback(n_clicks, json_data, selected_date, selected_year):
    if n_clicks is None:
        raise PreventUpdate
    if json_data is not None:
        # Wrap the JSON string in a StringIO object before reading
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')

        # Formatear la fecha de corte para incluirla en el nombre del archivo
        formatted_date = datetime.strptime(selected_date, "%Y-%m-%d").strftime(
            "%Y%m%d") if selected_date else "sin_fecha"
        filename = f"CZO2_{selected_year}_corte_{formatted_date}.xlsx"

        return dcc.send_data_frame(df.to_excel, filename=filename, index=False)
    return None


# Callback para la descarga del Excel para PACT
@app.callback(
    Output("download-excel1", "data"),
    [Input("btn_excel1", "n_clicks")],
    [State('stored-df-pact-verificables-filtered', 'data'),
     State('fecha-seleccionada', 'date'),
     State('stored-selected-year', 'data')],  # Añadido el año seleccionado
    prevent_initial_call=True
)
def download_excel_callback1(n_clicks, json_data, selected_date, selected_year):
    if n_clicks is None:
        raise PreventUpdate
    if json_data is not None:
        # Wrap the JSON string in a StringIO object before reading
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')

        # Formatear la fecha de corte para incluirla en el nombre del archivo
        formatted_date = datetime.strptime(selected_date, "%Y-%m-%d").strftime(
            "%Y%m%d") if selected_date else "sin_fecha"
        filename = f"PACT{selected_year}_corte_{formatted_date}.xlsx"

        return dcc.send_data_frame(df.to_excel, filename=filename, index=False)
    return None


@app.callback(
    Output("download-anexo31", "data"),
    [Input("btn_anexo31", "n_clicks")],
    [State('stored-df-pact-verificables-filtered', 'data'),
     State('stored-df-final-filtered', 'data'),
     State('fecha-seleccionada', 'date'),
     State('stored-selected-year', 'data')],
    prevent_initial_call=True
)
def download_anexo31_callback(n_clicks, json_data_pact, json_data_verificables, selected_date, selected_year):
    if n_clicks is None:
        raise PreventUpdate

    if json_data_pact is not None and json_data_verificables is not None:
        # Convertir los datos JSON a DataFrames
        df_pact = pd.read_json(io.StringIO(json_data_pact), orient='split')
        df_verificables = pd.read_json(io.StringIO(json_data_verificables), orient='split')

        # Diccionario de meses en español
        meses_esp = {
            1: "ENERO", 2: "FEBRERO", 3: "MARZO", 4: "ABRIL",
            5: "MAYO", 6: "JUNIO", 7: "JULIO", 8: "AGOSTO",
            9: "SEPTIEMBRE", 10: "OCTUBRE", 11: "NOVIEMBRE", 12: "DICIEMBRE"
        }

        # Crear el archivo Excel temporal
        fecha_dt = datetime.strptime(selected_date, "%Y-%m-%d")
        mes = meses_esp[fecha_dt.month]
        formatted_date = fecha_dt.strftime("%Y%m%d")
        filename = f"Anexo_3.1_{selected_year}_corte_{formatted_date}.xlsx"

        # Cargar variables de entorno para obtener la ruta de CCDE-10
        env_vars = load_environment_variables()
        ruta_ccde10 = env_vars.get('RUTA_CCDE_10_2025', '') if selected_year == 2025 else env_vars.get('RUTA_CCDE_10',
                                                                                                       '')

        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book

            # ====== PRIMERA HOJA: ANEXO 3.1 ======
            worksheet = workbook.add_worksheet('Anexo 3.1')

            # Formatos para el Excel
            title_format = workbook.add_format({
                'bold': True,
                'align': 'center',
                'valign': 'vcenter'
            })

            percent_format = workbook.add_format({
                'num_format': '0.00%',
                'align': 'center',
                'border': 1  # Añadir borde a todas las celdas
            })

            na_format = workbook.add_format({
                'align': 'center',
                'border': 1,
                'font_color': '#808080',  # Gris para N/A
                'pattern': 1,  # Patrón de fondo
                'bg_color': '#F2F2F2'  # Color de fondo gris claro
            })

            cell_format = workbook.add_format({
                'border': 1,  # Añadir borde a todas las celdas
                'align': 'center',
                'valign': 'vcenter'
            })

            header_format = workbook.add_format({
                'bold': True,
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })

            wrap_format = workbook.add_format({
                'text_wrap': True,
                'border': 1,
                'align': 'left',
                'valign': 'top'
            })

            note_format = workbook.add_format({
                'font_color': '#1F497D',  # Azul oscuro
                'italic': True,
                'align': 'left',
                'valign': 'top',
                'text_wrap': True
            })

            # Escribir los títulos
            worksheet.merge_range('A1:F1', 'ANEXO', title_format)
            worksheet.merge_range('A2:F2', '3.1 Porcentaje de cumplimiento PACT', title_format)
            worksheet.merge_range('A3:F3',
                                  f'RESUMEN RESULTADOS DE CUMPLIMIENTO DE INDICADORES PACT-{selected_year} A {mes} DE {selected_year}',
                                  title_format)

            # Columnas para la tabla de Excel
            col_total = f'TOTAL A CUMPLIR A {mes} DE {selected_year}'
            col_actividades = f'ACTIVIDADES CUMPLIDAS A {mes} DE {selected_year}'

            # Preparar los datos para el Anexo 3.1
            anexo_data = []
            for index, row in df_pact.iterrows():
                indicador_corto = row['INDICADOR_CORTO']
                verificables_df = df_verificables[df_verificables['INDICADOR_CORTO'] == indicador_corto]

                # Crear lista numerada de informes
                informes = ""

                # Manejo especial para CCDE-10
                if indicador_corto == 'CCDE-10' and ruta_ccde10 and os.path.exists(ruta_ccde10):
                    # Obtener todos los archivos en la ruta CCDE-10 (no solo PDFs)
                    fecha_limite = fecha_dt + pd.Timedelta(days=11)  # 11 días después de la fecha seleccionada
                    ccde10_files = []

                    for file in os.listdir(ruta_ccde10):
                        file_path = os.path.join(ruta_ccde10, file)
                        if os.path.isfile(file_path):
                            mod_time = os.path.getmtime(file_path)
                            mod_date = pd.to_datetime(mod_time, unit='s')
                            if mod_date <= fecha_limite:
                                ccde10_files.append(file)

                    # Ordenar alfabéticamente los archivos
                    ccde10_files.sort()

                    # Crear lista numerada de los archivos de CCDE-10
                    if ccde10_files:
                        informes = "ARCHIVOS:\n" + '\n'.join(f"{i + 1}. {file}" for i, file in enumerate(ccde10_files))
                else:
                    # Para el resto de indicadores, usar el método normal
                    informes_list = verificables_df['Nro. INFORME'].astype(str).tolist()
                    if informes_list:
                        informes = "INFORMES:\n" + '\n'.join(
                            f"{i + 1}. {informe}" for i, informe in enumerate(informes_list))

                # MODIFICACIÓN: Determinar valores a mostrar de manera consistente con los gráficos
                # Caso especial para CCDR-01
                if indicador_corto == 'CCDR-01':
                    # Para CCDR-01, el valor "a cumplir al corte" debe ser igual al valor planificado
                    total_a_cumplir = math.ceil(row['PLANIFICADA_META']) if pd.notnull(row['PLANIFICADA_META']) else 0
                # Caso para indicadores especiales que usan Nro_INSPECCIONES
                elif indicador_corto in ['CCDE-10', 'CCDS-01', 'CCDS-13', 'CCDR-06']:
                    # Usar el valor CUMPLIR_META normal para estos indicadores
                    total_a_cumplir = math.ceil(row['CUMPLIR_META']) if pd.notnull(row['CUMPLIR_META']) else 0
                    actividades_cumplidas = row['Nro_INSPECCIONES'] if pd.notnull(row.get('Nro_INSPECCIONES')) else 0
                else:
                    # Para el resto de indicadores, usar CUMPLIR_META normal
                    total_a_cumplir = math.ceil(row['CUMPLIR_META']) if pd.notnull(row['CUMPLIR_META']) else 0

                # Determinar actividades cumplidas según el tipo de indicador
                if indicador_corto == 'CCDE-10' and ruta_ccde10 and os.path.exists(ruta_ccde10):
                    # Para CCDE-10, contar los archivos en la ruta
                    fecha_limite = fecha_dt + pd.Timedelta(days=11)
                    actividades_cumplidas = sum(1 for file in os.listdir(ruta_ccde10)
                                                if os.path.isfile(os.path.join(ruta_ccde10, file)) and
                                                pd.to_datetime(os.path.getmtime(os.path.join(ruta_ccde10, file)),
                                                               unit='s') <= fecha_limite)
                elif indicador_corto in ['CCDS-01', 'CCDS-13', 'CCDR-06']:
                    # Para estos indicadores especiales, usar el valor de Nro_INSPECCIONES
                    actividades_cumplidas = row.get('Nro_INSPECCIONES', 0)
                    if pd.isna(actividades_cumplidas):
                        # Si el valor es NaN, intentar obtenerlo del DataFrame de verificables
                        if 'Nro_INSPECCIONES' in verificables_df.columns and not verificables_df.empty:
                            actividades_cumplidas = verificables_df['Nro_INSPECCIONES'].sum()
                        else:
                            actividades_cumplidas = 0
                else:
                    # Para indicadores normales, usar CANTIDAD_VERIFICABLES
                    actividades_cumplidas = verificables_df.shape[0] if not verificables_df.empty else 0

                # Asegurarse de que actividades_cumplidas sea un número
                if pd.isna(actividades_cumplidas):
                    actividades_cumplidas = 0

                anexo_data.append({
                    'Nro.': index + 1,
                    'INDICADOR PACT': row['INDICADOR'],
                    'INDICADOR_CORTO': indicador_corto,  # Añadimos esta columna para uso interno
                    'TIPO': row.get('TIPO', ''),  # Añadimos esta columna para uso interno
                    col_total: math.ceil(total_a_cumplir),
                    col_actividades: math.ceil(actividades_cumplidas),
                    'VERIFICABLES': informes,
                    'TIENE_META': total_a_cumplir > 0  # Campo adicional para saber si tiene meta
                })

            # Crear el DataFrame
            df_anexo = pd.DataFrame(anexo_data)

            # Escribir encabezados manualmente para poder aplicar formato
            headers = ['Nro.', 'INDICADOR PACT', col_total, col_actividades, 'VERIFICABLES', 'PORCENTAJE (MAX 100%)']

            for col_idx, header in enumerate(headers):
                worksheet.write(3, col_idx, header, header_format)

            # Escribir datos con formato de borde
            for row_idx, row_data in enumerate(df_anexo.iterrows()):
                idx, row = row_data  # Desempaquetar el índice y los datos de la fila

                # Obtener los valores de las columnas necesarias
                valores = [
                    row['Nro.'],
                    row['INDICADOR PACT'],
                    row[col_total],
                    row[col_actividades],
                    row['VERIFICABLES']
                ]

                # Escribir las primeras 5 columnas
                for col_idx, cell_value in enumerate(valores):
                    if col_idx == 4:  # Columna VERIFICABLES
                        worksheet.write(row_idx + 4, col_idx, cell_value, wrap_format)
                    else:
                        worksheet.write(row_idx + 4, col_idx, cell_value, cell_format)

                # Para la columna de porcentaje (columna F, índice 5)
                if row['TIENE_META']:
                    # Si tiene meta, escribir la fórmula de porcentaje
                    formula = f'=MIN(IF(C{row_idx + 5}=0,1,IF(D{row_idx + 5}=0,0,D{row_idx + 5}/C{row_idx + 5})),1)'
                    worksheet.write_formula(row_idx + 4, 5, formula, percent_format)
                else:
                    # Si no tiene meta, mostrar "N/A"
                    worksheet.write(row_idx + 4, 5, "N/A", na_format)

            # Número de filas de datos
            num_rows = len(df_anexo) + 4  # +4 por las filas de encabezado

            # Añadir fórmula para el promedio inmediatamente después de la tabla
            worksheet.write(num_rows, 4, "Promedio General:", header_format)

            # Fórmula para el promedio
            promedio_formula = f'=AVERAGEIF(F5:F{num_rows},"<>N/A")'
            worksheet.write_formula(num_rows, 5, promedio_formula, percent_format)

            # Nota explicativa sobre N/A en la siguiente fila después del promedio
            # SOLO UNA VEZ para evitar duplicación
            nota_texto = "Nota: 'N/A' indica que no hay actividades planificadas para este indicador en el período de corte."
            worksheet.merge_range(f'A{num_rows + 1}:F{num_rows + 1}', nota_texto, note_format)

            # Inmovilizar paneles en A5
            worksheet.freeze_panes(4, 0)  # Filas 0-3 (A1-A4) serán visibles al desplazarse

            # Ajustar el ancho de las columnas según las especificaciones
            # Convertir píxeles a unidades de Excel (aproximadamente 7 píxeles = 1 unidad)
            worksheet.set_column('A:A', 34 / 7)  # 34 píxeles
            worksheet.set_column('B:B', 284 / 7)  # 284 píxeles
            worksheet.set_column('C:C', 234 / 7)  # 234 píxeles
            worksheet.set_column('D:D', 283 / 7)  # 283 píxeles
            worksheet.set_column('E:E', 157 / 7)  # 157 píxeles
            worksheet.set_column('F:F', 166 / 7)  # 166 píxeles

            # ====== SEGUNDA HOJA: PORCENTAJE DE CUMPLIMIENTO ======
            worksheet2 = workbook.add_worksheet('Porcentaje de Cumplimiento')

            # Formatos adicionales para la segunda hoja
            subtitle_format = workbook.add_format({
                'bold': True,
                'align': 'left',
                'valign': 'vcenter',
                'font_size': 12
            })

            explanation_format = workbook.add_format({
                'align': 'left',
                'valign': 'vcenter',
                'text_wrap': True
            })

            highlight_format = workbook.add_format({
                'bold': True,
                'align': 'center',
                'valign': 'vcenter',
                'font_size': 14,
                'num_format': '0.00%',
                'border': 1,
                'bg_color': '#D9E1F2'  # Color de fondo celeste claro
            })

            data_header_format = workbook.add_format({
                'bold': True,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'bg_color': '#E2EFDA'  # Color de fondo verde claro
            })

            data_cell_format = workbook.add_format({
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })

            excluded_header_format = workbook.add_format({
                'bold': True,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'bg_color': '#FFF2CC'  # Color de fondo amarillo claro
            })

            excluded_cell_format = workbook.add_format({
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'bg_color': '#FFF2CC',  # Color de fondo amarillo claro
                'font_color': '#808080'  # Texto gris
            })

            # Títulos
            worksheet2.merge_range('A1:F1',
                                   f'Porcentaje de Cumplimiento PACT {selected_year} al corte {mes} {selected_year}',
                                   title_format)
            worksheet2.write('A3', 'Explicación del cálculo:', subtitle_format)

            # Explicación mejorada
            explanation_text = (
                "El Porcentaje de Cumplimiento PACT se calcula como el promedio de los porcentajes "
                "individuales de cumplimiento de cada indicador con actividades planificadas en el período. "
                "Para cada indicador aplicable, se divide las actividades cumplidas entre el total a cumplir al corte "
                "(limitado a un máximo del 100%). Los indicadores sin actividades planificadas al corte "
                "no se incluyen en el cálculo del promedio."
            )
            worksheet2.merge_range('A4:F4', explanation_text, explanation_format)

            # Establecer altura de la fila 4 (68 píxeles)
            worksheet2.set_row(3, 68)  # La fila 4 es índice 3 (0-indexed)

            # Sección de datos detallados
            worksheet2.write('A6', 'Detalles del cálculo por indicador:', subtitle_format)

            # Encabezados de la tabla de cálculo
            headers2 = [
                'INDICADOR', 'TIPO', 'ACTIVIDADES CUMPLIDAS', 'TOTAL A CUMPLIR',
                'PORCENTAJE INDICADOR', 'CONTRIBUCIÓN AL PROMEDIO'
            ]

            for col_idx, header in enumerate(headers2):
                worksheet2.write(7, col_idx, header, data_header_format)

            # Separar indicadores con y sin meta
            indicadores_con_meta = df_anexo[df_anexo['TIENE_META']].copy()
            indicadores_sin_meta = df_anexo[~df_anexo['TIENE_META']].copy()

            # Llenar la tabla con los datos y fórmulas para indicadores CON meta
            fila_inicial = 8

            # Verificar si hay indicadores con meta antes de procesar
            if not indicadores_con_meta.empty:
                for row_idx, (_, row_data) in enumerate(indicadores_con_meta.iterrows()):
                    fila_actual = fila_inicial + row_idx

                    # Indicador y tipo
                    worksheet2.write(fila_actual, 0, row_data['INDICADOR_CORTO'], data_cell_format)
                    worksheet2.write(fila_actual, 1, row_data['TIPO'], data_cell_format)

                    # Actividades cumplidas
                    worksheet2.write(fila_actual, 2, row_data[col_actividades], data_cell_format)

                    # Total a cumplir
                    worksheet2.write(fila_actual, 3, row_data[col_total], data_cell_format)

                    # PORCENTAJE INDICADOR - Usar la fórmula especificada
                    # =MIN(SI(D9=0;1;SI(C9=0;0;C9/D9));1)
                    porcentaje_formula = f'=MIN(IF(D{fila_actual + 1}=0,1,IF(C{fila_actual + 1}=0,0,C{fila_actual + 1}/D{fila_actual + 1})),1)'
                    worksheet2.write_formula(fila_actual, 4, porcentaje_formula, percent_format)

                    # CONTRIBUCIÓN AL PROMEDIO - Formula
                    contribucion_formula = f'=E{fila_actual + 1}/COUNTIF(E{fila_inicial + 1}:E{fila_inicial + len(indicadores_con_meta)},"<>N/A")'
                    worksheet2.write_formula(fila_actual, 5, contribucion_formula, percent_format)

                # Calcular y mostrar el porcentaje final
                total_con_meta = len(indicadores_con_meta)
                worksheet2.write(f'A{fila_inicial + total_con_meta + 2}', 'Porcentaje de Cumplimiento PACT:',
                                 subtitle_format)

                # Primero haz el merge_range
                worksheet2.merge_range(f'B{fila_inicial + total_con_meta + 2}:D{fila_inicial + total_con_meta + 2}',
                                       '', highlight_format)

                # Luego escribe la fórmula en la celda B
                promedio_formula = f'=AVERAGE(E{fila_inicial + 1}:E{fila_inicial + total_con_meta})'
                worksheet2.write_formula(f'B{fila_inicial + total_con_meta + 2}', promedio_formula, highlight_format)

            # Mostrar sección de indicadores excluidos del cálculo
            if not indicadores_sin_meta.empty:
                fila_excluidos = fila_inicial + (0 if indicadores_con_meta.empty else len(indicadores_con_meta)) + 4
                worksheet2.write(f'A{fila_excluidos}',
                                 'Indicadores excluidos del cálculo (sin actividades planificadas al corte):',
                                 subtitle_format)

                # Encabezado para indicadores excluidos
                for col_idx, header in enumerate(headers2[:5]):  # Solo usamos las primeras 5 columnas
                    worksheet2.write(fila_excluidos + 1, col_idx, header, excluded_header_format)

                # Listar indicadores excluidos
                for row_idx, (_, row_data) in enumerate(indicadores_sin_meta.iterrows()):
                    fila_actual = fila_excluidos + 2 + row_idx

                    # Indicador y tipo
                    worksheet2.write(fila_actual, 0, row_data['INDICADOR_CORTO'], excluded_cell_format)
                    worksheet2.write(fila_actual, 1, row_data['TIPO'], excluded_cell_format)

                    # Actividades cumplidas y Total a cumplir (valores directos)
                    worksheet2.write(fila_actual, 2, row_data[col_actividades], excluded_cell_format)
                    worksheet2.write(fila_actual, 3, 0, excluded_cell_format)  # Total a cumplir siempre es 0

                    # Porcentaje (N/A)
                    worksheet2.write(fila_actual, 4, "N/A", excluded_cell_format)

            # Establecer anchos de columna para mejor visualización
            worksheet2.set_column('A:A', 20)  # Indicador
            worksheet2.set_column('B:B', 15)  # Tipo
            worksheet2.set_column('C:C', 25)  # Actividades cumplidas
            worksheet2.set_column('D:D', 20)  # Total a cumplir
            worksheet2.set_column('E:E', 162 / 7)  # Porcentaje indicador (162 píxeles)
            worksheet2.set_column('F:F', 197 / 7)  # Contribución al promedio (197 píxeles)

        # Leer el archivo guardado y enviarlo
        with open(filename, 'rb') as f:
            return dcc.send_file(filename)
    return None


@app.callback(
    Output('pie-chart-container', 'children'),
    [Input('stored-df-pact-verificables-filtered', 'data'),
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
