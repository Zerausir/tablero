import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
from dash import dcc, dash_table, html
from sklearn.metrics import auc


def convert_end_date(date_time: str) -> datetime.datetime:
    """
    Convert end date string to datetime object with time set to '23:59:59'.

    Args:
        date_time (str): End date string.

    Returns:
        datetime.datetime: Corresponding datetime object.
    """
    datetime_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d')
    datetime_obj = datetime_obj.replace(hour=23, minute=59, second=59)
    return datetime_obj


def convert_start_date(date_time: str) -> datetime.datetime:
    """
    Convert start date string to datetime object with time set to '00:00:01'.

    Args:
        date_time (str): Start date string.

    Returns:
        datetime.datetime: Corresponding datetime object.
    """
    datetime_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d')
    datetime_obj = datetime_obj.replace(hour=0, minute=0, second=1)
    return datetime_obj


def convert_timestamps_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    date_columns = ['tiempo']
    date_format = '%Y-%m-%d'  # Replace with your actual date format

    for col in date_columns:
        if col in df.columns:
            # Convert column to datetime with specified format, coercing errors to NaT
            df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
            # Format datetime as string, ignoring NaT (which become NaT again)
            df[col] = df[col].dt.strftime(date_format).replace('NaT', '-')

    # Handle MultiIndex for columns and index, if applicable
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(
            lambda x: [(y.strftime(date_format) if isinstance(y, pd.Timestamp) else y) for y in x])
    else:
        df.columns = [x.strftime(date_format) if isinstance(x, pd.Timestamp) else x for x in df.columns]

    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.map(lambda x: [(y.strftime(date_format) if isinstance(y, pd.Timestamp) else y) for y in x])
    else:
        df.index = [x.strftime(date_format) if isinstance(x, pd.Timestamp) else x for x in df.index]

    return df


def create_dash_datatable(table_id: str, style: dict = {}) -> dash_table.DataTable:
    """
    Create a Dash DataTable with a given table ID and style.

    Args:
        table_id (str): Identifier for the Dash DataTable.
        style (dict, optional): Style for the Dash DataTable.

    Returns:
        dash_table.DataTable: A Dash DataTable component.
    """
    return dash_table.DataTable(
        id=table_id,
        style_table={'overflowX': 'auto', 'maxHeight': '300px', **style},
        style_cell={  # Default cell style
            'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_header={  # Styling for the header to ensure it's consistent with the body
            'backgroundColor': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
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
        fixed_rows={'headers': True},  # This line fixes the header row at the top
    )


def create_heatmap_data(df: pd.DataFrame, value_column: str = 'level_dbuv_m', title: str = '',
                        x_range=None) -> go.Figure:
    """
    Create heatmap data from a DataFrame for any specified parameter.

    Args:
        df (pd.DataFrame): The DataFrame from which the heatmap data is to be created.
        value_column (str): The column to use for the heatmap values.
        title (str): The title for the heatmap.
        x_range (list, optional): The range of the x-axis.

    Returns:
        go.Figure: A Plotly Heatmap figure.
    """
    if df.empty:
        return go.Figure()

    df = df.fillna(0)
    df['tiempo'] = pd.to_datetime(df['tiempo'])

    # Asegurar que la columna sea numérica
    df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

    heatmap_data = df.pivot_table(
        values=value_column,
        index='tiempo',
        columns='frecuencia_hz',
        aggfunc='mean'
    )

    # Definir rangos de color según el parámetro
    if value_column == 'level_dbuv_m':
        zmin, zmax = 0, 100
    elif value_column == 'offset_hz':
        zmin, zmax = df[value_column].min(), df[value_column].max()
    elif value_column == 'modulation_value':
        zmin, zmax = 0, 100
    else:
        zmin, zmax = df[value_column].min(), df[value_column].max()

    layout = go.Layout(
        title=title,
        xaxis={'title': 'Frecuencia (Hz)'},
        yaxis={'title': 'Tiempo', 'tickfont': {'size': 11}},
        margin=dict(l=112)
    )

    if x_range is not None:
        layout['xaxis']['range'] = x_range

    return go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='rainbow',
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(orientation='h', y=1)
        ),
        layout=layout
    )


def create_scatter_plot(df: pd.DataFrame, x_range=None, threshold=None) -> go.Figure:
    """
    Create a scatter plot from a DataFrame. The DataFrame is expected to have 'occupation_percentage'
    and 'frecuencia_hz' columns. If the DataFrame is empty, an empty figure is returned.

    Args:
        df (pd.DataFrame): The DataFrame from which the scatter plot is to be created.
        x_range (list, optional): The range of the x-axis. If None, the range is determined from the data.

    Returns:
        go.Figure: A Plotly Scatter figure.
    """
    if df.empty:
        return go.Figure()

    title = 'Porcentaje de Ocupación vs Frecuencia'
    if threshold is not None:
        title += f' (Umbral = {threshold} dBµV/m)'

    # Sort the dataframe by 'frecuencia_hz'
    df = df.sort_values(by='frecuencia_hz')

    # Calculate AUC
    auc_value = auc(df['frecuencia_hz'], df['occupation_percentage'])

    # Calculate total rectangle area
    total_area = (df['frecuencia_hz'].max() - df['frecuencia_hz'].min()) * 100

    # Calculate band occupation percentage
    band_occupation_percentage = (auc_value / total_area) * 100

    title += f' - Porcentaje de Ocupación de la Banda = {band_occupation_percentage:.2f}%'

    layout = go.Layout(
        title=title,
        xaxis={'title': 'Frecuencia (Hz)'},
        yaxis={'title': 'Porcentaje de Ocupación (%)', 'range': [0, 100]},
        margin=dict(l=112)
    )

    if x_range is not None:
        layout['xaxis']['range'] = x_range

    return go.Figure(
        data=go.Scatter(
            x=df['frecuencia_hz'],
            y=df['occupation_percentage'],
            mode='markers',
            marker=dict(
                size=10,
                color='rgba(152, 0, 0, .8)',
                line=dict(
                    width=2,
                    color='rgb(0, 0, 0)'
                )
            )
        ),
        layout=layout
    )


def create_heatmap_layout(df_original):
    return dcc.Tabs(id='tabs-container', children=[
        dcc.Tab(label='Banda de frecuencias seleccionada', children=[
            dcc.Graph(id='heatmap', figure=create_heatmap_data(df_original, 'level_dbuv_m',
                                                               'Nivel de Intensidad de Campo Eléctrico (dBµV/m)')),
            dcc.Slider(
                id='threshold-slider',
                min=0,
                max=100,
                step=1,
                value=None,
                marks={i: f"{str(i)} dBµV/m" for i in range(0, 101, 10)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            dcc.Graph(id='scatter'),
            html.Button("Mostrar Datos Procesados", id="toggle-table", className="mr-2"),
            html.Div(id='table-container', children=[
                html.Button("Descargar Excel", id="download-excel", style={'display': 'none'}),
                dcc.Download(id="download-data"),
                create_dash_datatable('table'),
            ], style={'display': 'none'}),
        ]),
        dcc.Tab(label='Análisis por parámetro', children=[
            dcc.Dropdown(
                id='parameter-dropdown',
                options=[
                    {'label': 'Level (dBµV/m)', 'value': 'level_dbuv_m'},
                    {'label': 'Offset (Hz)', 'value': 'offset_hz'},
                    {'label': 'Modulación', 'value': 'modulation_value'},
                    {'label': 'Ancho de banda (Hz)', 'value': 'bandwidth_hz'}
                ],
                value='level_dbuv_m',
                style={'margin': '10px'}
            ),
            dcc.Graph(id='parameter-heatmap'),
        ]),
        create_intermodulation_tab()
    ])


def calculate_occupation_percentage(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Calculate the percentage of occupation for each frequency based on the threshold.
    """
    if df.empty:
        return pd.DataFrame(columns=['frecuencia_hz', 'occupation_percentage'])

    # Asegurar que tenemos las columnas necesarias
    if 'level_dbuv_m' not in df.columns:
        print("Error: 'level_dbuv_m' no encontrado en el DataFrame")
        return pd.DataFrame(columns=['frecuencia_hz', 'occupation_percentage'])

    # Convertir a numérico
    df = df.copy()
    df['level_dbuv_m'] = pd.to_numeric(df['level_dbuv_m'].astype(str).replace('-', 'nan'), errors='coerce')

    # Calcular ocupación
    df['occupied'] = df['level_dbuv_m'] >= threshold
    occupation_percentage = df.groupby('frecuencia_hz')['occupied'].mean() * 100

    return pd.DataFrame({
        'frecuencia_hz': occupation_percentage.index,
        'occupation_percentage': occupation_percentage.values
    })


def calculate_intermodulation_products(df: pd.DataFrame, intermod_types: list,
                                       source_bands: list = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['freq', 'type', 'equation', 'level', 'source_freqs'])

    try:
        products = []
        df = df.copy()
        df['level_dbuv_m'] = pd.to_numeric(df['level_dbuv_m'].astype(str).replace('-', 'nan'), errors='coerce')

        strong_signals = df[df['level_dbuv_m'] > 40].groupby('frecuencia_hz')[
            'level_dbuv_m'].max() if not source_bands else \
            df[df['frecuencia_hz'].apply(lambda x: any(start * 1e6 <= x <= end * 1e6 for start, end in source_bands)) &
               (df['level_dbuv_m'] > 40)].groupby('frecuencia_hz')['level_dbuv_m'].max()

        frequencies = strong_signals.index.values / 1e6

        def get_signal_level(freq_hz, tolerance_hz=1000):
            freq_hz = np.int64(freq_hz)
            mask = (np.abs(strong_signals.index - freq_hz) <= tolerance_hz)
            if mask.any():
                closest_freq = strong_signals.index[mask][0]
                return strong_signals[closest_freq]
            return None

        def calculate_im_level(levels: list, arim: int) -> float:
            uv_m_levels = [10 ** (level / 20) for level in levels]
            total_uv_m = sum(uv_m_levels)
            return 20 * np.log10(total_uv_m) - arim

        if '2nd' in intermod_types:
            for i, f1 in enumerate(frequencies):
                for f2 in frequencies[i + 1:]:
                    f1_hz = np.int64(f1 * 1e6)
                    f2_hz = np.int64(f2 * 1e6)

                    level1 = get_signal_level(f1_hz)
                    level2 = get_signal_level(f2_hz)

                    if level1 is None or level2 is None:
                        continue

                    level_im = calculate_im_level([level1, level2], 45)  # ARIM = 45 dB

                    products.append({
                        'freq': float(f1 + f2),
                        'type': '2nd Order',
                        'equation': f'{f1:.3f} + {f2:.3f}',
                        'level': float(level_im),
                        'source_freqs': f'f1: {f1:.3f}MHz ({level1:.1f}dBµV/m), f2: {f2:.3f}MHz ({level2:.1f}dBµV/m)'
                    })

                    products.append({
                        'freq': float(abs(f1 - f2)),
                        'type': '2nd Order',
                        'equation': f'|{f1:.3f} - {f2:.3f}|',
                        'level': float(level_im),
                        'source_freqs': f'f1: {f1:.3f}MHz ({level1:.1f}dBµV/m), f2: {f2:.3f}MHz ({level2:.1f}dBµV/m)'
                    })

        if '3rd' in intermod_types:
            for f1 in frequencies:
                f1_hz = np.int64(f1 * 1e6)
                second_harmonic_f1_hz = np.int64(2 * f1 * 1e6)

                level1 = get_signal_level(f1_hz)
                level_2f1 = get_signal_level(second_harmonic_f1_hz)

                if level1 is None or level_2f1 is None:
                    continue

                for f2 in frequencies:
                    if f1 != f2:
                        f2_hz = np.int64(f2 * 1e6)
                        level2 = get_signal_level(f2_hz)

                        if level2 is None:
                            continue

                        level_im = calculate_im_level([level_2f1, level2], 70)  # ARIM = 70 dB

                        products.append({
                            'freq': float(2 * f1 - f2),
                            'type': '3rd Order',
                            'equation': f'2({f1:.3f}) - {f2:.3f}',
                            'level': float(level_im),
                            'source_freqs': f'2*f1: {2 * f1:.3f}MHz ({level_2f1:.1f}dBµV/m), f2: {f2:.3f}MHz ({level2:.1f}dBµV/m)'
                        })

                        for f3 in frequencies:
                            if f3 != f1 and f3 != f2:
                                f3_hz = np.int64(f3 * 1e6)
                                level3 = get_signal_level(f3_hz)

                                if level3 is None:
                                    continue

                                level_im = calculate_im_level([level1, level2, level3], 50)  # ARIM = 50 dB

                                products.append({
                                    'freq': float(f1 + f2 - f3),
                                    'type': '3rd Order',
                                    'equation': f'{f1:.3f} + {f2:.3f} - {f3:.3f}',
                                    'level': float(level_im),
                                    'source_freqs': f'f1: {f1:.3f}MHz ({level1:.1f}dBµV/m), f2: {f2:.3f}MHz ({level2:.1f}dBµV/m), f3: {f3:.3f}MHz ({level3:.1f}dBµV/m)'
                                })

        return pd.DataFrame(products)

    except Exception as e:
        print(f"Error en calculate_intermodulation_products: {str(e)}")
        return pd.DataFrame(columns=['freq', 'type', 'equation', 'level', 'source_freqs'])


def create_intermodulation_tab() -> dcc.Tab:
    """
    Crea la tab de análisis de intermodulación con selección flexible de rangos de frecuencia
    y un botón de ejecución.
    """
    return dcc.Tab(
        label='Análisis de Intermodulación',
        children=[
            html.Div([
                # Rango de análisis
                html.Div([
                    html.Label('Rango de Análisis (MHz)', className='font-bold mb-2'),
                    html.Div([
                        dcc.Input(
                            id='analysis-start-freq',
                            type='number',
                            placeholder='Frecuencia inicial',
                            className='w-32 p-2 border rounded'
                        ),
                        html.Span(' - ', className='mx-2'),
                        dcc.Input(
                            id='analysis-end-freq',
                            type='number',
                            placeholder='Frecuencia final',
                            className='w-32 p-2 border rounded'
                        )
                    ], className='flex items-center mb-4')
                ], className='mb-6'),

                # Selector de frecuencias específicas
                html.Div([
                    html.Label('Frecuencias específicas a analizar (MHz)', className='font-bold mb-2'),
                    dcc.Input(
                        id='specific-frequencies',
                        type='text',
                        placeholder='Ej: 98.5, 100.1, 102.3',
                        className='w-64 p-2 border rounded',
                        debounce=True
                    ),
                ], className='mb-6'),

                # Rangos de señales fuente
                html.Div([
                    html.Label('Rangos de Señales Fuente (MHz)', className='font-bold mb-2'),
                    html.Div(id='source-ranges-container', children=[
                        html.Div([
                            dcc.Input(
                                id={'type': 'source-start', 'index': 0},
                                type='number',
                                placeholder='Inicio rango 1',
                                className='w-32 p-2 border rounded'
                            ),
                            html.Span(' - ', className='mx-2'),
                            dcc.Input(
                                id={'type': 'source-end', 'index': 0},
                                type='number',
                                placeholder='Fin rango 1',
                                className='w-32 p-2 border rounded'
                            )
                        ], className='flex items-center mb-2')
                    ]),
                    html.Button(
                        'Agregar Rango',
                        id='add-range-button',
                        className='bg-blue-500 text-white px-4 py-2 rounded mt-2'
                    )
                ], className='mb-6'),

                # Tipos de productos
                html.Div([
                    dcc.Checklist(
                        id='intermod-type',
                        options=[
                            {'label': 'Productos de 2do Orden', 'value': '2nd'},
                            {'label': 'Productos de 3er Orden', 'value': '3rd'}
                        ],
                        value=['2nd', '3rd'],
                        className='space-y-2'
                    )
                ], className='mb-6'),

                # Botón de ejecución
                html.Button(
                    'Ejecutar Análisis',
                    id='execute-analysis-button',
                    className='bg-green-500 text-white px-6 py-3 rounded font-bold mb-6'
                ),

                # Visualización
                html.Div([
                    dcc.Loading(
                        id="loading-intermod",
                        type="default",
                        children=[
                            dcc.Graph(id='intermod-heatmap'),
                            dash_table.DataTable(
                                id='intermod-table',
                                columns=[
                                    {'name': 'Frecuencia (MHz)', 'id': 'freq'},
                                    {'name': 'Tipo', 'id': 'type'},
                                    {'name': 'Ecuación', 'id': 'equation'},
                                    {'name': 'Frecuencias Origen', 'id': 'source_freqs'}
                                ],
                                style_table={'height': '300px', 'overflowY': 'auto'},
                                style_cell={'textAlign': 'left'},
                                style_header={'fontWeight': 'bold'}
                            ),
                            # Contenedor para resultados de frecuencias específicas
                            html.Div(id='frequency-analysis-results', className='mt-6')
                        ]
                    )
                ])
            ], className='p-4')
        ]
    )


def create_intermod_heatmap(df: pd.DataFrame, products_df: pd.DataFrame, analysis_range: tuple,
                            source_ranges: list) -> go.Figure:
    """
    Crea el heatmap con productos de intermodulación y señales fuente superpuestas,
    asegurando que los marcadores se alineen con el último tiempo del heatmap.
    """
    # Convertir los datos para el heatmap
    df = df.copy()
    df['tiempo'] = pd.to_datetime(df['tiempo'])
    df['level_dbuv_m'] = pd.to_numeric(df['level_dbuv_m'], errors='coerce')

    # Crear matriz para el heatmap
    pivot_data = df.pivot_table(
        values='level_dbuv_m',
        index='tiempo',
        columns='frecuencia_hz',
        aggfunc='max'
    ).fillna(0)

    # Obtener el último tiempo del pivot_data
    last_time = pivot_data.index[-1]

    # Crear la figura base
    fig = go.Figure()

    # Agregar el heatmap como la capa base
    fig.add_trace(go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='rainbow',
        zmin=0,
        zmax=100,
        name='Nivel de Campo',
        showscale=True,
        colorbar=dict(
            title='dBµV/m',
            x=1.1
        )
    ))

    # Agregar señales fuente si existen
    for start_freq, end_freq in source_ranges:
        if start_freq and end_freq:  # Verificar que los valores no sean None
            start_hz = start_freq * 1e6
            end_hz = end_freq * 1e6
            mask = (df['frecuencia_hz'] >= start_hz) & (df['frecuencia_hz'] <= end_hz)
            source_freqs = df[mask]['frecuencia_hz'].unique()

            if len(source_freqs) > 0:
                fig.add_trace(go.Scatter(
                    x=source_freqs,
                    y=[last_time] * len(source_freqs),  # Usar el último tiempo del heatmap
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=12,
                        color='yellow',
                        line=dict(width=2, color='black')
                    ),
                    name='Señales Fuente',
                    hovertemplate='Frecuencia: %{x:.2f} Hz<extra></extra>'
                ))

    # Agregar productos de intermodulación si existen
    if not products_df.empty:
        fig.add_trace(go.Scatter(
            x=products_df['freq'] * 1e6,  # Convertir MHz a Hz
            y=[last_time] * len(products_df),  # Usar el último tiempo del heatmap
            mode='markers',
            marker=dict(
                symbol='x',
                size=10,
                color='red',
                line=dict(width=2)
            ),
            name='Productos de Intermodulación',
            hovertemplate='Frecuencia: %{x:.2f} Hz<extra></extra>'
        ))

    # Configurar el layout
    fig.update_layout(
        title='Nivel de Campo con Productos de Intermodulación',
        xaxis_title='Frecuencia (Hz)',
        yaxis_title='Tiempo',
        showlegend=True,
        legend=dict(
            x=1.2,
            y=1,
            bordercolor='Black',
            borderwidth=1
        ),
        margin=dict(r=150),
        plot_bgcolor='rgba(240,240,240,0.95)',
        paper_bgcolor='white'
    )

    # Ajustar el rango del eje x si se proporciona
    if analysis_range:
        fig.update_xaxes(range=analysis_range)

    # Asegurar que el rango del eje y incluya todos los datos
    fig.update_yaxes(range=[pivot_data.index[0], pivot_data.index[-1]])

    return fig


def analyze_specific_frequencies(df: pd.DataFrame, products_df: pd.DataFrame, selected_frequencies: list) -> dict:
    """
    Analiza frecuencias específicas y sus productos de intermodulación.
    """
    results = {}

    try:
        for freq in selected_frequencies:
            freq_hz = freq * 1e6
            tolerance = 0.001 * 1e6

            freq_data = df[
                (df['frecuencia_hz'] >= freq_hz - tolerance) &
                (df['frecuencia_hz'] <= freq_hz + tolerance)
                ].copy()

            freq_data['level_dbuv_m'] = pd.to_numeric(freq_data['level_dbuv_m'], errors='coerce')
            avg_level = freq_data['level_dbuv_m'].mean() if not freq_data.empty else 0

            affecting_products = products_df[
                (products_df['freq'] >= freq - 0.001) &
                (products_df['freq'] <= freq + 0.001)
                ].copy() if 'freq' in products_df.columns else pd.DataFrame()

            # Ordenar productos por nivel descendente
            affecting_products = affecting_products.sort_values('level', ascending=False)

            contributing_products = []
            if not affecting_products.empty:
                for _, product in affecting_products.iterrows():
                    try:
                        contributing_products.append({
                            'type': str(product.get('type', '')),
                            'equation': str(product.get('equation', '')),
                            'level': f"{float(product.get('level', 0)):.1f}",
                            'source_freqs': str(product.get('source_freqs', ''))
                        })
                    except (ValueError, TypeError) as e:
                        print(f"Error procesando producto: {e}")
                        continue

            # Calcular contribución total
            total_contribution = 0
            if contributing_products:
                try:
                    powers = [10 ** (float(product['level']) / 20) for product in contributing_products]
                    total_contribution = 20 * np.log10(sum(powers))
                except (ValueError, TypeError) as e:
                    print(f"Error calculando contribución total: {e}")

            results[freq] = {
                'average_level': float(avg_level),
                'num_products': len(contributing_products),
                'total_contribution': float(total_contribution),
                'contributing_products': contributing_products,
                'measurements': create_measurements_list(freq_data)
            }

    except Exception as e:
        print(f"Error en analyze_specific_frequencies: {str(e)}")
        raise

    return results


def create_measurements_list(freq_data):
    measurements = []
    if not freq_data.empty:
        for _, row in freq_data.iterrows():
            try:
                measurements.append({
                    'tiempo': row['tiempo'],
                    'level_dbuv_m': float(row['level_dbuv_m']) if pd.notnull(row['level_dbuv_m']) else 0
                })
            except (ValueError, TypeError) as e:
                print(f"Error procesando medición: {e}")
    return measurements


def create_frequency_timeline(measurements: list, frequency: float) -> go.Figure:
    """
    Crea un gráfico de línea temporal para una frecuencia específica.
    """
    df = pd.DataFrame(measurements)
    if df.empty:
        return go.Figure()

    # Asegurar tipos de datos correctos
    df['tiempo'] = pd.to_datetime(df['tiempo'])
    df['level_dbuv_m'] = pd.to_numeric(df['level_dbuv_m'], errors='coerce')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['tiempo'],
        y=df['level_dbuv_m'],
        mode='lines+markers',
        name='Nivel de señal'
    ))

    fig.update_layout(
        title=f'Nivel de señal en el tiempo para {frequency:.3f} MHz',
        xaxis_title='Tiempo',
        yaxis_title='Nivel (dBµV/m)',
        showlegend=True,
        yaxis=dict(range=[0, max(100, df['level_dbuv_m'].max() if not df.empty else 100)])
    )

    return fig
