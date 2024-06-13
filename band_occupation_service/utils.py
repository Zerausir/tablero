import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
from dash import dcc, dash_table, html


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


def create_dash_datatable(table_id: str, dataframe: pd.DataFrame, style: dict = {}) -> dash_table.DataTable:
    """
    Create a Dash DataTable from a Pandas DataFrame with certain pre-defined styles and functionalities.

    Args:
        table_id (str): Identifier for the Dash DataTable.
        dataframe (pd.DataFrame): The DataFrame from which the Dash DataTable is to be created.

    Returns:
        dash_table.DataTable: A Dash DataTable component.
    """
    # Define the range for your color scale
    zmin = 0
    zmax = 130

    # Define the Plotly rainbow color scale
    plotly_rainbow_scale = ['rgb(150,0,90)', 'rgb(0,0,200)', 'rgb(0,25,255)', 'rgb(0,152,255)',
                            'rgb(44,255,150)', 'rgb(151,255,0)', 'rgb(255,234,0)', 'rgb(255,111,0)', 'rgb(255,0,0)']

    # Function to determine if text color should be white or black based on background color
    def text_color_based_on_background(background_color):
        # Parse the background color string and get R, G, B values
        r, g, b = [int(x) for x in background_color.replace('rgb(', '').replace(')', '').split(',')]
        # Calculate the luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        # Return 'white' for dark background colors and 'black' for light background colors
        return 'white' if luminance < 150 else 'black'

    # Create conditional formatting for the 'Level (dBµV/m)' column
    style_data_conditional = [
        {
            'if': {
                'filter_query': f'{{level_dbuv_m}} >= {zmin + (zmax - zmin) * i / (len(plotly_rainbow_scale) - 1)} && {{level_dbuv_m}} < {zmin + (zmax - zmin) * (i + 1) / (len(plotly_rainbow_scale) - 1)}',
                'column_id': 'level_dbuv_m'
            },
            'backgroundColor': plotly_rainbow_scale[i],
            'color': text_color_based_on_background(plotly_rainbow_scale[i])
        }
        for i in range(len(plotly_rainbow_scale))  # Iterate through the color scale
    ]

    return dash_table.DataTable(
        data=dataframe.to_dict('records'),
        columns=[{"name": col, "id": col} for col in dataframe.columns],
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
        id=table_id,
        style_data_conditional=style_data_conditional,
        fixed_rows={'headers': True},  # This line fixes the header row at the top
    )


def create_heatmap_data(df: pd.DataFrame, x_range=None) -> go.Figure:
    """
    Create heatmap data from a DataFrame. The DataFrame is expected to have 'Level (dBµV/m)',
    'Tiempo', and 'Frecuencia (Hz)' columns. If the DataFrame is empty, an empty figure is returned.

    Args:
        df (pd.DataFrame): The DataFrame from which the heatmap data is to be created.
        x_range (list, optional): The range of the x-axis. If None, the range is determined from the data.

    Returns:
        go.Figure: A Plotly Heatmap figure.
    """
    if df.empty:
        return go.Figure()

    df = df.fillna(0)
    df['tiempo'] = pd.to_datetime(df['tiempo'])

    # Asegurarse de que 'Level (dBµV/m)' sea numérico, reemplazar '-' por NaN
    df['level_dbuv_m'] = pd.to_numeric(df['level_dbuv_m'], errors='coerce')

    # Utilizar "mean" en lugar de np.mean para evitar advertencias futuras
    heatmap_data = df.pivot_table(values='level_dbuv_m', index='tiempo', columns='frecuencia_hz', aggfunc='mean')

    layout = go.Layout(
        title='Nivel de Intensidad de Campo Eléctrico (dBµV/m) vs Frecuencia',
        xaxis={'title': 'Frecuencia (Hz)'},
        yaxis={'title': 'Tiempo'},
    )

    if x_range is not None:
        layout['xaxis']['range'] = x_range

    return go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='rainbow',
            zmin=0,
            zmax=100,
            colorbar=dict(orientation='h', y=1)
        ),
        layout=layout
    )


def create_scatter_plot(df: pd.DataFrame, x_range=None) -> go.Figure:
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

    layout = go.Layout(
        title='Porcentaje de Ocupación vs Frecuencia',
        xaxis={'title': 'Frecuencia (Hz)'},
        yaxis={'title': 'Porcentaje de Ocupación (%)', 'range': [0, 100]},
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


def create_heatmap_layout(df_original1: pd.DataFrame, df_original2: pd.DataFrame,
                          df_original3: pd.DataFrame, threshold: float) -> dcc.Tabs:
    """
    Create a tabs layout with dropdowns, tables, and placeholders for heatmaps and scatter plots for three different data sources.

    Args:
        df_original1 (pd.DataFrame): DataFrame for the first tab.
        df_original2 (pd.DataFrame): DataFrame for the second tab.
        df_original3 (pd.DataFrame): DataFrame for the third tab.
        threshold (float): The threshold value for level_dbuv_m.

    Returns:
        dcc.Tabs: Tabs component containing dropdowns, tables, graph placeholders, and scatter plots.
    """
    scatter1_df = calculate_occupation_percentage(df_original1, threshold)
    scatter2_df = calculate_occupation_percentage(df_original2, threshold)
    scatter3_df = calculate_occupation_percentage(df_original3, threshold)

    scatter1_df_final = scatter1_df.rename(
        columns={'frecuencia_hz': 'Frecuencia (Hz)', 'occupation_percentage': 'Ocupación (%)'})
    scatter2_df_final = scatter2_df.rename(
        columns={'frecuencia_hz': 'Frecuencia (Hz)', 'occupation_percentage': 'Ocupación (%)'})
    scatter3_df_final = scatter3_df.rename(
        columns={'frecuencia_hz': 'Frecuencia (Hz)', 'occupation_percentage': 'Ocupación (%)'})

    table1 = create_dash_datatable('table1', scatter1_df_final)
    table2 = create_dash_datatable('table2', scatter2_df_final)
    table3 = create_dash_datatable('table3', scatter3_df_final)

    x_range1 = [scatter1_df['frecuencia_hz'].min(), scatter1_df['frecuencia_hz'].max()]
    x_range2 = [scatter2_df['frecuencia_hz'].min(), scatter2_df['frecuencia_hz'].max()]
    x_range3 = [scatter3_df['frecuencia_hz'].min(), scatter3_df['frecuencia_hz'].max()]

    tabs_layout = dcc.Tabs(id='tabs-container', children=[
        dcc.Tab(label='Banda de frecuencias: 703-733 MHz', children=[
            html.Button("Toggle Table", id="toggle-table1", className="mr-2"),
            html.Div(id='table1-container', children=[
                html.Button("Download Excel", id="download-excel1", style={'display': 'none'}),
                dcc.Download(id="download-data1"),
                table1,
            ], style={'display': 'none'}),  # Initially hide the table container
            dcc.Graph(id='heatmap1', figure=create_heatmap_data(df_original1, x_range1)),
            dcc.Graph(id='scatter1', figure=create_scatter_plot(scatter1_df, x_range1)),
        ]),
        dcc.Tab(label='Banda de frecuencias: 758-788 MHz', children=[
            html.Button("Toggle Table", id="toggle-table2", className="mr-2"),
            html.Div(id='table2-container', children=[
                html.Button("Download Excel", id="download-excel2", style={'display': 'none'}),
                dcc.Download(id="download-data2"),
                table2,
            ], style={'display': 'none'}),
            dcc.Graph(id='heatmap2', figure=create_heatmap_data(df_original2, x_range2)),
            dcc.Graph(id='scatter2', figure=create_scatter_plot(scatter2_df, x_range2)),
        ]),
        dcc.Tab(label='Banda de frecuencias: 2500-2690 MHz', children=[
            html.Button("Toggle Table", id="toggle-table3", className="mr-2"),
            html.Div(id='table3-container', children=[
                html.Button("Download Excel", id="download-excel3", style={'display': 'none'}),
                dcc.Download(id="download-data3"),
                table3,
            ], style={'display': 'none'}),
            dcc.Graph(id='heatmap3', figure=create_heatmap_data(df_original3, x_range3)),
            dcc.Graph(id='scatter3', figure=create_scatter_plot(scatter3_df, x_range3)),
        ]),
    ])

    return tabs_layout


def calculate_occupation_percentage(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Calculate the percentage of occupation for each frequency based on the threshold.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        threshold (float): The threshold value for level_dbuv_m.

    Returns:
        pd.DataFrame: A DataFrame with the frequency and the corresponding occupation percentage.
    """
    # Replace non-numeric values with NaN
    df['level_dbuv_m'] = df['level_dbuv_m'].replace('-', np.nan)

    df['level_dbuv_m'] = pd.to_numeric(df['level_dbuv_m'], errors='coerce')

    df['occupied'] = df['level_dbuv_m'] >= threshold
    occupation_percentage = df.groupby('frecuencia_hz')['occupied'].mean() * 100
    return pd.DataFrame(
        {'frecuencia_hz': occupation_percentage.index, 'occupation_percentage': occupation_percentage.values})
