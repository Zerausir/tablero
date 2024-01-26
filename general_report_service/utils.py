import dask.dataframe as dd
import numpy as np
import pandas as pd
import datetime
from django.conf import settings
import plotly.graph_objs as go
from dash import dcc, html, dash_table, no_update
from typing import Union

MONTH_TRANSLATIONS = settings.MONTH_TRANSLATIONS


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
    """
    Convert timestamps in the DataFrame to strings and handle MultiIndex for columns and index.
    This function specifically targets 'Tiempo' columns for conversion and also adjusts
    MultiIndex headers and indexes if present.

    Args:
        df (pd.DataFrame): The DataFrame whose timestamps are to be converted.

    Returns:
        pd.DataFrame: DataFrame with timestamps converted to string format.
    """
    if 'Tiempo' in df.columns:
        df['Tiempo'] = df['Tiempo'].dt.strftime('%Y-%m-%d')

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(
            lambda x: [(y.strftime('%Y-%m-%d') if isinstance(y, pd.Timestamp) else y) for y in x])
    else:
        df.columns = [x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x for x in df.columns]

    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.map(lambda x: [(y.strftime('%Y-%m-%d') if isinstance(y, pd.Timestamp) else y) for y in x])
    else:
        df.index = [x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x for x in df.index]

    return df


def create_dash_datatable(table_id: str, dataframe: pd.DataFrame) -> dash_table.DataTable:
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
                'filter_query': f'{{Level (dBµV/m)}} >= {zmin + (zmax - zmin) * i / (len(plotly_rainbow_scale) - 1)} && {{Level (dBµV/m)}} < {zmin + (zmax - zmin) * (i + 1) / (len(plotly_rainbow_scale) - 1)}',
                'column_id': 'Level (dBµV/m)'
            },
            'backgroundColor': plotly_rainbow_scale[i],
            'color': text_color_based_on_background(plotly_rainbow_scale[i])
        }
        for i in range(len(plotly_rainbow_scale))  # Iterate through the color scale
    ]

    return dash_table.DataTable(
        data=dataframe.to_dict('records'),
        columns=[{"name": col, "id": col} for col in dataframe.columns],
        style_table={'overflowX': 'auto', 'maxHeight': '300px'},
        style_cell={  # Default cell style
            'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_cell_conditional=[  # Conditional styling to handle different column types or specific columns
            {'if': {'column_id': c},
             'textAlign': 'left'} for c in ['column_name_1', 'column_name_2']
            # Example for text alignment, replace with your column names
        ],
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


def create_frequency_dropdown(dropdown_id: str, dataframe: pd.DataFrame, placeholder_text: str) -> dcc.Dropdown:
    """
    Create a dropdown for frequency selection from a DataFrame column 'Frecuencia (Hz)'.
    If the DataFrame is empty or the specified column doesn't exist, provide a fallback placeholder text.

    Args:
        dropdown_id (str): Identifier for the dropdown.
        dataframe (pd.DataFrame): DataFrame containing the frequency data.
        placeholder_text (str): Text to display when no options are available or as a placeholder.

    Returns:
        dcc.Dropdown: A Dash core component Dropdown.
    """
    if not dataframe.empty and 'Frecuencia (Hz)' in dataframe.columns:
        options = [{'label': freq, 'value': freq} for freq in
                   sorted(dataframe['Frecuencia (Hz)'].unique().tolist())]
    else:
        options = []
        placeholder_text = "No existe información disponible"
    return dcc.Dropdown(
        id=dropdown_id,
        options=options,
        placeholder=placeholder_text,
        multi=True,
        style={'margin': '10px'}
    )


def create_heatmap_data(df: pd.DataFrame, selected_frequencies=None) -> dict:
    """
    Create heatmap data from a DataFrame. The DataFrame is expected to have 'Level (dBµV/m)',
    'Tiempo', and 'Frecuencia (Hz)' columns. If the DataFrame is empty, an empty figure is returned.

    Args:
        df (pd.DataFrame): The DataFrame from which the heatmap data is to be created.

    Returns:
        dict: A dictionary containing 'data' and 'layout' for the heatmap.
    """
    if df.empty:
        return go.Figure()

    df = df.fillna(0)

    heatmap_data = df.pivot_table(values='Level (dBµV/m)', index='Tiempo', columns=['Frecuencia (Hz)', 'Estación'])

    # Create the x-axis labels. If 'Estación' is "-", only use 'Frecuencia (Hz)'
    freq_categories = [
        "{} Hz".format(freq) if station == "-" else "{} Hz | {}".format(freq, station)
        for freq, station in heatmap_data.columns
    ]

    if selected_frequencies:
        bottom_margin = 100  # Smaller margin if frequencies are selected
        plot_height = 600  # Smaller height if frequencies are selected
    else:
        bottom_margin = 400  # Larger margin to accommodate x-axis labels
        plot_height = 900  # Larger height for initial view

    return {
        'data': [go.Heatmap(
            z=heatmap_data.values[::-1],
            x=freq_categories,
            y=heatmap_data.index[::-1],
            colorscale='rainbow',
            zmin=0,
            zmax=130,
            colorbar=dict(title='Nivel (dBµV/m)')
        )],
        'layout': go.Layout(
            title='Nivel de Intensidad de Campo Eléctrico por Frecuencia',
            xaxis={
                'title': 'Frecuencia (Hz)',
                'type': 'category',
                'tickvals': list(range(len(freq_categories))),
                'ticktext': freq_categories
            },
            yaxis={'title': 'Tiempo'},
            margin=go.layout.Margin(
                l=100,
                r=100,
                b=bottom_margin,
                t=100,
            ),
            autosize=True,
            height=plot_height,
        )
    }


def create_heatmap_layout(df_original1: pd.DataFrame, df_original2: pd.DataFrame,
                          df_original3: pd.DataFrame) -> dcc.Tabs:
    """
    Create a tabs layout with dropdowns, tables, and placeholders for heatmaps for three different data sources.

    Args:
        df_original1 (pd.DataFrame): DataFrame for the first tab.
        df_original2 (pd.DataFrame): DataFrame for the second tab.
        df_original3 (pd.DataFrame): DataFrame for the third tab.

    Returns:
        dcc.Tabs: Tabs component containing dropdowns, tables, and graph placeholders.
    """
    dropdown1 = create_frequency_dropdown('frequency-dropdown1', df_original1, "Seleccione una Frecuencia")
    dropdown2 = create_frequency_dropdown('frequency-dropdown2', df_original2, "Seleccione una Frecuencia")
    dropdown3 = create_frequency_dropdown('frequency-dropdown3', df_original3, "Seleccione una Frecuencia")

    table1 = create_dash_datatable('table1', df_original1)
    table2 = create_dash_datatable('table2', df_original2)
    table3 = create_dash_datatable('table3', df_original3)

    tabs_layout = dcc.Tabs([
        dcc.Tab(label='Radiodifusión FM', children=[
            html.Div(dropdown1, style={'marginBottom': '10px'}),
            table1,
            dcc.Graph(id='heatmap1'),
        ]),
        dcc.Tab(label='Televisión', children=[
            html.Div(dropdown2, style={'marginBottom': '10px'}),
            table2,
            dcc.Graph(id='heatmap2'),
        ]),
        dcc.Tab(label='Radiodifusión AM', children=[
            html.Div(dropdown3, style={'marginBottom': '10px'}),
            table3,
            dcc.Graph(id='heatmap3'),
        ]),
    ])

    return tabs_layout


def filter_dataframe_by_frequencies(df: pd.DataFrame, selected_frequencies: list) -> pd.DataFrame:
    """
    Filter the DataFrame based on a list of selected frequencies.

    Args:
        df (pd.DataFrame): The original DataFrame.
        selected_frequencies (list): List of frequencies to filter by.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if selected_frequencies:
        # Convert selected frequencies to float if they are not already.
        selected_frequencies = [float(freq) for freq in selected_frequencies]
        # Filter the DataFrame.
        return df[df['Frecuencia (Hz)'].isin(selected_frequencies)]
    else:
        # If no frequencies are selected, return the DataFrame as is.
        return df


def read_and_fill_excel(file_path: str, sheet_name: str, fill_value: str = '-') -> pd.DataFrame:
    """
    Read and fill missing values in an Excel file.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str): Sheet name in the Excel file.
        fill_value (str): Value to fill missing cells.

    Returns:
        pd.DataFrame: DataFrame with filled values.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name).fillna(fill_value)


def read_csv_file(file_path: str, columns: list[str]) -> dd.DataFrame:
    """
    Read CSV file using Dask and return data as a Dask DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        columns (List[str]): List of columns to be read.

    Returns:
        dd.DataFrame: Dask DataFrame containing the data.
    """
    try:
        return dd.read_csv(file_path, usecols=columns, assume_missing=True, encoding='latin1')
    except IOError:
        return dd.from_pandas(pd.DataFrame(np.full((1, len(columns)), np.nan), columns=columns), npartitions=1)


def translate_month(month: str) -> str:
    """
    Translate month abbreviation to full month name.

    Args:
        month (str): Month abbreviation.

    Returns:
        str: Full month name.
    """
    return MONTH_TRANSLATIONS.get(month, month)


def update_heatmap(selected_frequencies: list, stored_data: list) -> Union[go.Figure, dict]:
    """
    Update heatmap based on the selected frequencies and stored data.

    Args:
        selected_frequencies (list): List of selected frequencies.
        stored_data (list): Previously stored data as a list of dictionaries.

    Returns:
        Union[go.Figure, dict]: Updated heatmap figure or a dictionary containing heatmap data and layout if not empty.
    """
    if stored_data is None:
        return go.Figure()

    df_clean = pd.DataFrame.from_records(stored_data)

    if not selected_frequencies:
        # If no frequencies are selected, use the entire dataset
        filtered_df = df_clean
    else:
        # If frequencies are selected, filter the DataFrame
        filtered_df = df_clean[df_clean['Frecuencia (Hz)'].isin(selected_frequencies)]

    # Generate the heatmap figure using the filtered DataFrame
    return create_heatmap_data(filtered_df)


def update_table(selected_frequencies: list, stored_data: list, table_id: str) -> list:
    """
    Update table based on the selected frequencies.

    Args:
        selected_frequencies (list): List of selected frequencies.
        stored_data (list): Previously stored data as a list of dictionaries.
        table_id (str): ID of the table to be updated.

    Returns:
        list: Updated data for the table in dictionary format.
    """
    if stored_data is None:
        return []

    df_original = pd.DataFrame.from_records(stored_data)

    if not selected_frequencies:
        return df_original.to_dict('records')

    filtered_df = df_original[df_original['Frecuencia (Hz)'].isin(selected_frequencies)]
    return filtered_df.to_dict('records')


def create_station_plot(df, frequency):
    # Check if frequency is a list and filter the DataFrame accordingly
    if isinstance(frequency, list):
        df_filtered = df[df['Frecuencia (Hz)'].isin(frequency)]
    else:
        df_filtered = df[df['Frecuencia (Hz)'] == frequency]

    # Create the plot
    fig = go.Figure(
        data=go.Scatter(
            x=df_filtered['Tiempo'],
            y=df_filtered['Level (dBµV/m)'],
            mode='lines+markers',
            name='Level vs. Tiempo'
        ),
        layout=go.Layout(
            title=f'Level (dBµV/m) vs. Tiempo for {frequency} Hz',
            xaxis_title='Tiempo',
            yaxis_title='Level (dBµV/m)',
        )
    )
    return fig


def update_station_plot(selected_frequencies, stored_data):
    if not selected_frequencies or not stored_data:
        return no_update

    # Convert stored_data to DataFrame
    df = pd.DataFrame.from_records(stored_data)

    # Container for the plots
    plots = []

    # Create a plot for each selected frequency
    for frequency in selected_frequencies:
        df_filtered = df[df['Frecuencia (Hz)'] == frequency]
        fig = go.Figure(
            data=go.Scatter(
                x=df_filtered['Tiempo'],
                y=df_filtered['Level (dBµV/m)'],
                mode='lines+markers',
                name=f'Level vs. Tiempo for {frequency} Hz'
            ),
            layout=go.Layout(
                title=f'Level (dBµV/m) vs. Tiempo for {frequency} Hz',
                xaxis_title='Tiempo',
                yaxis_title='Level (dBµV/m)',
            )
        )
        plots.append(dcc.Graph(figure=fig))

    # Return a Div containing all the plots
    return html.Div(plots)
