import datetime
import json
import numpy as np
import pandas as pd
import dask.dataframe as dd

from django.conf import settings
from pandas import DataFrame
from dash import dcc, html, Input, Output, dash_table
from django_plotly_dash import DjangoDash
from .api.api_client import get_options_from_index_service_api

# Initialize the Dash app
app = DjangoDash(
    name='GeneralReportApp',
    add_bootstrap_links=True,
    external_stylesheets=[
        "/static/css/inner.css",
    ]
)

# Define the layout of the app
app.layout = html.Div([
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date_placeholder_text="Fecha Inicio",
        end_date_placeholder_text="Fecha Fin",
        style={'margin': '10px'}
    ),
    dcc.Dropdown(
        id='city-dropdown',
        options=[{'label': ciudad, 'value': ciudad} for ciudad in json.loads(settings.CITIES)],
        placeholder="Selecciona una ciudad",
        style={'margin': '10px'}
    ),
    dcc.Checklist(
        id='checkbox',
        options=[
            {'label': 'Autorizaciones Suspensión/Baja Potencia', 'value': 'auth_suspension'}
        ],
        value=[],
        style={'margin': '10px'}
    ),
    dcc.Loading(
        id="loading-1",
        type="default",  # Options: "graph", "cube", "circle", "dot", or "default"
        children=[
            html.Div(id='data-tables-container'),
            dcc.Graph(id='heatmap-graph')
        ],
        style={'margin': '10px'}
    ),
    dcc.Store(id='store-df-original1'),
    dcc.Store(id='store-df-original2'),
    dcc.Store(id='store-df-original3'),
    dcc.Store(id='store-df-original4'),
    dcc.Store(id='store-df-original5'),
    dcc.Store(id='store-df-original6'),
])


# Function to create a dropdown for frequency selection
def create_frequency_dropdown(dropdown_id, dataframe, placeholder_text):
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


# Function to create a Dash DataTable
def create_dash_datatable(table_id, dataframe):
    return dash_table.DataTable(
        data=dataframe.to_dict('records'),
        columns=[{"name": col, "id": col} for col in dataframe.columns],
        style_table={'overflowX': 'auto', 'maxHeight': '300px'},
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
        id=table_id
    )


# Callback for updating tables
@app.callback(
    [Output('store-df-original1', 'data'),
     Output('store-df-original2', 'data'),
     Output('store-df-original3', 'data'),
     Output('data-tables-container', 'children')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('city-dropdown', 'value')]
)
def update_tables(fecha_inicio, fecha_fin, ciudad):
    if not all([fecha_inicio, fecha_fin, ciudad]):
        return {}, {}, {}, html.Div("Por favor selecciona una fecha de inicio, una fecha de fin y una ciudad.")

    if fecha_inicio is not None and fecha_fin is not None and ciudad is not None:
        selected_options = {
            'start_date': fecha_inicio,
            'end_date': fecha_fin,
            'city': ciudad,
        }
        df_original1, df_original2, df_original3 = customize_data(selected_options)[:3]
        df_original1 = convert_timestamps_to_strings(df_original1)
        df_original2 = convert_timestamps_to_strings(df_original2)
        df_original3 = convert_timestamps_to_strings(df_original3)

        # Create Dropdowns and Tables
        dropdown1 = create_frequency_dropdown('frequency-dropdown1', df_original1, "Seleccione una Frecuencia")
        dropdown2 = create_frequency_dropdown('frequency-dropdown2', df_original2, "Seleccione una Frecuencia")
        dropdown3 = create_frequency_dropdown('frequency-dropdown3', df_original3, "Seleccione una Frecuencia")

        table1 = create_dash_datatable('table1', df_original1)
        table2 = create_dash_datatable('table2', df_original2)
        table3 = create_dash_datatable('table3', df_original3)

        # Prepare the tabs layout with dropdowns and tables
        tabs_layout = dcc.Tabs([
            dcc.Tab(label='Radiodifusión FM', children=[
                html.Div(dropdown1, style={'marginBottom': '10px'}),
                table1
            ]),
            dcc.Tab(label='Televisión', children=[
                html.Div(dropdown2, style={'marginBottom': '10px'}),
                table2
            ]),
            dcc.Tab(label='Radiodifusión AM', children=[
                html.Div(dropdown3, style={'marginBottom': '10px'}),
                table3
            ]),
        ])

        # Return the data for the stores and the tabs layout for the 'data-tables-container'
        return df_original1.to_dict('records'), df_original2.to_dict('records'), df_original3.to_dict(
            'records'), tabs_layout


def update_table(selected_frequencies, stored_data, table_id):
    if stored_data is None:
        return []

    df_original = pd.DataFrame.from_records(stored_data)

    if not selected_frequencies:
        return df_original.to_dict('records')

    filtered_df = df_original[df_original['Frecuencia (Hz)'].isin(selected_frequencies)]
    return filtered_df.to_dict('records')


@app.callback(
    Output('table1', 'data'),
    [Input('frequency-dropdown1', 'value'),
     Input('store-df-original1', 'data')]
)
def update_table1(selected_frequencies1, stored_data1):
    return update_table(selected_frequencies1, stored_data1, 'table1')


@app.callback(
    Output('table2', 'data'),
    [Input('frequency-dropdown2', 'value'),
     Input('store-df-original2', 'data')]
)
def update_table2(selected_frequencies2, stored_data2):
    return update_table(selected_frequencies2, stored_data2, 'table2')


@app.callback(
    Output('table3', 'data'),
    [Input('frequency-dropdown3', 'value'),
     Input('store-df-original3', 'data')]
)
def update_table3(selected_frequencies3, stored_data3):
    return update_table(selected_frequencies3, stored_data3, 'table3')


CITIES1 = settings.CITIES1
MONTH_TRANSLATIONS = settings.MONTH_TRANSLATIONS


def get_options(self) -> dict:
    """
    Retrieve options from the index service API.

    Returns:
        dict: Options obtained from the API.
    """
    return get_options_from_index_service_api()


def customize_data(selected_options: dict) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Customize data based on selected city and date range.

    Args:
        ciudad (str): Selected city.
        fecha_inicio (str): Start date in the format 'YYYY-MM-DD'.
        fecha_fin (str): End date in the format 'YYYY-MM-DD'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Customized dataframes.
    """
    ciudad = selected_options['city']
    fecha_inicio = selected_options['start_date']
    fecha_fin = selected_options['end_date']

    ciu, autori, sheet_name1, sheet_name2, *rest = CITIES1.get(ciudad, (None, None, None, None, None))
    sheet_name3 = rest[0] if rest else None  # Assign sheet_name3 only if rest is not empty

    def convert(date_time):
        format_str = '%Y-%m-%d %H:%M:%S' if ' ' in date_time else '%Y-%m-%d'
        datetime_obj = datetime.datetime.strptime(date_time, format_str)
        return datetime_obj

    Year1, Year2 = convert(fecha_inicio).year, convert(fecha_fin).year

    month_year = generate_month_year_vector(Year1, Year2)

    df_data1, df_data2, df_data3 = read_data_files(selected_options, ciu, month_year, sheet_name1, sheet_name2,
                                                   sheet_name3)

    dfau = pd.concat([read_and_process_aut(settings.FILE_AUT_SUS, settings.COLUMNS_AUT, 'S'),
                      read_and_process_aut(settings.FILE_AUT_BP, settings.COLUMNS_AUTBP, 'BP')],
                     ignore_index=True)

    df_origin1 = pd.DataFrame(df_data1, columns=settings.COLUMNS_FM)
    df_origin1['Tiempo'] = pd.to_datetime(df_origin1['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
    df_clean1 = clean_data(fecha_inicio, fecha_fin, df_origin1, sheet_name1)
    df_original1 = simplify_fm_broadcasting(df_clean1, dfau, autori)

    df_origin2 = pd.DataFrame(df_data2, columns=settings.COLUMNS_TV)
    df_origin2['Tiempo'] = pd.to_datetime(df_origin2['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
    df_clean2 = clean_data(fecha_inicio, fecha_fin, df_origin2, sheet_name2)
    df_original2 = simplify_tv_broadcasting(df_clean2, dfau, autori)

    if df_data3 is not None:
        df_origin3 = pd.DataFrame(df_data3, columns=settings.COLUMNS_AM)
        df_origin3['Tiempo'] = pd.to_datetime(df_origin3['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
        df_clean3 = clean_data(fecha_inicio, fecha_fin, df_origin3, sheet_name3)
        df_original3 = simplify_am_broadcasting(df_clean3, dfau, autori)
    else:
        df_clean3 = pd.DataFrame()
        df_original3 = pd.DataFrame()
    return df_original1, df_original2, df_original3, df_clean1, df_clean2, df_clean3


def translate_month(month: str) -> str:
    """
    Translate month abbreviation to full month name.

    Args:
        month (str): Month abbreviation.

    Returns:
        str: Full month name.
    """
    return MONTH_TRANSLATIONS.get(month, month)


def generate_month_year_vector(year1: int, year2: int) -> list[str]:
    """
    Generate a vector of month-year strings between the given years.

    Args:
        year1 (int): Start year.
        year2 (int): End year.

    Returns:
        List[str]: List of month-year strings.
    """
    vector = []
    for year in range(year1, year2 + 1):
        meses = [f"{translate_month(datetime.date(year, month, 1).strftime('%B'))}_{year}" for month in
                 range(1, 13)]
        vector.append(meses)
    return [num for elem in vector for num in elem]


def read_data_files(selected_options: dict, ciu: str, month_year: list[str], sheet_name1: str,
                    sheet_name2: str, sheet_name3: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read data files based on selected options and return data within the specified date range.

    Args:
        selected_options (dict): Selected options.
        ciu (str): City.
        month_year (List[str]): List of month-year strings.
        sheet_name1 (str): Sheet name for data set 1.
        sheet_name2 (str): Sheet name for data set 2.
        sheet_name3 (str): Sheet name for data set 3.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Data arrays within the selected date range.
    """
    # Convert start and end dates from string to datetime objects
    start_date = datetime.datetime.strptime(selected_options['start_date'], '%Y-%m-%d')
    end_date = datetime.datetime.strptime(selected_options['end_date'], '%Y-%m-%d')
    # Calculate the start and end indices for the month_year based on selected_options
    start_idx = month_year.index(f"{translate_month(start_date.strftime('%B'))}_{start_date.year}")
    end_idx = month_year.index(f"{translate_month(end_date.strftime('%B'))}_{end_date.year}")

    # Initialize empty lists to store data
    df_data1, df_data2, df_data3 = [], [], []

    # Iterate over the relevant months and read data
    for mes in month_year[start_idx:end_idx + 1]:
        df_data1.append(read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/FM_{ciu}_{mes}.csv', settings.COLUMNS_FM))
        df_data2.append(read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/TV_{ciu}_{mes}.csv', settings.COLUMNS_TV))

        if sheet_name3:
            df_data3.append(
                read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/AM_{ciu}_{mes}.csv', settings.COLUMNS_AM))

    # Concatenate the data and handle None for df_d3
    df_data1 = dd.concat(df_data1).compute() if df_data1 else dd.from_array(np.array([]))
    df_data2 = dd.concat(df_data2).compute() if df_data2 else dd.from_array(np.array([]))
    df_data3 = dd.concat(df_data3).compute() if df_data3 else None

    return df_data1, df_data2, df_data3


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


def clean_data(start_date: str, end_date: str, df_primary: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Clean and preprocess data based on specified date range.

    Args:
        start_date (str): Start date.
        end_date (str): End date.
        df_primary (pd.DataFrame): Primary DataFrame.
        sheet_name (str): Sheet name.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_tx = pd.read_excel(f'{settings.SERVER_ROUTE}/{settings.FILE_ESTACIONES}', sheet_name=sheet_name).fillna('-')
    df_tx['Frecuencia (Hz)'] = df_tx['Frecuencia (Hz)'].astype('float64')
    # Convert the Pandas DataFrame to a Dask DataFrame
    df_tx = dd.from_pandas(df_tx, npartitions=5)  # Adjust npartitions based on your dataset size and memory

    add_string1 = ' 00:00:01'
    add_string2 = ' 23:59:59'
    start_date += add_string1
    end_date += add_string2

    def convert(date_time):
        try:
            datetime_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                datetime_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Unable to parse date: {date_time}")
        return datetime_obj

    start_date = convert(start_date)
    end_date = convert(end_date)

    df_missing_dates = pd.DataFrame(
        [(t, f) for t in pd.date_range(start=start_date, end=end_date)
         for f in df_tx['Frecuencia (Hz)'].compute().tolist()],
        columns=('Tiempo', 'Frecuencia (Hz)'))

    df_missing_dates['Frecuencia (Hz)'] = df_missing_dates['Frecuencia (Hz)'].astype('float64')
    df_primary['Frecuencia (Hz)'] = df_primary['Frecuencia (Hz)'].astype('float64')

    df_full_dates = dd.concat([df_missing_dates, df_primary])

    df_complete = df_full_dates.merge(df_tx, how='right', on='Frecuencia (Hz)').compute()

    df_complete = df_complete[(df_complete.Tiempo >= start_date) & (df_complete.Tiempo <= end_date)].fillna(0)

    return df_complete


def read_and_process_aut(file_name: str, column_names: list[str], kind: str) -> pd.DataFrame:
    """
    Read and process authorization data.

    Args:
        file_name (str): Name of the authorization file.
        column_names (List[str]): List of column names.
        kind (str): Type of authorization.

    Returns:
        pd.DataFrame: Processed authorization DataFrame.
    """
    df = pd.read_excel(f'{settings.SERVER_ROUTE}/{file_name}', skiprows=1, usecols=column_names)
    df = df.fillna('-')
    df = df.rename(columns={
        'FECHA INGRESO': 'Fecha_ingreso',
        'FREC / CANAL': 'freq1',
        'CIUDAD PRINCIPAL COBERTURA': 'ciu',
        'No. OFICIO ARCOTEL': 'Oficio',
        'NOMBRE ESTACIÓN': 'est',
        f'FECHA INICIO {"SUSPENSION" if kind == "S" else "BAJA POTENCIA"}': 'Fecha_inicio',
        'FECHA OFICIO': 'Fecha_oficio',
        'DIAS': 'Plazo'
    })
    df['Tipo'] = pd.Series([kind for _ in range(len(df.index))])
    df = df[df.Oficio != '-']
    df = df[df.Fecha_inicio != '-']
    df['Fecha_ingreso'] = df['Fecha_ingreso'].replace({'-': ''})
    df['Fecha_ingreso'] = pd.to_datetime(df['Fecha_ingreso'])
    df['Fecha_oficio'] = df['Fecha_oficio'].replace({'-': ''})
    df['Fecha_oficio'] = pd.to_datetime(df['Fecha_oficio'])
    df['Fecha_inicio'] = df['Fecha_inicio'].replace({'-': ''})
    df['Fecha_inicio'] = pd.to_datetime(df['Fecha_inicio'])
    df['Fecha_fin'] = df['Fecha_inicio'] + pd.to_timedelta(df['Plazo'] - 1, unit='d')
    df['freq1'] = df['freq1'].replace('-', np.nan)
    df['freq1'] = pd.to_numeric(df['freq1'])

    def freq(row):
        """function to modify the values in freq1 column to present all in Hz, except if is a TV channel number"""
        if row['freq1'] >= 570 and row['freq1'] <= 1590:
            return row['freq1'] * 1000
        elif row['freq1'] >= 88 and row['freq1'] <= 108:
            return row['freq1'] * 1000000
        else:
            return row['freq1']

    # Create a new column in the df dataframe using the last function def freq(row)
    df['freq'] = df.apply(lambda row: freq(row), axis=1)
    df['freq'] = df['freq'].astype('float64')
    df = df.drop(columns=['freq1'])

    return df


def process_authorization_am_fm_df(dfau: pd.DataFrame, freq_range_start: int, freq_range_end: int,
                                   city: str) -> pd.DataFrame:
    """
    Process authorization data for AM/FM broadcasting.

    Args:
        dfau (pd.DataFrame): Authorization DataFrame.
        freq_range_start (int): Start of frequency range.
        freq_range_end (int): End of frequency range.
        city (str): City name.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    dfau_filtered = dfau[(dfau.freq >= freq_range_start) & (dfau.freq <= freq_range_end)]
    dfau_filtered = dfau_filtered.rename(columns={'freq': 'Frecuencia (Hz)', 'Fecha_inicio': 'Tiempo'})
    dfau_filtered = dfau_filtered.loc[dfau_filtered['ciu'] == city]
    dfau_filtered = dfau_filtered.drop(columns=['est'])
    dfau_filtered['Frecuencia (Hz)'] = dfau_filtered['Frecuencia (Hz)'].astype('float64')

    result_df = []
    for index, row in dfau_filtered.iterrows():
        for t in pd.date_range(start=row['Tiempo'], end=row['Fecha_fin']):
            result_df.append(
                (row['Frecuencia (Hz)'], row['Tipo'], row['Plazo'], t, row['Oficio'], row['Fecha_oficio'],
                 row['Fecha_fin'])
            )

    return pd.DataFrame(result_df, columns=(
        'Frecuencia (Hz)', 'Tipo', 'Plazo', 'Tiempo', 'Oficio', 'Fecha_oficio', 'Fecha_fin'))


def process_authorization_tv_df(dfau: pd.DataFrame, freq_range_start: int, freq_range_end: int,
                                city: str) -> pd.DataFrame:
    """
    Process authorization data for TV broadcasting.

    Args:
        dfau (pd.DataFrame): Authorization DataFrame.
        freq_range_start (int): Start of frequency range.
        freq_range_end (int): End of frequency range.
        city (str): City name.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    dfau_filtered = dfau[(dfau.freq >= freq_range_start) & (dfau.freq <= freq_range_end)]
    dfau_filtered = dfau_filtered.rename(columns={'freq': 'Canal (Número)', 'Fecha_inicio': 'Tiempo'})
    dfau_filtered = dfau_filtered.loc[dfau_filtered['ciu'] == city]
    dfau_filtered = dfau_filtered.drop(columns=['est'])
    dfau_filtered['Canal (Número)'] = dfau_filtered['Canal (Número)'].astype('float64')

    result_df = []
    for index, row in dfau_filtered.iterrows():
        for t in pd.date_range(start=row['Tiempo'], end=row['Fecha_fin']):
            result_df.append(
                (row['Canal (Número)'], row['Tipo'], row['Plazo'], t, row['Oficio'], row['Fecha_oficio'],
                 row['Fecha_fin'])
            )

    return pd.DataFrame(result_df, columns=(
        'Canal (Número)', 'Tipo', 'Plazo', 'Tiempo', 'Oficio', 'Fecha_oficio', 'Fecha_fin'))


def merge_authorization_with_data(df_authorization: pd.DataFrame, df_data: pd.DataFrame,
                                  merge_columns: list[str]) -> pd.DataFrame:
    """
    Merge authorization data with broadcasting data.

    Args:
        df_authorization (pd.DataFrame): Authorization DataFrame.
        df_data (pd.DataFrame): Broadcasting DataFrame.
        merge_columns (List[str]): List of columns to merge on.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    df_authorization = df_authorization.rename(columns={'Fecha_inicio': 'Tiempo'})

    result_df = df_authorization.merge(df_data, how='right', on=merge_columns)
    return result_df.fillna('-')


def create_pivot_table(df: pd.DataFrame, index: list, values: list, columns: list,
                       aggfunc: dict) -> pd.DataFrame:
    """
    Create a pivot table with specified parameters.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - index: list
        List of index columns in the form [pd.Grouper(key='ColumnName')].
    - values: list
        List of values to be aggregated.
    - columns: list
        List of columns to be used for pivoting.
    - aggfunc: dict
        Dictionary specifying the aggregation functions for each value.

    Returns:
    - DataFrame
        The resulting pivot table.
    """
    # Make the pivot table with the data structured in the way we want to show in the report
    pivot_table = pd.pivot_table(df, index=index, values=values, columns=columns, aggfunc=aggfunc).round(2)

    # Rename column
    pivot_table = pivot_table.rename(columns={'Fecha_fin': 'Fin de Autorización'})

    # Transpose the DataFrame
    pivot_table = pivot_table.T

    # Replace 0 with '-'
    pivot_table = pivot_table.replace(0, '-')

    # Reset the index (unstack)
    pivot_table = pivot_table.reset_index()

    # Sort the DataFrame
    sorter = ['Level (dBµV/m)', 'Bandwidth (Hz)', 'Fin de Autorización']
    pivot_table.level_0 = pivot_table.level_0.astype("category")
    pivot_table.level_0 = pivot_table.level_0.cat.set_categories(sorter)
    pivot_table = pivot_table.sort_values(['level_0', 'Frecuencia (Hz)'])

    # Rename columns
    final_result = pivot_table.rename(columns={'level_0': 'Param'})

    return final_result


def simplify_fm_broadcasting(df: pd.DataFrame, dfau: pd.DataFrame, autori: str) -> DataFrame:
    """
    Simplify FM broadcasting data.

    Args:
        df9 (pd.DataFrame): FM broadcasting DataFrame.
        dfau1 (pd.DataFrame): Authorization DataFrame.
        autori (str): Authorization type.

    Returns:
        pd.DataFrame: Simplified DataFrame.
    """
    df_authorization = process_authorization_am_fm_df(dfau, 87700000, 108100000, autori)
    df_merge_data_aut = merge_authorization_with_data(df_authorization, df, ['Tiempo', 'Frecuencia (Hz)'])
    df_data_aut = df_merge_data_aut.groupby(['Frecuencia (Hz)', 'Estación', pd.Grouper(key='Tiempo', freq='D')]).agg(
        {'Level (dBµV/m)': 'max', 'Bandwidth (Hz)': np.average, 'Fecha_fin': 'max'})
    df_data_aut = df_data_aut.reset_index().sort_values(by='Tiempo', ascending=True)
    new_order = ['Tiempo', 'Frecuencia (Hz)', 'Estación', 'Level (dBµV/m)', 'Bandwidth (Hz)', 'Fecha_fin']
    df_data_aut = df_data_aut[new_order]
    return df_data_aut


def simplify_tv_broadcasting(df: pd.DataFrame, dfau: pd.DataFrame, autori: str) -> pd.DataFrame:
    """
    Simplify TV broadcasting data.

    Args:
        df10 (pd.DataFrame): TV broadcasting DataFrame.
        dfau1 (pd.DataFrame): Authorization DataFrame.
        autori (str): Authorization type.

    Returns:
        pd.DataFrame: Simplified DataFrame.
    """
    df_authorization = process_authorization_tv_df(dfau, 2, 51, autori)
    df_merge_data_aut = merge_authorization_with_data(df_authorization, df, ['Tiempo', 'Canal (Número)'])
    df_data_aut = df_merge_data_aut.groupby(
        ['Frecuencia (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital', pd.Grouper(key='Tiempo', freq='D')]).agg(
        {'Level (dBµV/m)': 'max', 'Fecha_fin': 'max'})
    df_data_aut = df_data_aut.reset_index().sort_values(by='Tiempo', ascending=True)
    new_order = ['Tiempo', 'Frecuencia (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital', 'Level (dBµV/m)',
                 'Fecha_fin']
    df_data_aut = df_data_aut[new_order]
    return df_data_aut


def simplify_am_broadcasting(df: pd.DataFrame, dfau: pd.DataFrame, autori: str) -> pd.DataFrame:
    """
    Simplify AM broadcasting data.

    Args:
        df17 (pd.DataFrame): AM broadcasting DataFrame.
        dfau1 (pd.DataFrame): Authorization DataFrame.
        autori (str): Authorization type.

    Returns:
        pd.DataFrame: Simplified DataFrame.
    """
    df_authorization = process_authorization_am_fm_df(dfau, 570000, 1590000, autori)
    df_merge_data_aut = merge_authorization_with_data(df_authorization, df, ['Tiempo', 'Frecuencia (Hz)'])
    df_data_aut = df_merge_data_aut.groupby(['Frecuencia (Hz)', 'Estación', pd.Grouper(key='Tiempo', freq='D')]).agg(
        {'Level (dBµV/m)': 'max', 'Bandwidth (Hz)': np.average, 'Fecha_fin': 'max'})
    df_data_aut = df_data_aut.reset_index().sort_values(by='Tiempo', ascending=True)
    new_order = ['Tiempo', 'Frecuencia (Hz)', 'Estación', 'Level (dBµV/m)', 'Bandwidth (Hz)', 'Fecha_fin']
    df_data_aut = df_data_aut[new_order]
    return df_data_aut


def convert_timestamps_to_strings(df):
    # Check if 'Tiempo' is in the columns of the DataFrame
    if 'Tiempo' in df.columns:
        # Convert 'Tiempo' to string format only with year, month, and day
        df['Tiempo'] = df['Tiempo'].dt.strftime('%Y-%m-%d')

    # Convert MultiIndex column headers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(
            lambda x: [(y.strftime('%Y-%m-%d') if isinstance(y, pd.Timestamp) else y) for y in x])
    else:
        df.columns = [x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x for x in df.columns]

    # Convert MultiIndex index
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.map(lambda x: [(y.strftime('%Y-%m-%d') if isinstance(y, pd.Timestamp) else y) for y in x])
    else:
        df.index = [x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x for x in df.index]

    return df


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
