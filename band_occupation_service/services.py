import datetime
import pandas as pd
import psycopg
from django.conf import settings

from .utils import convert_start_date, convert_end_date

pd.set_option('future.no_silent_downcasting', True)


def get_db_connection():
    return psycopg.connect(
        dbname=settings.DBNAME,
        user=settings.USER,
        password=settings.PASSWORD,
        host=settings.HOST,
        port=settings.PORT
    )


def fetch_data_from_db(start_date, end_date, city):
    conn = get_db_connection()
    query = """
        SELECT *
        FROM band_occupation
        WHERE city = %s AND tiempo BETWEEN %s AND %s
    """
    print(f"Executing query: {query} with params: city={city}, start_date={start_date}, end_date={end_date}")
    df = pd.read_sql(query, conn, params=[city, start_date, end_date])
    conn.close()
    print(f"Data fetched: {df.head()}")
    return df


def customize_data(selected_options: dict) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ciudad = selected_options['city']
    fecha_inicio = convert_start_date(selected_options['start_date'])
    fecha_fin = convert_end_date(selected_options['end_date'])

    df_data = fetch_data_from_db(fecha_inicio, fecha_fin, ciudad)
    print(f"Fetched data for {ciudad} from {fecha_inicio} to {fecha_fin}: {df_data.head()}")

    if df_data.empty:
        print("No data found for the given parameters.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_data1 = df_data[df_data['frecuencia_hz'].between(703 * 1e6, 733 * 1e6)]
    df_data2 = df_data[df_data['frecuencia_hz'].between(758 * 1e6, 788 * 1e6)]
    df_data3 = df_data[df_data['frecuencia_hz'].between(2500 * 1e6, 2690 * 1e6)]

    df_original1 = df_data1.copy()
    df_clean1 = clean_data(fecha_inicio, fecha_fin, df_original1)
    df_original1 = df_clean1.sort_values(by='tiempo', ascending=False).fillna('-')
    df_clean1 = df_clean1.groupby(['frecuencia_hz', pd.Grouper(key='tiempo', freq='D')]).agg(
        {'level_dbuv_m': 'max'}).reset_index()

    df_original2 = df_data2.copy()
    df_clean2 = clean_data(fecha_inicio, fecha_fin, df_original2)
    df_original2 = df_clean2.sort_values(by='tiempo', ascending=False).fillna('-')
    df_clean2 = df_clean2.groupby(['frecuencia_hz', pd.Grouper(key='tiempo', freq='D')]).agg(
        {'level_dbuv_m': 'max'}).reset_index()

    df_original3 = df_data3.copy()
    df_clean3 = clean_data(fecha_inicio, fecha_fin, df_original3)
    df_original3 = df_clean3.sort_values(by='tiempo', ascending=False).fillna('-')
    df_clean3 = df_clean3.groupby(['frecuencia_hz', pd.Grouper(key='tiempo', freq='D')]).agg(
        {'level_dbuv_m': 'max'}).reset_index()

    return df_original1, df_original2, df_original3, df_clean1, df_clean2, df_clean3


def clean_data(start_date: datetime.datetime, end_date: datetime.datetime, df_primary: pd.DataFrame) -> pd.DataFrame:
    start_date_day = start_date.strftime('%Y-%m-%d')
    start_date_day = pd.to_datetime(start_date_day)
    end_date_day = end_date.strftime('%Y-%m-%d')
    end_date_day = pd.to_datetime(end_date_day)

    df_primary['tiempo'] = pd.to_datetime(df_primary['tiempo'])

    all_dates = pd.date_range(start=start_date_day, end=end_date_day, freq='D')

    existing_dates = df_primary['tiempo'].dt.normalize().unique()
    missing_dates = all_dates.difference(existing_dates)

    if not missing_dates.empty:
        df_missing_dates = pd.DataFrame(
            [(t, f) for t in missing_dates
             for f in df_primary['frecuencia_hz'].unique()],
            columns=('tiempo', 'frecuencia_hz')
        )
        df_missing_dates['frecuencia_hz'] = df_missing_dates['frecuencia_hz'].astype('float64')

        df_primary = pd.concat([df_primary, df_missing_dates], ignore_index=True)

    df_complete = df_primary
    df_complete['tiempo'] = pd.to_datetime(df_complete['tiempo'], format="%d/%m/%Y %H:%M:%S.%f")

    df_complete = df_complete[(df_complete['tiempo'] >= start_date) & (df_complete['tiempo'] <= end_date)]

    return df_complete
