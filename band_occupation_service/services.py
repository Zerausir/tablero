import datetime
import pandas as pd
import psycopg2
from django.conf import settings
from django.views.decorators.cache import cache_page

from .utils import convert_start_date, convert_end_date

pd.set_option('future.no_silent_downcasting', True)


def get_db_connection():
    return psycopg2.connect(
        dbname=settings.DBNAME,
        user=settings.USER,
        password=settings.PASSWORD,
        host=settings.HOST,
        port=settings.PORT
    )


@cache_page(60 * 10)
def fetch_data_from_db(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    city = request.GET.get('city')
    last_query_time = request.GET.get('last_query_time')
    start_freq = float(request.GET.get('start_freq')) * 1e6
    end_freq = float(request.GET.get('end_freq')) * 1e6

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    if last_query_time:
        last_query_time = datetime.datetime.strptime(last_query_time, '%Y-%m-%d %H:%M:%S')

    conn = get_db_connection()

    # Query for FM data
    query_fm = """
        SELECT tiempo, frecuencia_hz, level_dbuv_m, offset_hz, fm_hz as modulation_value, bandwidth_hz, city, 'FM' as tipo
        FROM band_occupation
        WHERE city = %s AND tiempo BETWEEN %s AND %s AND frecuencia_hz BETWEEN %s AND %s
    """

    # Query for AM data
    query_am = """
        SELECT tiempo, frecuencia_hz, level_dbuv_m, offset_hz, am_percentage as modulation_value, bandwidth_hz, city, 'AM' as tipo
        FROM band_occupation_am
        WHERE city = %s AND tiempo BETWEEN %s AND %s AND frecuencia_hz BETWEEN %s AND %s
    """

    # Query for DN data
    query_dn = """
            SELECT tiempo, frecuencia_hz, level_dbuv_m, city
            FROM band_occupation_dn
            WHERE city = %s AND tiempo BETWEEN %s AND %s AND frecuencia_hz BETWEEN %s AND %s
        """

    params = [city, start_date, end_date, start_freq, end_freq]

    if last_query_time is not None:
        query_fm += " AND tiempo > %s"
        query_am += " AND tiempo > %s"
        query_dn += " AND tiempo > %s"
        params.append(last_query_time)

    # Fetch data from both tables
    df_fm = pd.read_sql(query_fm, conn, params=params)
    df_am = pd.read_sql(query_am, conn, params=params)
    df_dn = pd.read_sql(query_dn, conn, params=params)

    # Combine the dataframes
    df = pd.concat([df_fm, df_am, df_dn], ignore_index=True)

    conn.close()
    return df


def customize_data(request):
    ciudad = request.GET.get('city')
    fecha_inicio = convert_start_date(request.GET.get('start_date'))
    fecha_fin = convert_end_date(request.GET.get('end_date'))
    start_freq = float(request.GET.get('start_freq')) * 1e6  # Convert to Hz
    end_freq = float(request.GET.get('end_freq')) * 1e6  # Convert to Hz

    df_data = fetch_data_from_db(request)
    print(f"Fetched data for {ciudad} from {fecha_inicio} to {fecha_fin}: {df_data.head()}")

    if df_data.empty:
        print("No data found for the given parameters.")
        return pd.DataFrame(), pd.DataFrame()

    df_data = df_data[df_data['frecuencia_hz'].between(start_freq, end_freq)]

    df_original = df_data.copy()
    df_clean = clean_data(fecha_inicio, fecha_fin, df_original)
    df_original = df_clean.sort_values(by='tiempo', ascending=False).fillna('-')
    df_clean = df_clean.groupby(['frecuencia_hz', pd.Grouper(key='tiempo', freq='D')]).agg(
        {'level_dbuv_m': 'max'}).reset_index()

    return df_original, df_clean


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
