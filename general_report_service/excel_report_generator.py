import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import xlsxwriter
from datetime import datetime
from django.http import HttpRequest, QueryDict
from .services import fm_fetch_data_from_db, tv_fetch_data_from_db, am_fetch_data_from_db


def generate_excel_report(ciudad, fecha_inicio, fecha_fin, umbral_am, umbral_fm, umbral_tv):
    """
    Generate the Excel report with the same structure as sacer.py

    Args:
        ciudad: City name
        fecha_inicio: Start date (datetime object)
        fecha_fin: End date (datetime object)
        umbral_am: AM threshold
        umbral_fm: FM threshold
        umbral_tv: TV threshold

    Returns:
        BytesIO: Excel file in memory
    """

    # Normalizar el nombre de la ciudad (capitalizar correctamente)
    ciudad_normalized = ciudad.strip().title()

    # Create request object
    request = HttpRequest()
    request.GET = QueryDict(mutable=True)
    request.GET['start_date'] = fecha_inicio.strftime('%Y-%m-%d')
    request.GET['end_date'] = fecha_fin.strftime('%Y-%m-%d')
    request.GET['city'] = ciudad  # Usar el nombre original para la consulta

    # Fetch data from database
    df_data1 = fm_fetch_data_from_db(request)
    df_data2 = tv_fetch_data_from_db(request)

    # SIEMPRE intentar obtener datos AM para Quito, Guayaquil, Cuenca
    df_data3 = None
    if ciudad_normalized in ['Quito', 'Guayaquil', 'Cuenca']:
        print(f"Fetching AM data for {ciudad_normalized}")
        df_data3 = am_fetch_data_from_db(request)
        if df_data3 is not None:
            print(f"AM data fetched: {len(df_data3)} rows")
        else:
            print(f"No AM data returned for {ciudad_normalized}")

    # Process data
    df9, df10, df17, df_original1, df_original2, df_original3 = process_data(
        df_data1, df_data2, df_data3, fecha_inicio, fecha_fin, ciudad_normalized
    )

    print(f"After process_data - df17 (AM) empty: {df17.empty}, shape: {df17.shape if not df17.empty else 'N/A'}")

    # Generate pivot tables for main sheets
    df_final5, df_final6, df_final9 = generate_pivot_tables(
        df9, df10, df17, fecha_inicio, fecha_fin, ciudad_normalized
    )

    print(
        f"After pivot tables - df_final9 (AM) empty: {df_final9.empty}, shape: {df_final9.shape if not df_final9.empty else 'N/A'}")

    # Generate occupation data and plots
    contar1, contar2, contar3, image1, image2, image3 = generate_occupation_data(
        df9, df10, df17, umbral_fm, umbral_tv, umbral_am, ciudad_normalized, fecha_inicio, fecha_fin
    )

    # Create Excel file in memory
    output = BytesIO()
    excel_file = create_excel_file(
        output, df_final5, df_final6, df_final9, contar1, contar2, contar3,
        image1, image2, image3, df_original1, df_original2, df_original3,
        df9, df10, df17, ciudad_normalized, fecha_inicio, fecha_fin,
        umbral_am, umbral_fm, umbral_tv
    )

    output.seek(0)
    return output


def process_data(df_data1, df_data2, df_data3, fecha_inicio, fecha_fin, ciudad):
    """Process raw data from database"""

    # Process FM data
    if df_data1 is not None and not df_data1.empty:
        df9 = df_data1.copy()
        df9['tiempo'] = pd.to_datetime(df9['tiempo'])
        df9 = df9[(df9['tiempo'] >= fecha_inicio) & (df9['tiempo'] <= fecha_fin)]
        df9 = df9.rename(columns={
            'tiempo': 'Tiempo',
            'frecuencia_hz': 'Frecuencia (Hz)',
            'level_dbuv_m': 'Level (dBµV/m)',
            'bandwidth_hz': 'Bandwidth (Hz)',
            'estacion': 'Estación',
            'potencia': 'Potencia',
            'bw_asignado': 'BW Asignado',
            'fecha_inicio': 'Fecha_inicio',
            'fecha_fin': 'Fecha_fin',
            'tipo': 'Tipo'
        })
        df9 = df9.fillna('-')
        df_original1 = df9.sort_values(by='Tiempo', ascending=False)
    else:
        df9 = pd.DataFrame()
        df_original1 = pd.DataFrame()

    # Process TV data
    if df_data2 is not None and not df_data2.empty:
        df10 = df_data2.copy()
        df10['tiempo'] = pd.to_datetime(df10['tiempo'])
        df10 = df10[(df10['tiempo'] >= fecha_inicio) & (df10['tiempo'] <= fecha_fin)]
        df10 = df10.rename(columns={
            'tiempo': 'Tiempo',
            'frecuencia_hz': 'Frecuencia (Hz)',
            'level_dbuv_m': 'Level (dBµV/m)',
            'estacion': 'Estación',
            'canal_numero': 'Canal (Número)',
            'analogico_digital': 'Analógico/Digital',
            'fecha_inicio': 'Fecha_inicio',
            'fecha_fin': 'Fecha_fin',
            'tipo': 'Tipo'
        })
        df10 = df10.fillna('-')
        df_original2 = df10.sort_values(by='Tiempo', ascending=False)
    else:
        df10 = pd.DataFrame()
        df_original2 = pd.DataFrame()

    # Process AM data
    if df_data3 is not None and not df_data3.empty:
        df17 = df_data3.copy()
        df17['tiempo'] = pd.to_datetime(df17['tiempo'])
        df17 = df17[(df17['tiempo'] >= fecha_inicio) & (df17['tiempo'] <= fecha_fin)]
        df17 = df17.rename(columns={
            'tiempo': 'Tiempo',
            'frecuencia_hz': 'Frecuencia (Hz)',
            'level_dbuv_m': 'Level (dBµV/m)',
            'bandwidth_hz': 'Bandwidth (Hz)',
            'estacion': 'Estación',
            'fecha_inicio': 'Fecha_inicio',
            'fecha_fin': 'Fecha_fin',
            'tipo': 'Tipo'
        })
        df17 = df17.fillna('-')
        df_original3 = df17.sort_values(by='Tiempo', ascending=False)
    else:
        df17 = pd.DataFrame()
        df_original3 = pd.DataFrame()

    return df9, df10, df17, df_original1, df_original2, df_original3


def generate_pivot_tables(df9, df10, df17, fecha_inicio, fecha_fin, ciudad):
    """Generate pivot tables for main sheets"""

    # Calculate if it's a single month
    Year1 = fecha_inicio.year
    Year2 = fecha_fin.year
    Mes_inicio = fecha_inicio.strftime('%B')
    Mes_fin = fecha_fin.strftime('%B')
    is_single_month = (Year1 == Year2 and Mes_inicio == Mes_fin)

    # Crear rango completo de fechas
    all_dates = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')

    # FM pivot table
    if not df9.empty:
        df11 = df9.groupby(by=[
            pd.Grouper(key='Tiempo', freq='D'),
            pd.Grouper(key='Frecuencia (Hz)'),
            pd.Grouper(key='Estación'),
            pd.Grouper(key='Potencia'),
            pd.Grouper(key='BW Asignado')
        ]).agg({
            'Level (dBµV/m)': 'max',
            'Bandwidth (Hz)': 'mean'
        }).reset_index()

        # Obtener todas las combinaciones únicas de frecuencia/estación/potencia/bw
        unique_combinations = df11[['Frecuencia (Hz)', 'Estación', 'Potencia', 'BW Asignado']].drop_duplicates()

        # Crear un DataFrame con todas las fechas para cada combinación
        complete_data = []
        for _, combo in unique_combinations.iterrows():
            for date in all_dates:
                complete_data.append({
                    'Tiempo': date,
                    'Frecuencia (Hz)': combo['Frecuencia (Hz)'],
                    'Estación': combo['Estación'],
                    'Potencia': combo['Potencia'],
                    'BW Asignado': combo['BW Asignado'],
                    'Level (dBµV/m)': 0,
                    'Bandwidth (Hz)': 0
                })

        df_complete = pd.DataFrame(complete_data)

        # Merge con los datos reales, dando prioridad a los datos reales
        df11_full = df_complete.merge(
            df11,
            on=['Tiempo', 'Frecuencia (Hz)', 'Estación', 'Potencia', 'BW Asignado'],
            how='left',
            suffixes=('_default', '')
        )

        # Usar los valores reales si existen, sino usar 0
        df11_full['Level (dBµV/m)'] = df11_full['Level (dBµV/m)'].fillna(df11_full['Level (dBµV/m)_default'])
        df11_full['Bandwidth (Hz)'] = df11_full['Bandwidth (Hz)'].fillna(df11_full['Bandwidth (Hz)_default'])
        df11_full = df11_full.drop(columns=['Level (dBµV/m)_default', 'Bandwidth (Hz)_default'], errors='ignore')

        df_final1 = pd.pivot_table(df11_full,
                                   index=[pd.Grouper(key='Tiempo')],
                                   values=['Level (dBµV/m)', 'Bandwidth (Hz)'],
                                   columns=['Frecuencia (Hz)', 'Estación', 'Potencia', 'BW Asignado'],
                                   aggfunc={'Level (dBµV/m)': max, 'Bandwidth (Hz)': np.average}).round(2)
        df_final1 = df_final1.T
        df_final3 = df_final1.replace(0, '-')
        df_final3 = df_final3.reset_index()

        sorter = ['Level (dBµV/m)', 'Bandwidth (Hz)']
        df_final3.level_0 = df_final3.level_0.astype("category")
        df_final3.level_0 = df_final3.level_0.cat.set_categories(sorter)
        df_final3 = df_final3.sort_values(['level_0', 'Frecuencia (Hz)'])
        df_final5 = df_final3.rename(columns={'level_0': 'Param'})

        if is_single_month:
            df_final5['Promedio'] = df_final5.drop(
                ['Param', 'Frecuencia (Hz)', 'Estación', 'Potencia', 'BW Asignado'],
                axis=1).replace('-', np.nan).apply(lambda x: x.mean(), axis=1).round(2)
            df_final5['Observaciones'] = ''
        else:
            df_final5['Observaciones'] = ''

        df_final5 = df_final5.rename(columns={'Param': 'Parámetro'}).set_index('Parámetro')
    else:
        df_final5 = pd.DataFrame()

    # TV pivot table
    if not df10.empty:
        df12 = df10.groupby(by=[
            pd.Grouper(key='Tiempo', freq='D'),
            pd.Grouper(key='Frecuencia (Hz)'),
            pd.Grouper(key='Estación'),
            pd.Grouper(key='Canal (Número)'),
            pd.Grouper(key='Analógico/Digital'),
        ]).agg({
            'Level (dBµV/m)': 'max'
        }).reset_index()

        # Obtener todas las combinaciones únicas
        unique_combinations = df12[
            ['Frecuencia (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital']].drop_duplicates()

        # Crear un DataFrame con todas las fechas para cada combinación
        complete_data = []
        for _, combo in unique_combinations.iterrows():
            for date in all_dates:
                complete_data.append({
                    'Tiempo': date,
                    'Frecuencia (Hz)': combo['Frecuencia (Hz)'],
                    'Estación': combo['Estación'],
                    'Canal (Número)': combo['Canal (Número)'],
                    'Analógico/Digital': combo['Analógico/Digital'],
                    'Level (dBµV/m)': 0
                })

        df_complete = pd.DataFrame(complete_data)

        # Merge con los datos reales
        df12_full = df_complete.merge(
            df12,
            on=['Tiempo', 'Frecuencia (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital'],
            how='left',
            suffixes=('_default', '')
        )

        df12_full['Level (dBµV/m)'] = df12_full['Level (dBµV/m)'].fillna(df12_full['Level (dBµV/m)_default'])
        df12_full = df12_full.drop(columns=['Level (dBµV/m)_default'], errors='ignore')

        df_final2 = pd.pivot_table(df12_full,
                                   index=[pd.Grouper(key='Tiempo')],
                                   values=['Level (dBµV/m)'],
                                   columns=['Frecuencia (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital'],
                                   aggfunc={'Level (dBµV/m)': max}).round(2)
        df_final2 = df_final2.T
        df_final4 = df_final2.replace(0, '-')
        df_final4 = df_final4.reset_index()

        sorter1 = ['Level (dBµV/m)']
        df_final4.level_0 = df_final4.level_0.astype("category")
        df_final4.level_0 = df_final4.level_0.cat.set_categories(sorter1)
        df_final4 = df_final4.sort_values(['level_0', 'Frecuencia (Hz)'])
        df_final6 = df_final4.rename(columns={'level_0': 'Param'})

        if is_single_month:
            df_final6['Promedio'] = df_final6.drop(
                ['Param', 'Frecuencia (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital'],
                axis=1).replace('-', np.nan).apply(lambda x: x.mean(), axis=1).round(2)
            df_final6['Observaciones'] = ''
        else:
            df_final6['Observaciones'] = ''

        df_final6 = df_final6.rename(columns={'Param': 'Parámetro'}).set_index('Parámetro')
    else:
        df_final6 = pd.DataFrame()

    # AM pivot table
    if not df17.empty and ciudad in ['Quito', 'Guayaquil', 'Cuenca']:
        df18 = df17.groupby(by=[
            pd.Grouper(key='Tiempo', freq='D'),
            pd.Grouper(key='Frecuencia (Hz)'),
            pd.Grouper(key='Estación')
        ]).agg({
            'Level (dBµV/m)': 'max',
            'Bandwidth (Hz)': 'mean'
        }).reset_index()

        # Obtener todas las combinaciones únicas
        unique_combinations = df18[['Frecuencia (Hz)', 'Estación']].drop_duplicates()

        # Crear un DataFrame con todas las fechas para cada combinación
        complete_data = []
        for _, combo in unique_combinations.iterrows():
            for date in all_dates:
                complete_data.append({
                    'Tiempo': date,
                    'Frecuencia (Hz)': combo['Frecuencia (Hz)'],
                    'Estación': combo['Estación'],
                    'Level (dBµV/m)': 0,
                    'Bandwidth (Hz)': 0
                })

        df_complete = pd.DataFrame(complete_data)

        # Merge con los datos reales
        df18_full = df_complete.merge(
            df18,
            on=['Tiempo', 'Frecuencia (Hz)', 'Estación'],
            how='left',
            suffixes=('_default', '')
        )

        df18_full['Level (dBµV/m)'] = df18_full['Level (dBµV/m)'].fillna(df18_full['Level (dBµV/m)_default'])
        df18_full['Bandwidth (Hz)'] = df18_full['Bandwidth (Hz)'].fillna(df18_full['Bandwidth (Hz)_default'])
        df18_full = df18_full.drop(columns=['Level (dBµV/m)_default', 'Bandwidth (Hz)_default'], errors='ignore')

        df_final7 = pd.pivot_table(df18_full,
                                   index=[pd.Grouper(key='Tiempo')],
                                   values=['Level (dBµV/m)', 'Bandwidth (Hz)'],
                                   columns=['Frecuencia (Hz)', 'Estación'],
                                   aggfunc={'Level (dBµV/m)': max, 'Bandwidth (Hz)': np.average}).round(2)
        df_final7 = df_final7.T
        df_final8 = df_final7.replace(0, '-')
        df_final8 = df_final8.reset_index()

        sorter = ['Level (dBµV/m)', 'Bandwidth (Hz)']
        df_final8.level_0 = df_final8.level_0.astype("category")
        df_final8.level_0 = df_final8.level_0.cat.set_categories(sorter)
        df_final8 = df_final8.sort_values(['level_0', 'Frecuencia (Hz)'])
        df_final9 = df_final8.rename(columns={'level_0': 'Param'})

        if is_single_month:
            df_final9['Promedio'] = df_final9.drop(['Param', 'Frecuencia (Hz)', 'Estación'], axis=1).replace(
                '-', np.nan).apply(lambda x: x.mean(), axis=1).round(2)
            df_final9['Observaciones'] = ''
        else:
            df_final9['Observaciones'] = ''

        df_final9 = df_final9.rename(columns={'Param': 'Parámetro'}).set_index('Parámetro')
    else:
        df_final9 = pd.DataFrame()

    return df_final5, df_final6, df_final9


def generate_occupation_data(df9, df10, df17, umbral_fm, umbral_tv, umbral_am, ciudad, fecha_inicio, fecha_fin):
    """Generate occupation data and matplotlib plots"""

    fecha_init = fecha_inicio.strftime('%Y-%m-%d')
    fecha_end = fecha_fin.strftime('%Y-%m-%d')

    # Helper function for threshold
    def threshold_level(row, threshold):
        return row['level'] if row['level'] >= threshold else None

    # FM Occupation
    image1 = None
    contar1 = pd.DataFrame()
    if not df9.empty:
        df13 = df9.drop(columns=['Potencia', 'BW Asignado', 'Fecha_fin', 'Tipo'], errors='ignore').copy()
        df13['label'] = df13['Frecuencia (Hz)'].astype(str) + ' - ' + df13['Estación'].astype(str)
        df13 = df13.rename(columns={'Level (dBµV/m)': 'level', 'Tiempo': 'tiempo', 'Frecuencia (Hz)': 'freq'})
        df19 = df13.copy()

        series1 = df19.drop(columns=['tiempo', 'Estación'], errors='ignore').copy()
        series1['above_threshold'] = series1.apply(threshold_level, axis=1, threshold=umbral_fm)
        occupation_counts1 = series1.groupby('label').agg(
            total_measurements=('level', 'count'),
            above_threshold_count=('above_threshold', 'count')
        )
        occupation_counts1['occupation'] = ((occupation_counts1['above_threshold_count'] /
                                             occupation_counts1['total_measurements']) * 100).round(6)
        contar1 = occupation_counts1.drop(columns=['total_measurements', 'above_threshold_count'])
        contar1 = contar1.rename(columns={'occupation': 'Ocupación (%)'})
        contar1.index.names = ['Frecuencia (Hz) - Estación']

        # Generate matplotlib plot
        image1 = generate_occupation_plot(df19, occupation_counts1, ciudad, umbral_fm,
                                          fecha_init, fecha_end, 'FM')

    # TV Occupation
    image2 = None
    contar2 = pd.DataFrame()
    if not df10.empty:
        df14 = df10.drop(columns=['Canal (Número)', 'Analógico/Digital', 'Fecha_fin', 'Tipo'], errors='ignore').copy()
        df14['label'] = df14['Frecuencia (Hz)'].astype(str) + ' - ' + df14['Estación'].astype(str)
        df14 = df14.rename(columns={'Level (dBµV/m)': 'level', 'Tiempo': 'tiempo', 'Frecuencia (Hz)': 'freq'})
        df20 = df14.copy()

        series2 = df20.drop(columns=['tiempo', 'Estación'], errors='ignore').copy()
        series2['above_threshold'] = series2.apply(threshold_level, axis=1, threshold=umbral_tv)
        occupation_counts2 = series2.groupby('label').agg(
            total_measurements=('level', 'count'),
            above_threshold_count=('above_threshold', 'count')
        )
        occupation_counts2['occupation'] = ((occupation_counts2['above_threshold_count'] /
                                             occupation_counts2['total_measurements']) * 100).round(6)
        contar2 = occupation_counts2.drop(columns=['total_measurements', 'above_threshold_count'])
        contar2 = contar2.rename(columns={'occupation': 'Ocupación (%)'})
        contar2.index.names = ['Frecuencia (Hz) - Estación']

        # Generate matplotlib plot
        image2 = generate_occupation_plot(df20, occupation_counts2, ciudad, umbral_tv,
                                          fecha_init, fecha_end, 'TV')

    # AM Occupation
    image3 = None
    contar3 = pd.DataFrame()
    if not df17.empty and ciudad in ['Quito', 'Guayaquil', 'Cuenca']:
        df18 = df17.drop(columns=['Fecha_fin', 'Tipo'], errors='ignore').copy()
        df18['label'] = df18['Frecuencia (Hz)'].astype(str) + ' - ' + df18['Estación'].astype(str)
        df18 = df18.rename(columns={'Level (dBµV/m)': 'level', 'Tiempo': 'tiempo', 'Frecuencia (Hz)': 'freq'})
        df21 = df18.copy()

        series3 = df21.drop(columns=['tiempo', 'Estación', 'Bandwidth (Hz)'], errors='ignore').copy()
        series3['above_threshold'] = series3.apply(threshold_level, axis=1, threshold=umbral_am)
        occupation_counts3 = series3.groupby('label').agg(
            total_measurements=('level', 'count'),
            above_threshold_count=('above_threshold', 'count')
        )
        occupation_counts3['occupation'] = ((occupation_counts3['above_threshold_count'] /
                                             occupation_counts3['total_measurements']) * 100).round(6)
        contar3 = occupation_counts3.drop(columns=['total_measurements', 'above_threshold_count'])
        contar3 = contar3.rename(columns={'occupation': 'Ocupación (%)'})
        contar3.index.names = ['Frecuencia (Hz) - Estación']

        # Generate matplotlib plot
        image3 = generate_occupation_plot(df21, occupation_counts3, ciudad, umbral_am,
                                          fecha_init, fecha_end, 'AM')

    return contar1, contar2, contar3, image1, image2, image3


def generate_occupation_plot(df, occupation_counts, ciudad, umbral, fecha_init, fecha_end, banda):
    """Generate matplotlib occupation plot (scatter + heatmap)"""

    label_values = df['label'].unique()
    label_dict = {label: i for i, label in enumerate(label_values)}
    df['label'] = pd.Categorical(df['label'], categories=label_values, ordered=True)

    # Configurar el tamaño de la figura igual que en sacer.py
    plt.rcParams["figure.figsize"] = (20, 10)

    fig, axes = plt.subplot_mosaic([['scatter'], ['heatmap']], constrained_layout=True,
                                   gridspec_kw={'height_ratios': [1, 2]})

    # Scatter plot
    for label, occupation in occupation_counts['occupation'].items():
        xpos = label_dict[label] + 0.5
        axes['scatter'].vlines(x=xpos, ymin=0, ymax=occupation, color='steelblue', alpha=0.7, linewidth=2)
        axes['scatter'].scatter(x=xpos, y=occupation, s=75, color='steelblue', alpha=0.7)

    axes['scatter'].set_title(
        f'Banda: Radiodifusión {banda}, Ciudad: {ciudad}, Umbral: {umbral} dBµV/m, Periodo: {fecha_init} a {fecha_end}',
        fontsize=16)
    axes['scatter'].set_ylabel('Ocupación (%)')
    axes['scatter'].set_ylim(0, 100)
    axes['scatter'].grid(color='gray', linestyle='--', linewidth=0.5)

    # Set the y-ticks labels with adjusted font size
    yticklabels = axes['scatter'].get_yticks().tolist()
    axes['scatter'].set_yticklabels(yticklabels, fontsize=8)

    # Annotations
    for label, occupation in occupation_counts['occupation'].items():
        xpos = label_dict[label] + 0.5
        axes['scatter'].text(xpos, occupation + 1, s=str(int(occupation)),
                             horizontalalignment='center', verticalalignment='bottom', fontsize=8)

    # Heatmap
    df_max_level = df.groupby(['freq', 'tiempo'])['level'].max().reset_index()
    df_max_level['tiempo'] = pd.to_datetime(df_max_level['tiempo']).dt.date
    heatmap_data = df_max_level.pivot_table(values='level', index='tiempo', columns='freq').sort_index(ascending=False)
    sns.heatmap(heatmap_data, cmap='rainbow', vmin=0, vmax=100, cbar_kws={'label': 'Level (dBµV/m)'},
                ax=axes['heatmap'])

    mid_positions = [i + 0.5 for i in range(len(label_values))]
    axes['heatmap'].set_xticks(mid_positions)
    axes['heatmap'].set_xticklabels(label_values, rotation=270, fontsize=8)
    axes['heatmap'].set_xlim(0, len(label_values))

    # Adjust the y-tick labels font size
    yticklabels = axes['heatmap'].get_yticklabels()
    axes['heatmap'].set_yticklabels(yticklabels, fontsize=8)

    axes['scatter'].tick_params(labelbottom=False)
    axes['scatter'].set_xticks(mid_positions)
    axes['scatter'].set_xticklabels(label_values, rotation=90, fontsize=8)
    axes['scatter'].set_xlim(0, len(label_values))

    axes['heatmap'].set_xlabel('Frecuencia (Hz)')
    axes['heatmap'].set_ylabel('Tiempo')

    # Save to BytesIO
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    img_buffer.seek(0)

    return img_buffer


def create_excel_file(output, df_final5, df_final6, df_final9, contar1, contar2, contar3,
                      image1, image2, image3, df_original1, df_original2, df_original3,
                      df9, df10, df17, ciudad, fecha_inicio, fecha_fin, umbral_am, umbral_fm, umbral_tv):
    """Create Excel file with all sheets and formatting"""

    Year1 = fecha_inicio.year
    Year2 = fecha_fin.year
    Mes_inicio = fecha_inicio.strftime('%B')
    Mes_fin = fecha_fin.strftime('%B')
    is_single_month = (Year1 == Year2 and Mes_inicio == Mes_fin)

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write main sheets
        if not df_final5.empty:
            df_final5.to_excel(writer, sheet_name='Radiodifusión FM')
        if not df_final6.empty:
            df_final6.to_excel(writer, sheet_name='Televisión')
        if not df_final9.empty and ciudad in ['Quito', 'Guayaquil', 'Cuenca']:
            df_final9.to_excel(writer, sheet_name='Radiodifusión AM')

        workbook = writer.book

        # Formato para las celdas de umbral
        format_umbral_title = workbook.add_format({
            'bold': True,
            'border': 1,
            'bg_color': '#C6EFCE',
            'font_color': '#006100'
        })
        format_umbral_value = workbook.add_format({
            'border': 1,
            'bg_color': '#C6EFCE',
            'font_color': '#006100'
        })

        # --- APLICAR FORMATO, FILTROS E INMOVILIZACIÓN ---

        # --- APLICAR FILTROS E INMOVILIZACIÓN (Freeze Panes) ---

        # 1. Radiodifusión FM
        if not df_final5.empty:
            worksheet_fm = writer.sheets['Radiodifusión FM']

            # Inmovilizar paneles desde F2 (Fila 1, Columnas A-E)
            worksheet_fm.freeze_panes('F2')

            # Aplicar Auto-filtro a todas las columnas (desde A1 hasta la última fecha)
            last_col_idx = len(df_final5.columns) - 1
            worksheet_fm.autofilter(0, 0, 0, last_col_idx)

        # 2. Televisión
        if not df_final6.empty:
            worksheet_tv = writer.sheets['Televisión']

            # Inmovilizar paneles desde F2 (Fila 1, Columnas A-E)
            worksheet_tv.freeze_panes('F2')

            # Aplicar Auto-filtro
            last_col_idx_tv = len(df_final6.columns) - 1
            worksheet_tv.autofilter(0, 0, 0, last_col_idx_tv)

        # 3. Radiodifusión AM
        if not df_final9.empty and ciudad in ['Quito', 'Guayaquil', 'Cuenca']:
            worksheet_am = writer.sheets['Radiodifusión AM']

            # Inmovilizar paneles desde D2 (Fila 1, Columnas A-C)
            worksheet_am.freeze_panes('D2')

            # Aplicar Auto-filtro
            last_col_idx_am = len(df_final9.columns) - 1
            worksheet_am.autofilter(0, 0, 0, last_col_idx_am)

        # Write occupation sheets with umbral cells
        if not contar1.empty:
            # Escribir la tabla empezando en A2 (dejando A1 y B1 para umbral)
            contar1.to_excel(writer, sheet_name='OC_Radiodifusión FM', startrow=1, startcol=0)
            worksheet_oc_fm = writer.sheets['OC_Radiodifusión FM']
            worksheet_oc_fm.write('A1', 'Umbral (dBµV/m)', format_umbral_title)
            worksheet_oc_fm.write('B1', float(umbral_fm), format_umbral_value)

        if not contar2.empty:
            contar2.to_excel(writer, sheet_name='OC_Televisión', startrow=1, startcol=0)
            worksheet_oc_tv = writer.sheets['OC_Televisión']
            worksheet_oc_tv.write('A1', 'Umbral (dBµV/m)', format_umbral_title)
            worksheet_oc_tv.write('B1', float(umbral_tv), format_umbral_value)

        if not contar3.empty and ciudad in ['Quito', 'Guayaquil', 'Cuenca']:
            contar3.to_excel(writer, sheet_name='OC_Radiodifusión AM', startrow=1, startcol=0)
            worksheet_oc_am = writer.sheets['OC_Radiodifusión AM']
            worksheet_oc_am.write('A1', 'Umbral (dBµV/m)', format_umbral_title)
            worksheet_oc_am.write('B1', float(umbral_am), format_umbral_value)

        # Write measurements sheets if single month
        if is_single_month:
            if not df_original1.empty:
                df_original1.to_excel(writer, sheet_name='Mediciones FM')
            if not df_original2.empty:
                df_original2.to_excel(writer, sheet_name='Mediciones TV')
            if not df_original3.empty and ciudad in ['Quito', 'Guayaquil', 'Cuenca']:
                df_original3.to_excel(writer, sheet_name='Mediciones AM')

        # Insert images in occupation sheets
        if not contar1.empty and image1 is not None:
            worksheet_oc_fm = writer.sheets['OC_Radiodifusión FM']
            worksheet_oc_fm.insert_image('E1', 'image1.png', {'image_data': image1})

        if not contar2.empty and image2 is not None:
            worksheet_oc_tv = writer.sheets['OC_Televisión']
            worksheet_oc_tv.insert_image('E1', 'image2.png', {'image_data': image2})

        if not contar3.empty and image3 is not None and ciudad in ['Quito', 'Guayaquil', 'Cuenca']:
            worksheet_oc_am = writer.sheets['OC_Radiodifusión AM']
            worksheet_oc_am.insert_image('E1', 'image3.png', {'image_data': image3})

        # Apply conditional formatting
        apply_conditional_formatting(writer, df_final5, df_final6, df_final9, df9, df10, df17, ciudad)

    return output


def apply_conditional_formatting(writer, df_final5, df_final6, df_final9, df9, df10, df17, ciudad):
    """
    Apply conditional formatting to Excel sheets based on utils.py color scheme

    Colors from utils.py:
    - Plus (Verde): #7fc97f - Valores superiores al umbral
    - Bet (Amarillo): #FFD700 - Valores intermedios
    - Minus (Rojo): #ff9999 - Valores inferiores al umbral
    - Valor (Morado/Sin datos): #beaed4 - Sin mediciones
    - Autorizaciones Suspensión: #386cb0 - Azul oscuro
    - Autorizaciones Baja Potencia: #23b4e8 - Azul claro
    - Sobrelapamiento S + BP: #8B008B - Púrpura oscuro
    """

    workbook = writer.book

    # Define formats matching utils.py colors
    format_green = workbook.add_format({
        'bg_color': '#7fc97f',
        'font_color': '#000000',
        'border': 1
    })
    format_yellow = workbook.add_format({
        'bg_color': '#FFD700',
        'font_color': '#000000',
        'border': 1
    })
    format_red = workbook.add_format({
        'bg_color': '#ff9999',
        'font_color': '#000000',
        'border': 1
    })
    format_purple = workbook.add_format({
        'bg_color': '#beaed4',
        'font_color': '#000000',
        'border': 1
    })
    format_blue_dark = workbook.add_format({
        'bg_color': '#386cb0',
        'font_color': '#FFFFFF',
        'border': 1
    })
    format_blue_light = workbook.add_format({
        'bg_color': '#23b4e8',
        'font_color': '#000000',
        'border': 1
    })
    format_overlap = workbook.add_format({
        'bg_color': '#8B008B',
        'font_color': '#FFFFFF',
        'border': 1
    })

    # Formato para texto de leyenda
    format_legend_title = workbook.add_format({
        'bold': True,
        'font_size': 11
    })
    format_legend_text = workbook.add_format({
        'font_size': 10
    })

    # Función auxiliar para detectar autorizaciones por fecha
    def get_authorization_for_date(station_authorizations, date_str):
        """
        Retorna el tipo de autorización(es) activa(s) para una fecha específica
        Returns: 'S', 'BP', 'BOTH', o None
        """
        date = pd.to_datetime(date_str)
        active_auths = []

        for auth in station_authorizations:
            if auth['fecha_inicio'] and auth['fecha_fin']:
                inicio = pd.to_datetime(auth['fecha_inicio'])
                fin = pd.to_datetime(auth['fecha_fin'])
                if inicio <= date <= fin:
                    active_auths.append(auth['tipo'])

        if 'S' in active_auths and 'BP' in active_auths:
            return 'BOTH'
        elif 'S' in active_auths:
            return 'S'
        elif 'BP' in active_auths:
            return 'BP'
        else:
            return None

    # FM Sheet
    if not df_final5.empty and 'Radiodifusión FM' in writer.sheets:
        worksheet_fm = writer.sheets['Radiodifusión FM']

        # Obtener información detallada de autorizaciones por estación (incluyendo fechas)
        station_info = {}
        if not df9.empty:
            for _, row in df9.iterrows():
                freq = row.get('Frecuencia (Hz)')
                estacion = row.get('Estación')
                potencia = row.get('Potencia', 0)
                bw = row.get('BW Asignado', 220)
                tipo_aut = row.get('Tipo', None)
                fecha_inicio = row.get('Fecha_inicio', None)
                fecha_fin = row.get('Fecha_fin', None)

                key = f"{freq}_{estacion}"
                if key not in station_info:
                    station_info[key] = {
                        'potencia': potencia,
                        'bw': bw,
                        'authorizations': []
                    }

                # Agregar autorización si existe
                if tipo_aut in ['S', 'BP'] and fecha_inicio and fecha_fin:
                    station_info[key]['authorizations'].append({
                        'tipo': tipo_aut,
                        'fecha_inicio': fecha_inicio,
                        'fecha_fin': fecha_fin
                    })

        # Iterar sobre las filas del dataframe
        for row_idx, (idx, row) in enumerate(df_final5.iterrows()):
            param = idx
            row_num = row_idx + 1

            if param == 'Level (dBµV/m)':
                freq = row.get('Frecuencia (Hz)', None)
                estacion = row.get('Estación', None)
                potencia = row.get('Potencia', 0)
                bw = row.get('BW Asignado', 220)

                # Determinar umbrales según potencia y BW
                if potencia == 0 and bw in [220, 200]:
                    maximo = 54
                    minimo = 30
                elif potencia == 0 and bw == 180:
                    maximo = 48
                    minimo = 30
                elif potencia == 1:
                    maximo = 43
                    minimo = 30
                else:
                    maximo = 54
                    minimo = 30

                key = f"{freq}_{estacion}"
                station_auths = station_info.get(key, {}).get('authorizations', [])

                for col_idx, col_name in enumerate(df_final5.columns):
                    if col_name not in ['Frecuencia (Hz)', 'Estación', 'Potencia', 'BW Asignado', 'Promedio',
                                        'Observaciones']:
                        cell_value = row[col_name]
                        col_num = col_idx + 1

                        # Determinar tipo de autorización para esta fecha
                        auth_type = get_authorization_for_date(station_auths, col_name)

                        # PRIORIDAD: Autorizaciones > Valores > Sin datos
                        if auth_type == 'BOTH':
                            worksheet_fm.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_overlap)
                        elif auth_type == 'S':
                            worksheet_fm.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_blue_dark)
                        elif auth_type == 'BP':
                            worksheet_fm.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_blue_light)
                        elif cell_value == '-' or pd.isna(cell_value):
                            worksheet_fm.write(row_num, col_num, '', format_purple)
                        elif cell_value == 0:
                            worksheet_fm.write(row_num, col_num, '', format_purple)
                        elif isinstance(cell_value, (int, float)):
                            if cell_value >= maximo:
                                worksheet_fm.write(row_num, col_num, cell_value, format_green)
                            elif minimo <= cell_value < maximo:
                                worksheet_fm.write(row_num, col_num, cell_value, format_yellow)
                            elif cell_value < minimo:
                                worksheet_fm.write(row_num, col_num, cell_value, format_red)

            elif param == 'Bandwidth (Hz)':
                freq = row.get('Frecuencia (Hz)', None)
                estacion = row.get('Estación', None)
                bw_asignado = row.get('BW Asignado', 220)

                key = f"{freq}_{estacion}"
                station_auths = station_info.get(key, {}).get('authorizations', [])

                for col_idx, col_name in enumerate(df_final5.columns):
                    if col_name not in ['Frecuencia (Hz)', 'Estación', 'Potencia', 'BW Asignado', 'Promedio',
                                        'Observaciones']:
                        cell_value = row[col_name]
                        col_num = col_idx + 1

                        auth_type = get_authorization_for_date(station_auths, col_name)

                        # PRIORIDAD: Autorizaciones > Valores > Sin datos
                        if auth_type == 'BOTH':
                            worksheet_fm.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_overlap)
                        elif auth_type == 'S':
                            worksheet_fm.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_blue_dark)
                        elif auth_type == 'BP':
                            worksheet_fm.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_blue_light)
                        elif cell_value == '-' or pd.isna(cell_value):
                            worksheet_fm.write(row_num, col_num, '', format_purple)
                        elif cell_value == 0:
                            worksheet_fm.write(row_num, col_num, '', format_purple)
                        elif isinstance(cell_value, (int, float)):
                            if abs(cell_value - bw_asignado) <= (bw_asignado * 0.05):
                                worksheet_fm.write(row_num, col_num, cell_value, format_green)
                            else:
                                worksheet_fm.write(row_num, col_num, cell_value, format_red)

        # Agregar leyenda al final de la hoja FM
        legend_row = len(df_final5) + 3
        worksheet_fm.write(legend_row, 0, 'LEYENDA:', format_legend_title)
        worksheet_fm.write(legend_row, 1,
                           'Los valores de intensidad de campo eléctrico en dBuV/m corresponden a los máximos diarios.',
                           format_legend_text)
        worksheet_fm.write(legend_row + 1, 0, 'A', format_green)
        worksheet_fm.write(legend_row + 1, 1,
                           'Color VERDE: los valores de campo eléctrico diario superan el valor del borde de área de cobertura (Estereofónico: >=54 dBuV/m, Monofónico: >=48 dBuV/m o Baja Potencia: >=43 dBuV/m).',
                           format_legend_text)
        worksheet_fm.write(legend_row + 2, 0, 'A', format_yellow)
        worksheet_fm.write(legend_row + 2, 1,
                           'Color AMARILLO: los valores de campo eléctrico diario se encuentran entre el valor del borde de área de protección y el valor del borde de área de cobertura (Estereofónico: entre 30 y 54 dBuV/m, Monofónico: entre 30 y 48 dBuV/m o Baja Potencia: entre 30 y 43 dBuV/m).',
                           format_legend_text)
        worksheet_fm.write(legend_row + 3, 0, 'A', format_red)
        worksheet_fm.write(legend_row + 3, 1,
                           'Color ROJO: los valores de campo eléctrico diario son inferiores al valor del borde de área de protección (<30 dBuV/m).',
                           format_legend_text)
        worksheet_fm.write(legend_row + 4, 0, 'A', format_purple)
        worksheet_fm.write(legend_row + 4, 1, 'Color MORADO: No se dispone de mediciones del sistema SACER.',
                           format_legend_text)
        worksheet_fm.write(legend_row + 5, 0, 'A', format_blue_dark)
        worksheet_fm.write(legend_row + 5, 1,
                           'Color AZUL OSCURO: Dispone de autorización para suspensión de emisiones.',
                           format_legend_text)
        worksheet_fm.write(legend_row + 6, 0, 'A', format_blue_light)
        worksheet_fm.write(legend_row + 6, 1,
                           'Color AZUL CLARO: Dispone de autorización para operación con baja potencia.',
                           format_legend_text)
        worksheet_fm.write(legend_row + 7, 0, 'A', format_overlap)
        worksheet_fm.write(legend_row + 7, 1,
                           'Color PÚRPURA OSCURO: Sobrelapamiento de autorizaciones (Suspensión + Baja Potencia).',
                           format_legend_text)
        worksheet_fm.write(legend_row + 8, 1,
                           'Para todos los casos el valor de ancho de banda corresponde a 220 kHz, excepto aquellos en los que se especifica que el valor es de 180 kHz o 200 kHz.',
                           format_legend_text)
        worksheet_fm.write(legend_row + 9, 0,
                           'CONSIDERACIONES GENERALES:',
                           format_legend_text)
        worksheet_fm.write(legend_row + 10, 1,
                           'Nota 1.- En apego a la Resolución ST-2014-0257 referida en los LINEAMIENTOS PARA EL CONTROL Y MONITOREO DE PARÁMETROS TÉCNICOS RTV CON EL SACER (CCDE-01, PACT-2025), se considera que un control individual a una estación RTV es pertinente cuando esta ha suspendido emisiones por más de 8 días.',
                           format_legend_text)
        worksheet_fm.write(legend_row + 11, 1,
                           'Nota 2.- Las mediciones de Ancho de Banda obtenidas con el SACER son únicamente referenciales. La Unión Internacional de Telecomunicaciones – UIT, a través del Manual COMPROBACIÓN TÉCNICA DEL ESPECTRO RADIOELÉCTRICO (numeral 4.5.3) define las condiciones que deben tenerse en cuenta al medir la anchura de banda, señalando: “(…) a causa de las imprecisiones debidas a las razones expuestas, estas mediciones a distancia sólo son útiles a título indicativo. Cuando se requiera una mayor precisión, es conveniente que las mediciones se realicen en las inmediaciones del transmisor.”.',
                           format_legend_text)
        worksheet_fm.write(legend_row + 12, 1,
                           'Nota 3.- De acuerdo a los numerales 4.1.a.5 y 4.1.a.6 de los LINEAMIENTOS PARA EL CONTROL Y MONITOREO DE PARÁMETROS TÉCNICOS RTV CON EL SACER (CCDE-01, PACT-2025), para cada observación detectada que contenga datos inconsistentes que no se encuentren dentro del rango autorizado, dependiendo del caso y de los resultados, corresponde programar una verificación en sitio. Dicha programación puede estar sujeta a los resultados de las mediciones del siguiente mes.',
                           format_legend_text)

    # TV Sheet
    if not df_final6.empty and 'Televisión' in writer.sheets:
        worksheet_tv = writer.sheets['Televisión']

        station_info_tv = {}
        if not df10.empty:
            for _, row in df10.iterrows():
                freq = row.get('Frecuencia (Hz)')
                estacion = row.get('Estación')
                tipo_aut = row.get('Tipo', None)
                fecha_inicio = row.get('Fecha_inicio', None)
                fecha_fin = row.get('Fecha_fin', None)

                key = f"{freq}_{estacion}"
                if key not in station_info_tv:
                    station_info_tv[key] = {'authorizations': []}

                if tipo_aut in ['S', 'BP'] and fecha_inicio and fecha_fin:
                    station_info_tv[key]['authorizations'].append({
                        'tipo': tipo_aut,
                        'fecha_inicio': fecha_inicio,
                        'fecha_fin': fecha_fin
                    })

        for row_idx, (idx, row) in enumerate(df_final6.iterrows()):
            param = idx
            row_num = row_idx + 1

            if param == 'Level (dBµV/m)':
                freq = row.get('Frecuencia (Hz)', None)
                estacion = row.get('Estación', None)
                andig = row.get('Analógico/Digital', None)

                if freq and isinstance(freq, (int, float)):
                    if 54e6 <= freq <= 88e6 and andig != 'D':
                        umbral_rojo = 47
                        umbral_amarillo = 68
                    elif 174e6 <= freq <= 216e6 and andig != 'D':
                        umbral_rojo = 56
                        umbral_amarillo = 71
                    elif 470e6 <= freq <= 880e6 and andig != 'D':
                        umbral_rojo = 64
                        umbral_amarillo = 74
                    elif andig == 'D':
                        umbral_rojo = 30
                        umbral_amarillo = 51
                    else:
                        umbral_rojo = 30
                        umbral_amarillo = 51
                else:
                    umbral_rojo = 47
                    umbral_amarillo = 68

                key = f"{freq}_{estacion}"
                station_auths = station_info_tv.get(key, {}).get('authorizations', [])

                for col_idx, col_name in enumerate(df_final6.columns):
                    if col_name not in ['Frecuencia (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital',
                                        'Promedio', 'Observaciones']:
                        cell_value = row[col_name]
                        col_num = col_idx + 1

                        auth_type = get_authorization_for_date(station_auths, col_name)

                        # PRIORIDAD: Autorizaciones > Valores > Sin datos
                        if auth_type == 'BOTH':
                            worksheet_tv.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_overlap)
                        elif auth_type == 'S':
                            worksheet_tv.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_blue_dark)
                        elif auth_type == 'BP':
                            worksheet_tv.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_blue_light)
                        elif cell_value == '-' or pd.isna(cell_value):
                            worksheet_tv.write(row_num, col_num, '', format_purple)
                        elif cell_value == 0:
                            worksheet_tv.write(row_num, col_num, '', format_purple)
                        elif isinstance(cell_value, (int, float)):
                            if cell_value >= umbral_amarillo:
                                worksheet_tv.write(row_num, col_num, cell_value, format_green)
                            elif umbral_rojo <= cell_value < umbral_amarillo:
                                worksheet_tv.write(row_num, col_num, cell_value, format_yellow)
                            elif cell_value < umbral_rojo:
                                worksheet_tv.write(row_num, col_num, cell_value, format_red)

        # Agregar leyenda al final de la hoja TV
        legend_row = len(df_final6) + 3
        worksheet_tv.write(legend_row, 0, 'LEYENDA:', format_legend_title)
        worksheet_tv.write(legend_row, 1,
                           'Los valores de intensidad de campo eléctrico en dBuV/m corresponden a los máximos diarios.',
                           format_legend_text)
        worksheet_tv.write(legend_row + 1, 0, 'A', format_green)
        worksheet_tv.write(legend_row + 1, 1,
                           'Color VERDE: Los valores de campo eléctrico diario superan el límite del área de cobertura primaria. (Canales 2-6: ≥68 dBµV/m, Canales 7-13: ≥71 dBµV/m, Canales 14-51: ≥74 dBµV/m, Digitales: ≥51 dBµV/m).',
                           format_legend_text)
        worksheet_tv.write(legend_row + 2, 0, 'A', format_yellow)
        worksheet_tv.write(legend_row + 2, 1,
                           'Color AMARILLO: Los valores de campo eléctrico diario superan el límite del área de cobertura secundario pero son inferiores al límite del área de cobertura principal.(Canales 2-6: 47-68 dBµV/m, Canales 7-13: 56-71 dBµV/m, Canales 14-51: 64-74 dBµV/m).',
                           format_legend_text)
        worksheet_tv.write(legend_row + 3, 0, 'A', format_red)
        worksheet_tv.write(legend_row + 3, 1,
                           'Color ROJO: Los valores de campo eléctrico diario son inferiores al límite de área de cobertura secundario. (Canales 2-6: <47 dBµV/m, Canales 7-13: <56 dBµV/m, Canales 14-51: <64 dBµV/m, Digitales: <30 dBµV/m).',
                           format_legend_text)
        worksheet_tv.write(legend_row + 4, 0, 'A', format_purple)
        worksheet_tv.write(legend_row + 4, 1, 'Color MORADO: No se dispone de mediciones del sistema SACER.',
                           format_legend_text)
        worksheet_tv.write(legend_row + 5, 0, 'A', format_blue_dark)
        worksheet_tv.write(legend_row + 5, 1,
                           'Color AZUL OSCURO: Dispone de autorización para suspensión de emisiones.',
                           format_legend_text)
        worksheet_tv.write(legend_row + 6, 0, 'A', format_blue_light)
        worksheet_tv.write(legend_row + 6, 1,
                           'Color AZUL CLARO: Dispone de autorización para operación con baja potencia.',
                           format_legend_text)
        worksheet_tv.write(legend_row + 7, 0, 'A', format_overlap)
        worksheet_tv.write(legend_row + 7, 1,
                           'Color PÚRPURA OSCURO: Sobrelapamiento de autorizaciones (Suspensión + Baja Potencia).',
                           format_legend_text)
        worksheet_tv.write(legend_row + 8, 0,
                           'CONSIDERACIONES GENERALES:',
                           format_legend_text)
        worksheet_tv.write(legend_row + 9, 1,
                           'Nota 1.- En apego a la Resolución ST-2014-0257 referida en los LINEAMIENTOS PARA EL CONTROL Y MONITOREO DE PARÁMETROS TÉCNICOS RTV CON EL SACER (CCDE-01, PACT-2025), se considera que un control individual a una estación RTV es pertinente cuando esta ha suspendido emisiones por más de 8 días.',
                           format_legend_text)
        worksheet_tv.write(legend_row + 10, 1,
                           'Nota 2.- De acuerdo a los numerales 4.1.a.5 y 4.1.a.6 de los LINEAMIENTOS PARA EL CONTROL Y MONITOREO DE PARÁMETROS TÉCNICOS RTV CON EL SACER (CCDE-01, PACT-2025), para cada observación detectada que contenga datos inconsistentes que no se encuentren dentro del rango autorizado, dependiendo del caso y de los resultados, corresponde programar una verificación en sitio. Dicha programación puede estar sujeta a los resultados de las mediciones del siguiente mes.',
                           format_legend_text)

    # AM Sheet
    if not df_final9.empty and ciudad in ['Quito', 'Guayaquil', 'Cuenca'] and 'Radiodifusión AM' in writer.sheets:
        worksheet_am = writer.sheets['Radiodifusión AM']

        station_info_am = {}
        if not df17.empty:
            for _, row in df17.iterrows():
                freq = row.get('Frecuencia (Hz)')
                estacion = row.get('Estación')
                tipo_aut = row.get('Tipo', None)
                fecha_inicio = row.get('Fecha_inicio', None)
                fecha_fin = row.get('Fecha_fin', None)

                key = f"{freq}_{estacion}"
                if key not in station_info_am:
                    station_info_am[key] = {'authorizations': []}

                if tipo_aut in ['S', 'BP'] and fecha_inicio and fecha_fin:
                    station_info_am[key]['authorizations'].append({
                        'tipo': tipo_aut,
                        'fecha_inicio': fecha_inicio,
                        'fecha_fin': fecha_fin
                    })

        for row_idx, (idx, row) in enumerate(df_final9.iterrows()):
            param = idx
            row_num = row_idx + 1

            if param == 'Level (dBµV/m)':
                freq = row.get('Frecuencia (Hz)', None)
                estacion = row.get('Estación', None)

                umbral_rojo = 40
                umbral_amarillo = 62

                key = f"{freq}_{estacion}"
                station_auths = station_info_am.get(key, {}).get('authorizations', [])

                for col_idx, col_name in enumerate(df_final9.columns):
                    if col_name not in ['Frecuencia (Hz)', 'Estación', 'Promedio', 'Observaciones']:
                        cell_value = row[col_name]
                        col_num = col_idx + 1

                        auth_type = get_authorization_for_date(station_auths, col_name)

                        # PRIORIDAD: Autorizaciones > Valores > Sin datos
                        if auth_type == 'BOTH':
                            worksheet_am.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_overlap)
                        elif auth_type == 'S':
                            worksheet_am.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_blue_dark)
                        elif auth_type == 'BP':
                            worksheet_am.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_blue_light)
                        elif cell_value == '-' or pd.isna(cell_value):
                            worksheet_am.write(row_num, col_num, '', format_purple)
                        elif cell_value == 0:
                            worksheet_am.write(row_num, col_num, '', format_purple)
                        elif isinstance(cell_value, (int, float)):
                            if cell_value >= umbral_amarillo:
                                worksheet_am.write(row_num, col_num, cell_value, format_green)
                            elif umbral_rojo <= cell_value < umbral_amarillo:
                                worksheet_am.write(row_num, col_num, cell_value, format_yellow)
                            elif cell_value < umbral_rojo:
                                worksheet_am.write(row_num, col_num, cell_value, format_red)

            elif param == 'Bandwidth (Hz)':
                freq = row.get('Frecuencia (Hz)', None)
                estacion = row.get('Estación', None)
                key = f"{freq}_{estacion}"
                station_auths = station_info_am.get(key, {}).get('authorizations', [])

                for col_idx, col_name in enumerate(df_final9.columns):
                    if col_name not in ['Frecuencia (Hz)', 'Estación', 'Promedio', 'Observaciones']:
                        cell_value = row[col_name]
                        col_num = col_idx + 1

                        auth_type = get_authorization_for_date(station_auths, col_name)

                        # PRIORIDAD: Autorizaciones > Valores > Sin datos
                        if auth_type == 'BOTH':
                            worksheet_am.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_overlap)
                        elif auth_type == 'S':
                            worksheet_am.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_blue_dark)
                        elif auth_type == 'BP':
                            worksheet_am.write(row_num, col_num,
                                               '' if (cell_value == '-' or cell_value == 0) else cell_value,
                                               format_blue_light)
                        elif cell_value == '-' or pd.isna(cell_value):
                            worksheet_am.write(row_num, col_num, '', format_purple)
                        elif cell_value == 0:
                            worksheet_am.write(row_num, col_num, '', format_purple)
                        elif isinstance(cell_value, (int, float)):
                            if abs(cell_value - 15000) <= 750:
                                worksheet_am.write(row_num, col_num, cell_value, format_green)
                            else:
                                worksheet_am.write(row_num, col_num, cell_value, format_red)

        # Agregar leyenda al final de la hoja AM
        legend_row = len(df_final9) + 3
        worksheet_am.write(legend_row, 0, 'LEYENDA:', format_legend_title)
        worksheet_am.write(legend_row, 1,
                           'Los valores de intensidad de campo eléctrico en dBuV/m corresponden a los máximos diarios.',
                           format_legend_text)
        worksheet_am.write(legend_row + 1, 0, 'A', format_green)
        worksheet_am.write(legend_row + 1, 1,
                           'Color VERDE: Los valores de campo eléctrico diario superan el valor del borde de área de cobertura (≥62 dBµV/m).',
                           format_legend_text)
        worksheet_am.write(legend_row + 2, 0, 'A', format_yellow)
        worksheet_am.write(legend_row + 2, 1,
                           'Color AMARILLO: Los valores de campo eléctrico diario se encuentran entre el valor del borde de área de protección y el valor del borde de área de cobertura (entre 40 y 62 dBµV/m).',
                           format_legend_text)
        worksheet_am.write(legend_row + 3, 0, 'A', format_red)
        worksheet_am.write(legend_row + 3, 1,
                           'Color ROJO: Los valores de campo eléctrico diario son inferiores al valor del borde de área de protección (<40 dBµV/m).',
                           format_legend_text)
        worksheet_am.write(legend_row + 4, 0, 'A', format_purple)
        worksheet_am.write(legend_row + 4, 1, 'Color MORADO: No se dispone de mediciones del sistema SACER.',
                           format_legend_text)
        worksheet_am.write(legend_row + 5, 0, 'A', format_blue_dark)
        worksheet_am.write(legend_row + 5, 1,
                           'Color AZUL OSCURO: Dispone de autorización para suspensión de emisiones.',
                           format_legend_text)
        worksheet_am.write(legend_row + 6, 0, 'A', format_blue_light)
        worksheet_am.write(legend_row + 6, 1,
                           'Color AZUL CLARO: Dispone de autorización para operación con baja potencia.',
                           format_legend_text)
        worksheet_am.write(legend_row + 7, 0, 'A', format_overlap)
        worksheet_am.write(legend_row + 7, 1,
                           'Color PÚRPURA OSCURO: Sobrelapamiento de autorizaciones (Suspensión + Baja Potencia).',
                           format_legend_text)
        worksheet_am.write(legend_row + 8, 1, 'El valor de ancho de banda corresponde a 15 kHz.', format_legend_text)
        worksheet_am.write(legend_row + 9, 0,
                           'CONSIDERACIONES GENERALES:',
                           format_legend_text)
        worksheet_am.write(legend_row + 10, 1,
                           'Nota 1.- En apego a la Resolución ST-2014-0257 referida en los LINEAMIENTOS PARA EL CONTROL Y MONITOREO DE PARÁMETROS TÉCNICOS RTV CON EL SACER (CCDE-01, PACT-2025), se considera que un control individual a una estación RTV es pertinente cuando esta ha suspendido emisiones por más de 8 días.',
                           format_legend_text)
        worksheet_am.write(legend_row + 11, 1,
                           'Nota 2.- Las mediciones de Ancho de Banda obtenidas con el SACER son únicamente referenciales. La Unión Internacional de Telecomunicaciones – UIT, a través del Manual COMPROBACIÓN TÉCNICA DEL ESPECTRO RADIOELÉCTRICO (numeral 4.5.3) define las condiciones que deben tenerse en cuenta al medir la anchura de banda, señalando: “(…) a causa de las imprecisiones debidas a las razones expuestas, estas mediciones a distancia sólo son útiles a título indicativo. Cuando se requiera una mayor precisión, es conveniente que las mediciones se realicen en las inmediaciones del transmisor.”.',
                           format_legend_text)
        worksheet_am.write(legend_row + 12, 1,
                           'Nota 3.- De acuerdo a los numerales 4.1.a.5 y 4.1.a.6 de los LINEAMIENTOS PARA EL CONTROL Y MONITOREO DE PARÁMETROS TÉCNICOS RTV CON EL SACER (CCDE-01, PACT-2025), para cada observación detectada que contenga datos inconsistentes que no se encuentren dentro del rango autorizado, dependiendo del caso y de los resultados, corresponde programar una verificación en sitio. Dicha programación puede estar sujeta a los resultados de las mediciones del siguiente mes.',
                           format_legend_text)
