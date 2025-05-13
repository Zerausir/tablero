from datetime import date, timedelta
from functools import lru_cache
import plotly.graph_objects as go
import pandas as pd


@lru_cache(maxsize=1)
def calculate_disabled_days_for_year(year):
    """
    Calcula el último día de cada mes en el año especificado y devuelve los días que no son
    el último día de cada mes (días deshabilitados).

    Args:
        year (int): El año para el cual calcular los días deshabilitados.

    Returns:
        list: Lista de objetos date que representan los días deshabilitados.
    """

    # Calcula el último día de cada mes en el año especificado
    def last_day_of_month(year, month):
        if month == 12:
            return date(year, month, 31)
        return date(year, month + 1, 1) - timedelta(days=1)

    # Lista de los últimos días de cada mes en el año especificado
    last_days = [last_day_of_month(year, month) for month in range(1, 13)]

    # Genera una lista de todos los días del año especificado
    all_days = [date(year, month, day) for month in range(1, 13)
                for day in range(1, (last_day_of_month(year, month).day + 1))]

    # Filtra los días que no son el último día de cada mes
    disabled_days = [day for day in all_days if day not in last_days]

    return disabled_days


def update_table(dataframe, selected_indicador, table_id):
    """
    Actualiza la tabla de datos aplicando filtros de indicadores.

    Args:
        dataframe (pd.DataFrame): El DataFrame a filtrar.
        selected_indicador (list): Lista de indicadores seleccionados para filtrar.
        table_id (str): ID de la tabla (no utilizado en esta función pero podría ser útil para futuras ampliaciones).

    Returns:
        list: Lista de diccionarios que representa las filas filtradas del DataFrame.
    """
    filtered_df = dataframe.copy()

    if selected_indicador:
        filtered_df = filtered_df[filtered_df['INDICADOR_CORTO'].isin(selected_indicador)]

    return filtered_df.to_dict('records')


def wrap_text(text, width=110):
    """
    Inserta saltos de línea en el texto para que se ajuste al ancho dado.

    Args:
        text: El texto a envolver, puede ser str, float, int, etc.
        width (int, optional): El ancho máximo por línea. Por defecto es 110.

    Returns:
        str: El texto con saltos de línea HTML (<br>) insertados.
    """
    # Convertir a cadena si no es ya una cadena
    if not isinstance(text, str):
        text = str(text)

    if text is None or text == "":
        return ""

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        # Verifica si añadiendo la palabra actual se supera el ancho máximo
        if sum(len(word) for word in current_line) + len(word) + len(current_line) > width:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)

    if current_line:
        lines.append(" ".join(current_line))

    return "<br>".join(lines)  # Usa <br> para saltos de línea en anotaciones de Plotly


def create_pie_charts_for_indicators(df, selected_date):
    """
    Crea gráficos de torta para cada indicador en el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con datos de indicadores.
        selected_date (str): Fecha seleccionada para el corte.

    Returns:
        dict: Diccionario con pares de gráficos de torta para cada indicador (global y corte).
    """
    pie_charts = {}
    for indicador in df['INDICADOR_CORTO'].unique():
        data_indicador = df[df['INDICADOR_CORTO'] == indicador]
        tipo = data_indicador['TIPO'].iloc[0]  # Asume que todos los registros para un indicador comparten el mismo tipo
        indicador_full = data_indicador['INDICADOR'].iloc[0]
        if pd.isna(indicador_full):
            # Si el valor es NaN, asignar una cadena vacía o un texto por defecto
            indicador_full = f"Indicador: {indicador}"
        else:
            indicador_full = str(indicador_full)

        indicador_full_wrapped = wrap_text(indicador_full)  # Envuelve el texto

        # Calcular planificada y cumplir (denominadores)
        if indicador == 'CCDR-01':
            # Para CCDR-01, forzar que el valor "a cumplir al corte" sea idéntico al global
            planificada = data_indicador['PLANIFICADA_META'].sum()
            cumplir = planificada  # Hacer que ambos valores sean iguales
        elif tipo == 'CONTINUO':
            planificada = data_indicador['PLANIFICADA_META'].sum()
            cumplir = data_indicador['CUMPLIR_META'].sum()
        else:  # 'DISCRETO'
            planificada = data_indicador['PLANIFICADA_META'].sum()
            cumplir = data_indicador['CUMPLIR_META'].sum()

        # Modificación: Añadir caso especial para CCDE-10
        if 'CCDE-10' in data_indicador['INDICADOR_CORTO'].values:
            cantidad_verificables = data_indicador[
                'Nro_INSPECCIONES'].sum() if 'Nro_INSPECCIONES' in data_indicador.columns else 0
        elif tipo == 'CONTINUO':
            # Manejo común para todos los CONTINUO, excepto indicadores especiales
            if 'CCDS-01' in data_indicador['INDICADOR_CORTO'].values:
                cantidad_verificables = data_indicador[
                    'Nro_INSPECCIONES'].sum() if 'Nro_INSPECCIONES' in data_indicador.columns else 0
            elif 'CCDS-13' in data_indicador['INDICADOR_CORTO'].values:
                cantidad_verificables = data_indicador[
                    'Nro_INSPECCIONES'].sum() if 'Nro_INSPECCIONES' in data_indicador.columns else 0
            else:
                cantidad_verificables = data_indicador[
                    'CANTIDAD_VERIFICABLES'].sum() if 'CANTIDAD_VERIFICABLES' in data_indicador.columns else 0
        else:  # 'DISCRETO' # Manejo común para todos los DISCRETO, excepto CCDR-06
            if 'CCDR-06' in data_indicador['INDICADOR_CORTO'].values:
                cantidad_verificables = data_indicador[
                    'Nro_INSPECCIONES'].sum() if 'Nro_INSPECCIONES' in data_indicador.columns else 0
            else:
                cantidad_verificables = data_indicador[
                    'CANTIDAD_VERIFICABLES'].sum() if 'CANTIDAD_VERIFICABLES' in data_indicador.columns else 0

        # Modificación para el cálculo del avance global
        if planificada == 0:
            if cantidad_verificables > 0:
                avance_global = 100  # Si hay verificables realizados pero nada planificado, mostrar 100% de avance
            else:
                avance_global = 100  # Si no hay nada planificado y nada realizado, considerar 100% completado
        else:
            avance_global = min(cantidad_verificables / planificada * 100 if planificada > 0 else 0, 100)
        restante_global = max(100 - avance_global, 0)

        # Modificación para el cálculo del avance al corte
        if cumplir == 0:
            if cantidad_verificables > 0:
                avance_corte = 100  # Si hay verificables realizados pero nada que cumplir, mostrar 100% de avance (azul)
            else:
                avance_corte = 100  # Si no hay nada que cumplir y nada realizado, también 100% completado
        else:
            avance_corte = min(cantidad_verificables / cumplir * 100 if cumplir > 0 else 0, 100)
        restante_corte = max(100 - avance_corte, 0)

        # Extraer año de la fecha seleccionada para el título
        year = int(selected_date.split('-')[0])

        # Modificación: usar colores diferentes para indicadores sin meta al corte
        if cumplir == 0:
            # Colores para indicadores sin actividades a cumplir (gris azulado y gris claro)
            colors = ['#78A2CC', '#D3D3D3']
            hole_size = 0.4  # Agujero más grande para diferenciar visualmente
            # Añadir nota explicativa
            info_text = "Este indicador no tiene actividades planificadas en el período de corte."
        else:
            # Colores normales (azul y rojo)
            colors = ['#007BFF', '#D62828']
            hole_size = 0.3
            info_text = None

        pie_global = go.Figure(data=[
            go.Pie(labels=["Avance", "Restante"], values=[avance_global, restante_global],
                   marker=dict(colors=colors), hole=hole_size)])

        # Actualizamos el título para mostrar N/A cuando no hay meta
        global_title = f"Global: {indicador} ({cantidad_verificables} realizados / {planificada} planificados)"

        annotations = [dict(text=indicador_full_wrapped, x=0.5, y=-0.15, xref="paper", yref="paper", showarrow=False,
                            font=dict(size=10, color="black"), bordercolor='black', borderpad=4, bgcolor='white',
                            borderwidth=1)]

        pie_global.update_layout(
            title_text=global_title,
            title_x=0.5,
            legend=dict(traceorder='normal'),
            annotations=annotations
        )

        pie_corte = go.Figure(data=[
            go.Pie(labels=["Avance", "Restante"], values=[avance_corte, restante_corte],
                   marker=dict(colors=colors), hole=hole_size)])

        # Para el gráfico de corte, añadimos "N/A" cuando el cumplir es 0
        if cumplir == 0:
            corte_title = f"Al {selected_date}: {indicador} ({cantidad_verificables} realizados / N/A)"
        else:
            corte_title = f"Al {selected_date}: {indicador} ({cantidad_verificables} realizados / {cumplir} a cumplir al corte)"

        corte_annotations = [
            dict(text=indicador_full_wrapped, x=0.5, y=-0.15, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=10, color="black"), bordercolor='black', borderpad=4, bgcolor='white',
                 borderwidth=1)]

        # Añadir nota informativa si es necesario
        if info_text:
            corte_annotations.append(dict(text=info_text, x=0.5, y=-0.20, xref="paper", yref="paper", showarrow=False,
                                          font=dict(size=9, color="#666666", style="italic"), align="center"))

        pie_corte.update_layout(
            title_text=corte_title,
            title_x=0.5,
            legend=dict(traceorder='normal'),
            annotations=corte_annotations
        )

        pie_charts[indicador] = (pie_global, pie_corte)

    return pie_charts


def create_summary_pie_charts(df, selected_date):
    """
    Crea gráficos de torta de resumen para todos los indicadores.

    Args:
        df (pd.DataFrame): DataFrame con datos de indicadores.
        selected_date (str): Fecha seleccionada para el corte.

    Returns:
        dict: Diccionario con gráficos de torta de resumen (global y corte).
    """
    pie_charts = {}

    # Extraer año de la fecha seleccionada para el título
    year = int(selected_date.split('-')[0])

    # Filtrar el DataFrame para el cálculo del porcentaje de corte (solo indicadores con meta)
    df_filtered = df[df['CUMPLIR_META'] != 0]
    total_indicadores = len(df)
    total_indicadores_con_meta = len(df_filtered)

    if not df_filtered.empty:
        # Para Global Planificado
        df_filtered = df_filtered.copy()

        # Modificación: Para CCDR-01, forzar que CUMPLIR_META sea igual a PLANIFICADA_META
        mask_ccdr01 = df_filtered['INDICADOR_CORTO'] == 'CCDR-01'
        if any(mask_ccdr01):
            df_filtered.loc[mask_ccdr01, 'CUMPLIR_META'] = df_filtered.loc[mask_ccdr01, 'PLANIFICADA_META']

        # Modificación: Añadir caso especial para CCDE-10 en el cálculo de Porcentaje_Global
        df_filtered.loc[:, 'Porcentaje_Global'] = df_filtered.apply(
            lambda row: min(
                (row['Nro_INSPECCIONES'] if row['INDICADOR_CORTO'] in ['CCDE-10', 'CCDS-01', 'CCDS-13', 'CCDR-06']
                 else row['CANTIDAD_VERIFICABLES']) / row['PLANIFICADA_META'] if row['PLANIFICADA_META'] > 0 else 0, 1),
            axis=1)

        porcentaje_global_total = df_filtered['Porcentaje_Global'].sum() / len(df_filtered) * 100
        restante_global = max(100 - porcentaje_global_total, 0)

        # Añadir texto informativo sobre el total de indicadores
        global_info = f"Total: {total_indicadores} indicadores"

        pie_chart_global = go.Figure(data=[
            go.Pie(labels=["Avance", "Restante"], values=[porcentaje_global_total, restante_global],
                   marker=dict(colors=['#007BFF', '#D62828']), hole=0.3)])
        pie_chart_global.update_layout(
            title_text=f"Global {year}: 3.1 Porcentaje de Cumplimiento PACT",
            title_x=0.5,
            legend=dict(traceorder='normal'),
            annotations=[
                dict(text=global_info, x=0.5, y=-0.15, xref="paper", yref="paper", showarrow=False,
                     font=dict(size=10, color="#666666"), align="center")
            ]
        )
        pie_charts['Global Planificado'] = pie_chart_global

        # Modificación: Añadir caso especial para CCDE-10 en el cálculo de Porcentaje_Corte
        df_filtered.loc[:, 'Porcentaje_Corte'] = df_filtered.apply(
            lambda row: min(
                (row['Nro_INSPECCIONES'] if row['INDICADOR_CORTO'] in ['CCDE-10', 'CCDS-01', 'CCDS-13', 'CCDR-06']
                 else row['CANTIDAD_VERIFICABLES']) / row['CUMPLIR_META'] if row['CUMPLIR_META'] > 0 else 0, 1),
            axis=1)

        porcentaje_corte_total = df_filtered['Porcentaje_Corte'].sum() / len(df_filtered) * 100
        restante_corte = max(100 - porcentaje_corte_total, 0)

        # Añadir texto informativo sobre los indicadores incluidos/excluidos
        if total_indicadores_con_meta < total_indicadores:
            corte_info = f"Incluidos: {total_indicadores_con_meta} de {total_indicadores} indicadores (se excluyen {total_indicadores - total_indicadores_con_meta} sin actividades planificadas al corte)"
        else:
            corte_info = f"Incluidos: {total_indicadores} indicadores"

        pie_chart_corte = go.Figure(data=[
            go.Pie(labels=["Avance", "Restante"], values=[porcentaje_corte_total, restante_corte],
                   marker=dict(colors=['#007BFF', '#D62828']), hole=0.3)])
        pie_chart_corte.update_layout(
            title_text=f"Al {selected_date}: 3.1 Porcentaje de Cumplimiento PACT {year}",
            title_x=0.5,
            legend=dict(traceorder='normal'),
            annotations=[
                dict(text=corte_info, x=0.5, y=-0.15, xref="paper", yref="paper", showarrow=False,
                     font=dict(size=10, color="#666666"), align="center")
            ]
        )
        pie_charts['Fecha de Corte'] = pie_chart_corte

    return pie_charts
