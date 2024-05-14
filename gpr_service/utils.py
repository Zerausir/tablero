from datetime import date, timedelta
from functools import lru_cache
import plotly.graph_objects as go


@lru_cache(maxsize=1)
def calculate_disabled_days_for_year(year):
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
    filtered_df = dataframe.copy()

    if selected_indicador:
        filtered_df = filtered_df[filtered_df['INDICADOR_CORTO'].isin(selected_indicador)]

    return filtered_df.to_dict('records')


def wrap_text(text, width=110):
    """ Inserta saltos de línea en el texto para que se ajuste al ancho dado. """
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
    pie_charts = {}
    for indicador in df['INDICADOR_CORTO'].unique():
        data_indicador = df[df['INDICADOR_CORTO'] == indicador]
        tipo = data_indicador['TIPO'].iloc[0]  # Asume que todos los registros para un indicador comparten el mismo tipo
        indicador_full = data_indicador['INDICADOR'].iloc[0]
        indicador_full_wrapped = wrap_text(indicador_full)  # Envuelve el texto

        if tipo == 'CONTINUO':
            planificada = data_indicador['PLANIFICADA_META'].sum()
            cumplir = data_indicador['CUMPLIR_META'].sum()
        else:  # 'DISCRETO'
            planificada = data_indicador['PLANIFICADA_META'].sum()
            cumplir = data_indicador['CUMPLIR_META'].sum()

        if tipo == 'CONTINUO':
            # Manejo común para todos los CONTINUO, excepto CCDS-13 y CCDR-06
            if 'CCDS-13' in data_indicador['INDICADOR_CORTO'].values:
                cantidad_verificables = data_indicador[
                    'Nro_INSPECCIONES'].sum() if 'Nro_INSPECCIONES' in data_indicador else 0
            elif 'CCDR-06' in data_indicador['INDICADOR_CORTO'].values:
                cantidad_verificables = data_indicador[
                    'Nro_INSPECCIONES'].sum() if 'Nro_INSPECCIONES' in data_indicador else 0
            else:
                cantidad_verificables = data_indicador[
                    'CANTIDAD_VERIFICABLES'].sum() if 'CANTIDAD_VERIFICABLES' in data_indicador else 0
        else:  # 'DISCRETO'
            cantidad_verificables = data_indicador[
                'CANTIDAD_VERIFICABLES'].sum() if 'CANTIDAD_VERIFICABLES' in data_indicador else 0

        avance_global = min(cantidad_verificables / planificada * 100 if planificada > 0 else 0, 100)
        restante_global = max(100 - avance_global, 0)

        avance_corte = min(cantidad_verificables / cumplir * 100 if cumplir > 0 else 0, 100)
        restante_corte = max(100 - avance_corte, 0)

        pie_global = go.Figure(data=[
            go.Pie(labels=["Avance", "Restante"], values=[avance_global, restante_global],
                   marker=dict(colors=['#007BFF', '#D62828']), hole=.3)])
        pie_global.update_layout(
            title_text=f"Global: {indicador} ({cantidad_verificables} realizados / {planificada} planificados)",
            title_x=0.5,
            legend=dict(traceorder='normal'),
            annotations=[dict(text=indicador_full_wrapped, x=0.5, y=-0.15, xref="paper", yref="paper", showarrow=False,
                              font=dict(size=10, color="black"), bordercolor='black', borderpad=4, bgcolor='white',
                              borderwidth=1)]
        )

        pie_corte = go.Figure(data=[
            go.Pie(labels=["Avance", "Restante"], values=[avance_corte, restante_corte],
                   marker=dict(colors=['#007BFF', '#D62828']), hole=.3)])
        pie_corte.update_layout(
            title_text=f"Al {selected_date}: {indicador} ({cantidad_verificables} realizados / {cumplir} a cumplir al corte)",
            title_x=0.5,
            legend=dict(traceorder='normal'),
            annotations=[dict(text=indicador_full_wrapped, x=0.5, y=-0.15, xref="paper", yref="paper", showarrow=False,
                              font=dict(size=10, color="black"), bordercolor='black', borderpad=4, bgcolor='white',
                              borderwidth=1)]
        )

        pie_charts[indicador] = (pie_global, pie_corte)

    return pie_charts


def create_summary_pie_charts(df, selected_date):
    pie_charts = {}

    df_filtered = df[df['CUMPLIR_META'] != 0]

    if not df_filtered.empty:
        # Para Global Planificado
        df_filtered = df_filtered.copy()
        df_filtered.loc[:, 'Porcentaje_Global'] = df_filtered.apply(
            lambda row: min(row['CANTIDAD_VERIFICABLES'] / row['PLANIFICADA_META'], 1), axis=1)
        porcentaje_global_total = df_filtered['Porcentaje_Global'].sum() / len(df_filtered) * 100
        restante_global = max(100 - porcentaje_global_total, 0)

        pie_chart_global = go.Figure(data=[
            go.Pie(labels=["Avance", "Restante"], values=[porcentaje_global_total, restante_global],
                   marker=dict(colors=['#007BFF', '#D62828']), hole=.3)])
        pie_chart_global.update_layout(
            title_text="Global: 3.1 Porcentaje de Cumplimiento PACT",
            title_x=0.5,
            legend=dict(traceorder='normal')
        )
        pie_charts['Global Planificado'] = pie_chart_global

        # Para Fecha de Corte
        df_filtered.loc[:, 'Porcentaje_Corte'] = df_filtered.apply(
            lambda row: min(row['CANTIDAD_VERIFICABLES'] / row['CUMPLIR_META'], 1), axis=1)
        porcentaje_corte_total = df_filtered['Porcentaje_Corte'].sum() / len(df_filtered) * 100
        restante_corte = max(100 - porcentaje_corte_total, 0)

        pie_chart_corte = go.Figure(data=[
            go.Pie(labels=["Avance", "Restante"], values=[porcentaje_corte_total, restante_corte],
                   marker=dict(colors=['#007BFF', '#D62828']), hole=.3)])
        pie_chart_corte.update_layout(
            title_text=f"Al {selected_date}: 3.1 Porcentaje de Cumplimiento PACT ",
            title_x=0.5,
            legend=dict(traceorder='normal')
        )
        pie_charts['Fecha de Corte'] = pie_chart_corte

    return pie_charts
