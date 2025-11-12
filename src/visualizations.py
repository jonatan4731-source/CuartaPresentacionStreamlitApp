"""
Visualizaciones de Altair basadas en CuartaPresentacion.ipynb
"""

import altair as alt
import pandas as pd
import numpy as np


# ============================================
# CONFIGURACIÓN GLOBAL
# ============================================

# Habilitar manejo de datasets grandes
alt.data_transformers.disable_max_rows()


# ============================================
# VIZ 1: EVOLUCIÓN TEMPORAL POR REGIÓN
# ============================================

def viz_evolucion_temporal_regiones(df):
    """
    Visualización 1: Evolución temporal de natalidad por región
    Replica la visualización del notebook
    
    Args:
        df: DataFrame procesado con columnas Año, Natalidad, Continente, Region
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    # Preparar datos agregados
    df_limpio = df.dropna(subset=['Continente', 'Region'])
    
    datos_agregados = df_limpio.groupby(['Año', 'Continente', 'Region']).agg({
        'Natalidad': 'mean',
        'Pais': 'count'
    }).reset_index()
    
    datos_agregados.columns = ['Año', 'Continente', 'Region', 'Natalidad_Promedio', 'Num_Paises']
    datos_agregados['Natalidad_Promedio'] = datos_agregados['Natalidad_Promedio'].round(2)
    
    # Selector de continente
    selector_continente = alt.selection_point(
        fields=['Continente'],
        bind=alt.binding_select(
            options=[None] + sorted(list(datos_agregados['Continente'].unique())),
            labels=['Todos'] + sorted(list(datos_agregados['Continente'].unique())),
            name='Filtrar por Continente: '
        ),
        value='América'
    )
    
    # Selector para highlight de línea
    hover_region_selection = alt.selection_point(
        fields=['Region'],
        on='mouseover',
        nearest=True,
        empty=False
    )
    
    # Selector para highlight de punto
    hover_point_selection = alt.selection_point(
        on='mouseover',
        nearest=True,
        empty=False
    )
    
    # Gráfico base: líneas por región
    base = alt.Chart(datos_agregados).mark_line(
        strokeWidth=2.5
    ).encode(
        x=alt.X('Año:O',
                axis=alt.Axis(
                    title='Año',
                    labelAngle=-45,
                    titleFontSize=14,
                    titleFontWeight='bold',
                    labelFontSize=11
                )),
        y=alt.Y('Natalidad_Promedio:Q',
                axis=alt.Axis(
                    title='Natalidad Promedio (nacimientos por 1000 hab)',
                    titleFontSize=14,
                    titleFontWeight='bold',
                    labelFontSize=11
                ),
                scale=alt.Scale(zero=False)),
        color=alt.Color('Continente:N',
                      legend=alt.Legend(
                          title='Continente',
                          titleFontSize=13,
                          titleFontWeight='bold',
                          labelFontSize=11
                      )),
        detail='Region:N',
        opacity=alt.condition(hover_region_selection, alt.value(1), alt.value(0.1)),
        tooltip=[
            alt.Tooltip('Region:N', title='Región'),
            alt.Tooltip('Continente:N', title='Continente'),
            alt.Tooltip('Año:O', title='Año'),
            alt.Tooltip('Natalidad_Promedio:Q', title='Natalidad Promedio', format='.2f'),
            alt.Tooltip('Num_Paises:Q', title='Número de Países')
        ]
    ).transform_filter(
        selector_continente
    )
    
    # Puntos
    points = base.mark_circle(size=60).add_params(
        hover_region_selection,
        hover_point_selection
    )
    
    # Línea de tendencia
    tendencia_global = alt.Chart(datos_agregados).mark_line(
        strokeDash=[5, 5],
        strokeWidth=3,
        color='red',
        opacity=0.6
    ).encode(
        x='Año:O',
        y='mean(Natalidad_Promedio):Q'
    ).transform_filter(
        selector_continente
    )
    
    # Texto con región
    text_region = base.mark_text(
        align='left',
        dx=5,
        dy=-10,
        fontSize=12,
        fontWeight='bold'
    ).encode(
        text='Region:N',
        color=alt.value('black'),
        opacity=alt.condition(hover_point_selection, alt.value(1), alt.value(0))
    )
    
    # Combinar
    chart = (base + points + tendencia_global + text_region).add_params(
        selector_continente
    ).properties(
        width=1080,
        height=720,
        title={
            'text': 'Evolución Temporal de la Natalidad por Región Geográfica',
            'subtitle': [
                'Promedio de nacimientos por 1000 habitantes | Interactivo: Selecciona continente y pasa el mouse sobre las líneas',
                'Línea roja punteada: Tendencia promedio del continente seleccionado'
            ],
            'fontSize': 18,
            'fontWeight': 'bold',
            'anchor': 'start',
            'subtitleFontSize': 12,
            'subtitleColor': 'gray'
        }
    ).configure_axis(
        gridColor='lightgray',
        gridOpacity=0.5
    ).configure_view(
        strokeWidth=0
    )
    
    return chart


# ============================================
# VIZ 2: CORRELACIONES INTERACTIVAS
# ============================================

def viz_correlaciones_interactivas(df):
    """
    Visualización 2: Scatter plot interactivo de correlaciones
    
    Args:
        df: DataFrame procesado
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    # Variables socioeconómicas clave
    variables_analisis = {
        'EsperanzaVida': 'Esperanza de Vida (años)',
        'PIB_per_capita': 'PIB per cápita (USD)',
        'Urbanizacion': 'Urbanización (%)',
        'GastoSalud': 'Gasto en Salud (% PIB)',
        'AccesoEducacion': 'Acceso a Educación (%)',
        'Desempleo': 'Desempleo (%)',
        'AccesoAguaPotable': 'Acceso a Agua Potable (%)',
        'MujeresParlamento': 'Mujeres en Parlamento (%)',
    }
    
    # Filtrar solo variables disponibles
    variables_disponibles = {
        var: label for var, label in variables_analisis.items()
        if var in df.columns
    }
    
    # Preparar datos (últimos 5 años)
    columnas_necesarias = ['Año', 'Pais', 'Natalidad', 'Continente'] + list(variables_disponibles.keys())
    df_viz = df[columnas_necesarias].dropna(subset=['Natalidad'])
    
    año_max = df_viz['Año'].max()
    df_viz = df_viz[df_viz['Año'] >= año_max - 4].copy()
    
    # Transformar a formato long
    df_long = df_viz.melt(
        id_vars=['Año', 'Pais', 'Natalidad', 'Continente'],
        value_vars=list(variables_disponibles.keys()),
        var_name='variable',
        value_name='valor'
    )
    
    # Limpiar datos
    df_clean = df_long.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=['valor', 'Natalidad'])
    
    # Calcular correlaciones
    df_corr = df_clean.groupby('variable', include_groups=False).apply(
        lambda g: g['valor'].corr(g['Natalidad'])
    ).reset_index(name='correlation')
    
    # Calcular líneas de regresión
    def get_reg_line(g):
        m, b = np.polyfit(g['valor'], g['Natalidad'], 1)
        x_min, x_max = g['valor'].min(), g['valor'].max()
        return pd.DataFrame({
            'valor': [x_min, x_max],
            'Natalidad_pred': [m * x_min + b, m * x_max + b]
        })
    
    df_reg_lines = df_clean.groupby('variable', include_groups=False).apply(get_reg_line).reset_index()
    
    # Selectores
    variable_input = alt.binding_select(
        options=list(variables_disponibles.keys()),
        labels=list(variables_disponibles.values()),
        name='Variable a comparar: '
    )
    
    variable_selection = alt.selection_point(
        fields=['variable'],
        bind=variable_input,
        value=list(variables_disponibles.keys())[0]
    )
    
    selector_continente = alt.selection_point(
        fields=['Continente'],
        bind='legend',
        on='click'
    )
    
    color_scale = alt.Scale(
        domain=['África', 'América', 'Asia', 'Europa', 'Oceanía'],
        range=['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
    )
    
    # Scatter plot
    scatter = alt.Chart(df_clean).mark_circle(
        size=100,
        opacity=0.7
    ).encode(
        x=alt.X('valor:Q',
                scale=alt.Scale(zero=False),
                axis=alt.Axis(
                    titleFontSize=13,
                    titleFontWeight='bold',
                    labelFontSize=11
                )),
        y=alt.Y('Natalidad:Q',
                scale=alt.Scale(zero=False),
                axis=alt.Axis(
                    title='Natalidad (nacimientos por 1000 hab)',
                    titleFontSize=13,
                    titleFontWeight='bold',
                    labelFontSize=11
                )),
        color=alt.condition(
            selector_continente,
            alt.Color('Continente:N',
                      scale=color_scale,
                      legend=alt.Legend(
                          title='Continente (click para filtrar)',
                          titleFontSize=12,
                          titleFontWeight='bold',
                          labelFontSize=11,
                          orient='right'
                      )),
            alt.value('lightgray')
        ),
        opacity=alt.condition(selector_continente, alt.value(0.8), alt.value(0.1)),
        tooltip=[
            alt.Tooltip('Pais:N', title='País'),
            alt.Tooltip('Continente:N', title='Continente'),
            alt.Tooltip('Año:O', title='Año'),
            alt.Tooltip('Natalidad:Q', title='Natalidad', format='.2f'),
            alt.Tooltip('valor:Q', title='Valor Variable', format='.2f')
        ]
    ).add_params(
        variable_selection,
        selector_continente
    ).transform_filter(
        variable_selection
    )
    
    # Línea de regresión
    regression = alt.Chart(df_reg_lines).mark_line(
        color='black',
        strokeWidth=3,
        strokeDash=[5, 5]
    ).encode(
        x=alt.X('valor:Q'),
        y=alt.Y('Natalidad_pred:Q', title='Natalidad')
    ).add_params(
        variable_selection
    ).transform_filter(
        variable_selection
    )
    
    # Texto de correlación
    correlation_text = alt.Chart(df_corr).mark_text(
        align='left',
        baseline='top',
        dx=10,
        dy=10,
        fontSize=14,
        fontWeight='bold',
        color='darkred'
    ).transform_filter(
        variable_selection
    ).transform_calculate(
        correlation_label='"Correlación: " + format(datum.correlation, ".3f")'
    ).encode(
        text='correlation_label:N',
        x=alt.value(10),
        y=alt.value(10)
    )
    
    # Combinar
    chart = (scatter + regression + correlation_text).properties(
        width=1080,
        height=720,
        title={
            'text': 'Explorador de Correlaciones: Variables vs Natalidad',
            'subtitle': [
                'Selecciona una variable para explorar su relación con la natalidad',
                'Click para filtrar por continente | Línea Negra: tendencia lineal'
            ],
            'fontSize': 16,
            'fontWeight': 'bold',
            'anchor': 'start',
            'subtitleFontSize': 11,
            'subtitleColor': 'gray'
        }
    ).configure_axis(
        gridColor='lightgray',
        gridOpacity=0.5
    ).configure_view(
        strokeWidth=0
    ).interactive()
    
    return chart


# ============================================
# VIZ 3: DISTRIBUCIÓN POR CONTINENTE
# ============================================

def viz_distribucion_continentes(df, year=None):
    """
    Visualización 3: Boxplot de distribución por continente
    
    Args:
        df: DataFrame procesado
        year: Año específico (si None, usa el más reciente)
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    # Usar año más reciente si no se especifica
    if year is None:
        year = int(df['Año'].max())
    
    # Filtrar datos
    df_year = df[df['Año'] == year].copy()
    df_year = df_year.dropna(subset=['Natalidad', 'Continente'])
    
    # Boxplot por continente
    chart = alt.Chart(df_year).mark_boxplot(
        size=50
    ).encode(
        x=alt.X('Continente:N',
                axis=alt.Axis(
                    title='Continente',
                    titleFontSize=14,
                    titleFontWeight='bold',
                    labelFontSize=12,
                    labelAngle=-45
                )),
        y=alt.Y('Natalidad:Q',
                axis=alt.Axis(
                    title='Tasa de Natalidad (por 1000 hab)',
                    titleFontSize=14,
                    titleFontWeight='bold',
                    labelFontSize=11
                ),
                scale=alt.Scale(zero=False)),
        color=alt.Color('Continente:N',
                       scale=alt.Scale(
                           domain=['África', 'América', 'Asia', 'Europa', 'Oceanía'],
                           range=['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
                       ),
                       legend=None),
        tooltip=[
            alt.Tooltip('Continente:N', title='Continente'),
            alt.Tooltip('min(Natalidad):Q', title='Mínimo', format='.2f'),
            alt.Tooltip('q1(Natalidad):Q', title='Q1', format='.2f'),
            alt.Tooltip('median(Natalidad):Q', title='Mediana', format='.2f'),
            alt.Tooltip('q3(Natalidad):Q', title='Q3', format='.2f'),
            alt.Tooltip('max(Natalidad):Q', title='Máximo', format='.2f')
        ]
    ).properties(
        width=800,
        height=500,
        title={
            'text': f'Distribución de Natalidad por Continente ({year})',
            'subtitle': 'Boxplot mostrando mediana, cuartiles y valores atípicos',
            'fontSize': 16,
            'fontWeight': 'bold',
            'anchor': 'start',
            'subtitleFontSize': 12,
            'subtitleColor': 'gray'
        }
    ).configure_axis(
        gridColor='lightgray',
        gridOpacity=0.5
    ).configure_view(
        strokeWidth=0
    )
    
    return chart


# ============================================
# UTILIDADES
# ============================================

def get_available_visualizations():
    """Retorna lista de visualizaciones disponibles"""
    return [
        {
            'id': 'evolucion_temporal',
            'nombre': 'Evolución Temporal por Región',
            'descripcion': 'Líneas interactivas mostrando la evolución de natalidad por región geográfica',
            'funcion': viz_evolucion_temporal_regiones
        },
        {
            'id': 'correlaciones',
            'nombre': 'Explorador de Correlaciones',
            'descripcion': 'Scatter plot interactivo para explorar relaciones entre variables',
            'funcion': viz_correlaciones_interactivas
        },
        {
            'id': 'distribucion',
            'nombre': 'Distribución por Continente',
            'descripcion': 'Boxplot mostrando la distribución de natalidad por continente',
            'funcion': viz_distribucion_continentes
        }
    ]


if __name__ == "__main__":
    print("✅ Módulo de visualizaciones cargado correctamente")
    print(f"📊 Visualizaciones disponibles: {len(get_available_visualizations())}")