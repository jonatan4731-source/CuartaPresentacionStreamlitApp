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
    
    # Calcular correlaciones (compatible con pandas antiguo)
    def calc_corr(g):
        return g['valor'].corr(g['Natalidad'])
    
    df_corr = df_clean.groupby('variable').apply(calc_corr).reset_index(name='correlation')
    
    # Calcular líneas de regresión
    def get_reg_line(g):
        m, b = np.polyfit(g['valor'], g['Natalidad'], 1)
        x_min, x_max = g['valor'].min(), g['valor'].max()
        return pd.DataFrame({
            'valor': [x_min, x_max],
            'Natalidad_pred': [m * x_min + b, m * x_max + b]
        })
    
    df_reg_lines = df_clean.groupby('variable').apply(get_reg_line).reset_index(drop=True)
    
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
# VIZ 3: MAPA MUNDIAL CON SLIDER (DEL NOTEBOOK)
# ============================================

def viz_mapa_mundial_natalidad(df):
    """
    Visualización 3: Mapa mundial interactivo con slider de años
    EXACTAMENTE como el notebook
    
    Args:
        df: DataFrame procesado (df_con_regiones del notebook)
        
    Returns:
        alt.Chart: Gráfico de Altair con mapa mundial
    """
    import altair as alt
    from vega_datasets import data
    
    alt.data_transformers.disable_max_rows()
    
    # Cargar geodata
    countries_url = alt.topo_feature(data.world_110m.url, 'countries')
    
    # Mapeo de países a IDs (COMPLETO del notebook)
    pais_a_id = {
        'Afghanistan': 4, 'Albania': 8, 'Algeria': 12, 'Angola': 24,
        'Argentina': 32, 'Armenia': 51, 'Australia': 36, 'Austria': 40,
        'Azerbaijan': 31, 'Bahamas, The': 44, 'Bangladesh': 50, 'Belarus': 112,
        'Belgium': 56, 'Belize': 84, 'Benin': 204, 'Bhutan': 64,
        'Bolivia': 68, 'Bosnia and Herzegovina': 70, 'Botswana': 72,
        'Brazil': 76, 'Brunei Darussalam': 96, 'Bulgaria': 100, 'Burkina Faso': 854,
        'Burundi': 108, 'Cambodia': 116, 'Cameroon': 120, 'Canada': 124,
        'Central African Republic': 140, 'Chad': 148, 'Chile': 152,
        'China': 156, 'Colombia': 170, 'Congo, Rep.': 178, 'Costa Rica': 188,
        'Croatia': 191, 'Cuba': 192, 'Cyprus': 196, 'Czechia': 203,
        'Congo, Dem. Rep.': 180, 'Denmark': 208, 'Djibouti': 262,
        'Dominican Republic': 214, 'Ecuador': 218, 'Egypt, Arab Rep.': 818,
        'El Salvador': 222, 'Equatorial Guinea': 226, 'Eritrea': 232,
        'Estonia': 233, 'Ethiopia': 231, 'Fiji': 242, 'Finland': 246,
        'France': 250, 'Gabon': 266, 'Gambia, The': 270, 'Georgia': 268,
        'Germany': 276, 'Ghana': 288, 'Greece': 300, 'Guatemala': 320,
        'Guinea': 324, 'Guinea-Bissau': 624, 'Guyana': 328, 'Haiti': 332,
        'Honduras': 340, 'Hungary': 348, 'Iceland': 352, 'India': 356,
        'Indonesia': 360, 'Iran, Islamic Rep.': 364, 'Iraq': 368, 'Ireland': 372,
        'Israel': 376, 'Italy': 380, "Cote d'Ivoire": 384, 'Jamaica': 388,
        'Japan': 392, 'Jordan': 400, 'Kazakhstan': 398, 'Kenya': 404,
        'Korea, Rep.': 410, 'Kuwait': 414, 'Kyrgyz Republic': 417, 'Lao PDR': 418,
        'Latvia': 428, 'Lebanon': 422, 'Lesotho': 426, 'Liberia': 430,
        'Libya': 434, 'Lithuania': 440, 'Luxembourg': 442, 'Madagascar': 450,
        'Malawi': 454, 'Malaysia': 458, 'Mali': 466, 'Mauritania': 478,
        'Mauritius': 480, 'Mexico': 484, 'Moldova': 498, 'Mongolia': 496,
        'Montenegro': 499, 'Morocco': 504, 'Mozambique': 508, 'Myanmar': 104,
        'Namibia': 516, 'Nepal': 524, 'Netherlands': 528, 'New Zealand': 554,
        'Nicaragua': 558, 'Niger': 562, 'Nigeria': 566, 'Norway': 578,
        'Oman': 512, 'Pakistan': 586, 'Panama': 591, 'Papua New Guinea': 598,
        'Paraguay': 600, 'Peru': 604, 'Philippines': 608, 'Poland': 616,
        'Portugal': 620, 'Qatar': 634, 'Romania': 642, 'Russian Federation': 643,
        'Rwanda': 646, 'Saudi Arabia': 682, 'Senegal': 686, 'Serbia': 688,
        'Sierra Leone': 694, 'Singapore': 702, 'Slovak Republic': 703, 'Slovenia': 705,
        'Solomon Islands': 90, 'Somalia': 706, 'South Africa': 710,
        'South Sudan': 728, 'Spain': 724, 'Sri Lanka': 144, 'Sudan': 729,
        'Suriname': 740, 'Eswatini': 748, 'Sweden': 752,
        'Switzerland': 756, 'Syrian Arab Republic': 760, 'Tajikistan': 762, 'Tanzania': 834,
        'Thailand': 764, 'Togo': 768, 'Trinidad and Tobago': 780,
        'Tunisia': 788, 'Turkiye': 792, 'Turkmenistan': 795, 'Uganda': 800,
        'Ukraine': 804, 'United Arab Emirates': 784, 'United Kingdom': 826,
        'United States': 840, 'Uruguay': 858, 'Uzbekistan': 860,
        'Vanuatu': 548, 'Venezuela, RB': 862, 'Viet Nam': 704, 'Yemen, Rep.': 887,
        'Zambia': 894, 'Zimbabwe': 716,
    }
    
    # Agregar ID al dataset
    df_con_regiones = df.copy()
    df_con_regiones['id'] = df_con_regiones['Pais'].map(pais_a_id)
    
    # Filtrar solo países con ID
    df_mapa = df_con_regiones[df_con_regiones['id'].notna()].copy()
    
    # Calcular estadísticas por año
    stats_por_año = df_mapa.groupby('Año')['Natalidad'].agg(['mean', 'min', 'max']).reset_index()
    
    # Asegurar tipos correctos
    df_mapa['Año'] = df_mapa['Año'].astype(int)
    stats_por_año['Año'] = stats_por_año['Año'].astype(int)
    
    años_únicos = sorted(df_mapa['Año'].unique())
    
    # Slider
    slider = alt.binding_range(
        min=int(años_únicos[0]),
        max=int(años_únicos[-1]),
        step=1,
        name='Año: '
    )
    
    year_param = alt.param(
        name='year',
        value=int(años_únicos[-1]),
        bind=slider
    )
    
    # Escala de colores
    color_scale = alt.Scale(
        domain=[5, 15, 25, 35, 45],
        range=['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'],
        type='threshold'
    )
    
    # Mapa base gris
    background = alt.Chart(countries_url).mark_geoshape(
        fill='#e0e0e0',
        stroke='white',
        strokeWidth=0.5
    ).project(
        type='naturalEarth1'
    ).properties(
        width=1080,
        height=600
    )
    
    # Crear capas por año (como en el notebook)
    data_layers = []
    for año in años_únicos:
        df_año = df_mapa[df_mapa['Año'] == año][['id', 'Pais', 'Continente', 'Region', 'Año', 'Natalidad']].copy()
        
        layer = alt.Chart(countries_url).mark_geoshape(
            stroke='white',
            strokeWidth=0.5
        ).encode(
            color=alt.Color(
                'Natalidad:Q',
                scale=color_scale,
                legend=None
            ),
            tooltip=[
                alt.Tooltip('Pais:N', title='País'),
                alt.Tooltip('Continente:N', title='Continente'),
                alt.Tooltip('Region:N', title='Región'),
                alt.Tooltip('Natalidad:Q', title='Natalidad', format='.2f')
            ]
        ).transform_lookup(
            lookup='id',
            from_=alt.LookupData(
                data=df_año,
                key='id',
                fields=['Pais', 'Continente', 'Region', 'Natalidad']
            )
        ).transform_filter(
            f'year == {año}'
        ).project(
            type='naturalEarth1'
        )
        
        data_layers.append(layer)
    
    # Capa dummy para leyenda
    legend_dummy = alt.Chart(df_mapa).mark_circle(opacity=0).encode(
        color=alt.Color(
            'Natalidad:Q',
            scale=color_scale,
            legend=alt.Legend(
                title='Natalidad (nacimientos/1000 hab)',
                titleFontSize=12,
                titleFontWeight='bold',
                labelFontSize=10
            )
        )
    )
    
    # Combinar todas las capas
    all_layers = [background] + [legend_dummy] + data_layers
    mapa_completo = alt.layer(*all_layers).properties(
        width=1080,
        height=600
    ).add_params(
        year_param
    )
    
    # Texto con estadísticas
    text_stats = alt.Chart(stats_por_año).mark_text(
        align='left',
        baseline='top',
        dx=10,
        dy=10,
        fontSize=13,
        fontWeight='bold',
        color='black'
    ).encode(
        text='label:N'
    ).transform_filter(
        'datum.Año == year'
    ).transform_calculate(
        label='toString(datum.Año) + " | Media Global: " + format(datum.mean, ".1f") + " | Rango: [" + format(datum.min, ".1f") + " - " + format(datum.max, ".1f") + "]"'
    ).properties(
        width=900,
        height=50
    ).add_params(
        year_param
    )
    
    # Combinar todo
    chart = (mapa_completo & text_stats).properties(
        title={
            'text': 'Evolución de la Natalidad Mundial',
            'subtitle': 'Usa el slider para explorar por año | Pasa el mouse sobre países para más información',
            'fontSize': 18,
            'fontWeight': 'bold',
            'anchor': 'start',
            'subtitleFontSize': 12,
            'subtitleColor': 'gray'
        }
    ).configure_view(
        strokeWidth=0
    )
    
    return chart


# ============================================
# VIZ 4: DISTRIBUCIÓN POR CONTINENTE (OPCIONAL)
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