"""
Funciones de visualización con Altair para el proyecto de natalidad
Basado en CuartaPresentacion.ipynb
"""

import altair as alt
import pandas as pd
import streamlit as st


# ============================================
# CONFIGURACIÓN GLOBAL
# ============================================

# Configurar Altair para Streamlit
alt.data_transformers.enable('default')
alt.renderers.enable('default')


# ============================================
# VISUALIZACIÓN 1: EVOLUCIÓN TEMPORAL GLOBAL
# ============================================

def plot_birth_rate_evolution(df, countries=None):
    """
    Gráfico de líneas: evolución temporal de la tasa de natalidad
    
    Args:
        df (pd.DataFrame): Dataset con columnas Year, Birth Rate, Country Name
        countries (list, optional): Lista de países a destacar
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    if 'Year' not in df.columns or 'Birth Rate' not in df.columns:
        st.warning(" Columnas necesarias no encontradas")
        return None
    
    # Calcular promedio global por año
    df_global = df.groupby('Year')['Birth Rate'].mean().reset_index()
    df_global.columns = ['Year', 'Birth Rate']
    
    # Gráfico base: promedio global
    base = alt.Chart(df_global).mark_line(
        color='steelblue',
        strokeWidth=3,
        point=alt.OverlayMarkDef(color='steelblue', size=60)
    ).encode(
        x=alt.X('Year:Q', 
                title='Año',
                axis=alt.Axis(format='d')),
        y=alt.Y('Birth Rate:Q', 
                title='Tasa de Natalidad (por 1000 habitantes)',
                scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('Year:Q', title='Año', format='d'),
            alt.Tooltip('Birth Rate:Q', title='Tasa de Natalidad', format='.2f')
        ]
    ).properties(
        width=700,
        height=400,
        title='Evolución Global de la Tasa de Natalidad'
    )
    
    # Si se especifican países, agregar sus líneas
    if countries and 'Country Name' in df.columns:
        df_countries = df[df['Country Name'].isin(countries)]
        
        countries_lines = alt.Chart(df_countries).mark_line(
            strokeWidth=2,
            opacity=0.7
        ).encode(
            x='Year:Q',
            y='Birth Rate:Q',
            color=alt.Color('Country Name:N', 
                          legend=alt.Legend(title='País')),
            tooltip=['Year:Q', 'Country Name:N', 'Birth Rate:Q']
        )
        
        return (base + countries_lines).interactive()
    
    return base.interactive()


# ============================================
# VISUALIZACIÓN 2: COMPARACIÓN POR REGIÓN
# ============================================

def plot_regional_comparison(df, year=None):
    """
    Gráfico de barras: comparación de natalidad por región
    
    Args:
        df (pd.DataFrame): Dataset con columnas Region, Birth Rate
        year (int, optional): Año específico a visualizar
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    if 'Region' not in df.columns or 'Birth Rate' not in df.columns:
        st.warning(" Columnas necesarias no encontradas")
        return None
    
    df_viz = df.copy()
    
    # Filtrar por año si se especifica
    if year and 'Year' in df.columns:
        df_viz = df_viz[df_viz['Year'] == year]
        title = f'Tasa de Natalidad por Región ({year})'
    else:
        title = 'Tasa de Natalidad Promedio por Región'
    
    # Agrupar por región
    df_regional = df_viz.groupby('Region')['Birth Rate'].mean().reset_index()
    df_regional = df_regional.sort_values('Birth Rate', ascending=False)
    
    chart = alt.Chart(df_regional).mark_bar().encode(
        x=alt.X('Birth Rate:Q', 
                title='Tasa de Natalidad Promedio',
                scale=alt.Scale(zero=True)),
        y=alt.Y('Region:N', 
                title='Región',
                sort='-x'),
        color=alt.Color('Birth Rate:Q',
                       scale=alt.Scale(scheme='blues'),
                       legend=None),
        tooltip=[
            alt.Tooltip('Region:N', title='Región'),
            alt.Tooltip('Birth Rate:Q', title='Tasa de Natalidad', format='.2f')
        ]
    ).properties(
        width=700,
        height=400,
        title=title
    )
    
    return chart.interactive()


# ============================================
# VISUALIZACIÓN 3: TOP PAÍSES
# ============================================

def plot_top_countries(df, n=15, year=None, ascending=False):
    """
    Gráfico de barras horizontales: top países por natalidad
    
    Args:
        df (pd.DataFrame): Dataset
        n (int): Número de países a mostrar
        year (int, optional): Año específico
        ascending (bool): Si True, muestra los países con menor natalidad
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    if 'Country Name' not in df.columns or 'Birth Rate' not in df.columns:
        st.warning(" Columnas necesarias no encontradas")
        return None
    
    df_viz = df.copy()
    
    # Filtrar por año si se especifica
    if year and 'Year' in df.columns:
        df_viz = df_viz[df_viz['Year'] == year]
    else:
        # Usar el año más reciente
        if 'Year' in df.columns:
            df_viz = df_viz[df_viz['Year'] == df_viz['Year'].max()]
            year = df_viz['Year'].max()
    
    # Obtener top/bottom países
    df_viz = df_viz.sort_values('Birth Rate', ascending=ascending)
    df_top = df_viz.head(n)[['Country Name', 'Birth Rate', 'Region']].copy()
    
    title = f'{"Bottom" if ascending else "Top"} {n} Países por Tasa de Natalidad'
    if year:
        title += f' ({int(year)})'
    
    chart = alt.Chart(df_top).mark_bar().encode(
        x=alt.X('Birth Rate:Q', 
                title='Tasa de Natalidad',
                scale=alt.Scale(zero=True)),
        y=alt.Y('Country Name:N', 
                title='País',
                sort='-x' if not ascending else 'x'),
        color=alt.Color('Region:N',
                       legend=alt.Legend(title='Región'),
                       scale=alt.Scale(scheme='category20')),
        tooltip=[
            alt.Tooltip('Country Name:N', title='País'),
            alt.Tooltip('Birth Rate:Q', title='Tasa de Natalidad', format='.2f'),
            alt.Tooltip('Region:N', title='Región')
        ]
    ).properties(
        width=700,
        height=500,
        title=title
    )
    
    return chart.interactive()


# ============================================
# VISUALIZACIÓN 4: SCATTER PLOT CORRELACIÓN
# ============================================

def plot_correlation_scatter(df, x_var, y_var, year=None):
    """
    Scatter plot: correlación entre dos variables
    
    Args:
        df (pd.DataFrame): Dataset
        x_var (str): Variable eje X
        y_var (str): Variable eje Y
        year (int, optional): Año específico
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    if x_var not in df.columns or y_var not in df.columns:
        st.warning(f" Variables {x_var} o {y_var} no encontradas")
        return None
    
    df_viz = df.copy()
    
    # Filtrar por año si se especifica
    if year and 'Year' in df.columns:
        df_viz = df_viz[df_viz['Year'] == year]
        title = f'{y_var} vs {x_var} ({year})'
    else:
        if 'Year' in df.columns:
            df_viz = df_viz[df_viz['Year'] == df_viz['Year'].max()]
        title = f'{y_var} vs {x_var}'
    
    # Remover nulos
    df_viz = df_viz.dropna(subset=[x_var, y_var])
    
    # Selector de región interactivo
    selection = alt.selection_point(fields=['Region'], bind='legend')
    
    chart = alt.Chart(df_viz).mark_circle(size=80).encode(
        x=alt.X(f'{x_var}:Q', 
                title=x_var.replace('_', ' ').title(),
                scale=alt.Scale(zero=False)),
        y=alt.Y(f'{y_var}:Q', 
                title=y_var.replace('_', ' ').title(),
                scale=alt.Scale(zero=False)),
        color=alt.condition(
            selection,
            alt.Color('Region:N', legend=alt.Legend(title='Región')),
            alt.value('lightgray')
        ),
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2)),
        tooltip=[
            alt.Tooltip('Country Name:N', title='País'),
            alt.Tooltip(f'{x_var}:Q', format='.2f'),
            alt.Tooltip(f'{y_var}:Q', format='.2f'),
            alt.Tooltip('Region:N', title='Región')
        ]
    ).add_params(
        selection
    ).properties(
        width=700,
        height=500,
        title=title
    )
    
    # Agregar línea de tendencia
    trend = chart.transform_regression(
        x_var, y_var
    ).mark_line(color='red', strokeDash=[5, 5])
    
    return (chart + trend).interactive()


# ============================================
# VISUALIZACIÓN 5: DISTRIBUCIÓN (HISTOGRAMA)
# ============================================

def plot_distribution(df, variable='Birth Rate', bins=30):
    """
    Histograma: distribución de una variable
    
    Args:
        df (pd.DataFrame): Dataset
        variable (str): Variable a visualizar
        bins (int): Número de bins
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    if variable not in df.columns:
        st.warning(f" Variable {variable} no encontrada")
        return None
    
    df_viz = df[[variable]].dropna()
    
    chart = alt.Chart(df_viz).mark_bar().encode(
        x=alt.X(f'{variable}:Q',
                bin=alt.Bin(maxbins=bins),
                title=variable.replace('_', ' ').title()),
        y=alt.Y('count()',
                title='Frecuencia'),
        tooltip=['count()', f'{variable}:Q']
    ).properties(
        width=700,
        height=400,
        title=f'Distribución de {variable.replace("_", " ").title()}'
    )
    
    # Agregar línea de media
    mean_line = alt.Chart(df_viz).mark_rule(color='red', strokeWidth=2).encode(
        x=f'mean({variable}):Q',
        size=alt.value(2)
    )
    
    return (chart + mean_line).interactive()


# ============================================
# VISUALIZACIÓN 6: MAPA DE CALOR TEMPORAL
# ============================================

def plot_temporal_heatmap(df, countries=None):
    """
    Mapa de calor: evolución temporal por país
    
    Args:
        df (pd.DataFrame): Dataset
        countries (list, optional): Lista de países a visualizar
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    if 'Year' not in df.columns or 'Birth Rate' not in df.columns or 'Country Name' not in df.columns:
        st.warning(" Columnas necesarias no encontradas")
        return None
    
    df_viz = df.copy()
    
    # Filtrar países si se especifican
    if countries:
        df_viz = df_viz[df_viz['Country Name'].isin(countries)]
    else:
        # Tomar una muestra de países para que sea legible
        top_countries = df_viz.groupby('Country Name')['Birth Rate'].mean().nlargest(15).index
        df_viz = df_viz[df_viz['Country Name'].isin(top_countries)]
    
    chart = alt.Chart(df_viz).mark_rect().encode(
        x=alt.X('Year:O', 
                title='Año',
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Country Name:N', 
                title='País'),
        color=alt.Color('Birth Rate:Q',
                       scale=alt.Scale(scheme='blues'),
                       legend=alt.Legend(title='Tasa de Natalidad')),
        tooltip=[
            alt.Tooltip('Country Name:N', title='País'),
            alt.Tooltip('Year:O', title='Año'),
            alt.Tooltip('Birth Rate:Q', title='Tasa de Natalidad', format='.2f')
        ]
    ).properties(
        width=700,
        height=400,
        title='Evolución Temporal de Natalidad por País'
    )
    
    return chart.interactive()


# ============================================
# UTILIDADES
# ============================================

def get_numeric_columns(df):
    """Retorna lista de columnas numéricas"""
    return df.select_dtypes(include=['float64', 'int64']).columns.tolist()


if __name__ == "__main__":
    print(" Módulo de visualizaciones cargado correctamente")