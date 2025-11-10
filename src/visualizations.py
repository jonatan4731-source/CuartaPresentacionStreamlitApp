"""
Funciones de visualización con Altair para el proyecto de natalidad
"""

import altair as alt
import pandas as pd


# ============================================
# CONFIGURACIÓN GLOBAL DE ALTAIR
# ============================================

# Tema personalizado
def configure_theme():
    """Configura el tema global de Altair"""
    return {
        'config': {
            'view': {'strokeWidth': 0},
            'axis': {
                'labelFontSize': 12,
                'titleFontSize': 14,
                'titleFont': 'Arial',
                'labelFont': 'Arial'
            },
            'legend': {
                'labelFontSize': 12,
                'titleFontSize': 13
            },
            'title': {
                'fontSize': 16,
                'font': 'Arial',
                'anchor': 'start'
            }
        }
    }


# ============================================
# VISUALIZACIÓN 1: TENDENCIA TEMPORAL
# ============================================

def plot_temporal_trend(df, country=None):
    """
    Gráfico de líneas: evolución temporal de natalidad
    
    Args:
        df (pd.DataFrame): Dataset con columnas 'year', 'natalidad', 'pais'
        country (str, optional): País específico para destacar
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    # TODO: Reemplazar con tu código de Altair
    
    if country:
        # Versión con país destacado
        base = alt.Chart(df).mark_line(opacity=0.2).encode(
            x=alt.X('year:Q', title='Año'),
            y=alt.Y('natalidad:Q', title='Tasa de Natalidad'),
            detail='pais:N',
            color=alt.value('lightgray')
        )
        
        highlight = alt.Chart(df[df['pais'] == country]).mark_line(
            strokeWidth=3,
            color='steelblue'
        ).encode(
            x='year:Q',
            y='natalidad:Q'
        )
        
        chart = (base + highlight).properties(
            width=700,
            height=400,
            title=f'Evolución de Natalidad - {country}'
        )
    else:
        # Versión promedio global
        df_agg = df.groupby('year')['natalidad'].mean().reset_index()
        
        chart = alt.Chart(df_agg).mark_line(
            point=True,
            color='steelblue',
            strokeWidth=3
        ).encode(
            x=alt.X('year:Q', title='Año'),
            y=alt.Y('natalidad:Q', title='Tasa de Natalidad Promedio'),
            tooltip=['year:Q', 'natalidad:Q']
        ).properties(
            width=700,
            height=400,
            title='Evolución Global de la Tasa de Natalidad'
        )
    
    return chart


# ============================================
# VISUALIZACIÓN 2: COMPARACIÓN REGIONAL
# ============================================

def plot_regional_comparison(df, year=2023):
    """
    Gráfico de barras: comparación por región
    
    Args:
        df (pd.DataFrame): Dataset con columnas 'region', 'natalidad'
        year (int): Año a visualizar
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    # TODO: Reemplazar con tu código de Altair
    
    df_filtered = df[df['year'] == year]
    df_agg = df_filtered.groupby('region')['natalidad'].mean().reset_index()
    df_agg = df_agg.sort_values('natalidad', ascending=False)
    
    chart = alt.Chart(df_agg).mark_bar().encode(
        x=alt.X('natalidad:Q', title='Tasa de Natalidad Promedio'),
        y=alt.Y('region:N', title='Región', sort='-x'),
        color=alt.Color('natalidad:Q', 
                       scale=alt.Scale(scheme='blues'),
                       legend=None),
        tooltip=['region:N', 'natalidad:Q']
    ).properties(
        width=700,
        height=400,
        title=f'Comparación de Natalidad por Región ({year})'
    )
    
    return chart


# ============================================
# VISUALIZACIÓN 3: CORRELACIONES
# ============================================

def plot_correlation_heatmap(df):
    """
    Heatmap de correlaciones entre variables
    
    Args:
        df (pd.DataFrame): Dataset con variables numéricas
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    # TODO: Reemplazar con tu código de Altair
    
    # Seleccionar variables numéricas
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Calcular matriz de correlación
    corr_matrix = df[numeric_cols].corr().reset_index()
    corr_matrix = corr_matrix.melt(id_vars='index')
    corr_matrix.columns = ['Variable1', 'Variable2', 'Correlacion']
    
    chart = alt.Chart(corr_matrix).mark_rect().encode(
        x=alt.X('Variable1:N', title=None),
        y=alt.Y('Variable2:N', title=None),
        color=alt.Color('Correlacion:Q',
                       scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                       legend=alt.Legend(title='Correlación')),
        tooltip=['Variable1:N', 'Variable2:N', 'Correlacion:Q']
    ).properties(
        width=600,
        height=600,
        title='Matriz de Correlaciones'
    )
    
    return chart


# ============================================
# VISUALIZACIÓN 4: SCATTER PLOT INTERACTIVO
# ============================================

def plot_scatter_interactive(df, x_var, y_var, color_var='region'):
    """
    Scatter plot interactivo con selección
    
    Args:
        df (pd.DataFrame): Dataset
        x_var (str): Variable eje X
        y_var (str): Variable eje Y
        color_var (str): Variable para color
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    # Selector interactivo
    selection = alt.selection_point(fields=[color_var], bind='legend')
    
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(f'{x_var}:Q', title=x_var.replace('_', ' ').title()),
        y=alt.Y(f'{y_var}:Q', title=y_var.replace('_', ' ').title()),
        color=alt.Color(f'{color_var}:N', legend=alt.Legend(title=color_var.title())),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=[x_var, y_var, color_var, 'pais:N']
    ).add_params(
        selection
    ).properties(
        width=700,
        height=400,
        title=f'{y_var} vs {x_var}'
    )
    
    return chart


# ============================================
# VISUALIZACIÓN 5: TOP/BOTTOM PAÍSES
# ============================================

def plot_top_bottom_countries(df, year=2023, top_n=10):
    """
    Gráfico de barras: top y bottom países
    
    Args:
        df (pd.DataFrame): Dataset
        year (int): Año a visualizar
        top_n (int): Número de países a mostrar
        
    Returns:
        alt.Chart: Gráfico de Altair
    """
    df_filtered = df[df['year'] == year].copy()
    df_sorted = df_filtered.sort_values('natalidad', ascending=False)
    
    # Top países
    df_top = df_sorted.head(top_n).copy()
    df_top['categoria'] = 'Top'
    
    # Bottom países
    df_bottom = df_sorted.tail(top_n).copy()
    df_bottom['categoria'] = 'Bottom'
    
    df_combined = pd.concat([df_top, df_bottom])
    
    chart = alt.Chart(df_combined).mark_bar().encode(
        x=alt.X('natalidad:Q', title='Tasa de Natalidad'),
        y=alt.Y('pais:N', title='País', sort='-x'),
        color=alt.Color('categoria:N', 
                       scale=alt.Scale(domain=['Top', 'Bottom'],
                                     range=['#2ecc71', '#e74c3c']),
                       legend=alt.Legend(title='Categoría')),
        tooltip=['pais:N', 'natalidad:Q', 'categoria:N']
    ).properties(
        width=700,
        height=400,
        title=f'Top {top_n} y Bottom {top_n} Países por Natalidad ({year})'
    )
    
    return chart


# ============================================
# UTILIDADES
# ============================================

def get_available_years(df):
    """Retorna lista de años disponibles"""
    return sorted(df['year'].unique().tolist())


def get_available_regions(df):
    """Retorna lista de regiones disponibles"""
    return sorted(df['region'].unique().tolist())


def get_available_countries(df):
    """Retorna lista de países disponibles"""
    return sorted(df['pais'].unique().tolist())


if __name__ == "__main__":
    # Pruebas locales
    print("Módulo de visualizaciones cargado correctamente")