"""
Funciones de procesamiento y pipeline para el proyecto de natalidad
Basado en CuartaPresentacion.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st

# ============================================
# CARGA DE DATOS
# ============================================

@st.cache_data
def load_data(filepath='data/raw/df_con_features_temporales.csv', add_geography=True):
    """
    Carga el dataset principal con cache de Streamlit
    
    Args:
        filepath (str): Ruta al archivo CSV
        add_geography (bool): Si True, agrega columnas de región y continente
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    import os
    
    try:
        # Verificar si el archivo existe
        if not os.path.exists(filepath):
            error_msg = f"❌ Archivo no encontrado: {filepath}"
            print(error_msg)  # Para scripts sin Streamlit
            try:
                st.error(error_msg)  # Para Streamlit
            except:
                pass
            return None
        
        df = pd.read_csv(filepath)
        print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Agregar información geográfica si se solicita
        if add_geography:
            from src.geography_utils import add_geographic_info
            df = add_geographic_info(df)
            print(f"✅ Información geográfica agregada")
        
        return df
    except Exception as e:
        error_msg = f"❌ Error al cargar datos: {e}"
        print(error_msg)  # Para scripts sin Streamlit
        try:
            st.error(error_msg)  # Para Streamlit
        except:
            pass
        return None


# ============================================
# INFORMACIÓN DEL DATASET
# ============================================

def get_data_info(df):
    """
    Retorna información básica del dataset
    
    Returns:
        dict: Diccionario con estadísticas
    """
    # Detectar nombres de columnas (español o inglés)
    year_col = 'Año' if 'Año' in df.columns else 'Year' if 'Year' in df.columns else None
    country_col = 'Pais' if 'Pais' in df.columns else 'País' if 'País' in df.columns else 'Country Name' if 'Country Name' in df.columns else None
    region_col = 'Region' if 'Region' in df.columns else 'Región' if 'Región' in df.columns else None
    
    info = {
        'n_filas': len(df),
        'n_columnas': len(df.columns),
        'años_disponibles': sorted(df[year_col].unique().tolist()) if year_col else [],
        'paises_unicos': df[country_col].nunique() if country_col else 0,
        'regiones_unicas': df[region_col].nunique() if region_col else 0,
        'columnas': df.columns.tolist(),
        'tipos': df.dtypes.to_dict(),
        'nulos_totales': df.isnull().sum().sum(),
        'nulos_por_columna': df.isnull().sum().to_dict()
    }
    return info


def get_data_summary(df):
    """
    Retorna estadísticas descriptivas del dataset
    
    Returns:
        pd.DataFrame: Describe del dataset
    """
    return df.describe()


# ============================================
# PREPROCESAMIENTO Y LIMPIEZA
# ============================================

def clean_data(df):
    """
    Limpia el dataset: manejo de nulos, outliers, etc.
    Basado en tu notebook
    
    Args:
        df (pd.DataFrame): Dataset original
        
    Returns:
        pd.DataFrame: Dataset limpio
    """
    df_clean = df.copy()
    
    # Eliminar filas con muchos nulos (ajustar threshold según tu análisis)
    threshold = len(df_clean.columns) * 0.5
    df_clean = df_clean.dropna(thresh=threshold)
    
    # Imputación de nulos numéricos con la mediana
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    return df_clean


def create_lag_features(df, target_col='Tasa de natalidad', lags=[1, 2, 3]):
    """
    Crea features de rezago temporal (lag features)
    Similar a tu notebook donde creás features temporales
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Columna objetivo
        lags (list): Lista de rezagos a crear
        
    Returns:
        pd.DataFrame: Dataset con lag features
    """
    df_with_lags = df.copy()
    
    country_col = 'País' if 'País' in df.columns else 'Country Name'
    
    if country_col in df.columns:
        for lag in lags:
            df_with_lags[f'{target_col}_lag_{lag}'] = df_with_lags.groupby(country_col)[target_col].shift(lag)
    
    return df_with_lags


def create_rolling_features(df, target_col='Tasa de natalidad', windows=[3, 5]):
    """
    Crea features de ventana móvil (rolling features)
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Columna objetivo
        windows (list): Lista de ventanas a crear
        
    Returns:
        pd.DataFrame: Dataset con rolling features
    """
    df_with_rolling = df.copy()
    
    country_col = 'País' if 'País' in df.columns else 'Country Name'
    
    if country_col in df.columns:
        for window in windows:
            df_with_rolling[f'{target_col}_rolling_mean_{window}'] = df_with_rolling.groupby(country_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df_with_rolling[f'{target_col}_rolling_std_{window}'] = df_with_rolling.groupby(country_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
    
    return df_with_rolling


# ============================================
# FILTROS Y SELECCIÓN
# ============================================

def filter_by_year(df, year_start, year_end):
    """
    Filtra dataset por rango de años
    
    Args:
        df (pd.DataFrame): Dataset completo
        year_start (int): Año inicial
        year_end (int): Año final
        
    Returns:
        pd.DataFrame: Dataset filtrado
    """
    year_col = 'Año' if 'Año' in df.columns else 'Year'
    if year_col in df.columns:
        return df[(df[year_col] >= year_start) & (df[year_col] <= year_end)]
    return df


def filter_by_region(df, regions):
    """
    Filtra dataset por región(es)
    
    Args:
        df (pd.DataFrame): Dataset completo
        regions (list): Lista de regiones
        
    Returns:
        pd.DataFrame: Dataset filtrado
    """
    region_col = 'Región' if 'Región' in df.columns else 'Region'
    if region_col in df.columns and regions:
        return df[df[region_col].isin(regions)]
    return df


def filter_by_country(df, countries):
    """
    Filtra dataset por país(es)
    
    Args:
        df (pd.DataFrame): Dataset completo
        countries (list): Lista de países
        
    Returns:
        pd.DataFrame: Dataset filtrado
    """
    country_col = 'País' if 'País' in df.columns else 'Country Name'
    if country_col in df.columns and countries:
        return df[df[country_col].isin(countries)]
    return df


# ============================================
# FEATURES PARA MODELO
# ============================================

def get_feature_columns():
    """
    Retorna las columnas de features usadas en el modelo
    Ajustar según las features de tu notebook
    
    Returns:
        list: Lista de nombres de columnas
    """
    # TODO: Actualizar con las features exactas de tu modelo
    features = [
        'GDP per capita',
        'Infant mortality',
        'Life expectancy',
        'Urban population',
        'Female education years',
        # Agregar las demás features que usás en tu modelo
    ]
    return [f for f in features]  # Filtrar las que existen


def prepare_features_for_prediction(input_dict, scaler_path='models/scaler.pkl'):
    """
    Prepara features de input del usuario para predicción
    
    Args:
        input_dict (dict): Diccionario con valores de input
        scaler_path (str): Ruta al scaler guardado
        
    Returns:
        np.array: Features escaladas listas para predicción
    """
    try:
        # Cargar scaler
        scaler = joblib.load(scaler_path)
        
        # Convertir input a DataFrame
        features = get_feature_columns()
        input_df = pd.DataFrame([input_dict])[features]
        
        # Escalar
        input_scaled = scaler.transform(input_df)
        
        return input_scaled
    except Exception as e:
        st.error(f"Error al preparar features: {e}")
        return None


# ============================================
# UTILIDADES
# ============================================

def get_available_years(df):
    """Retorna lista de años disponibles"""
    year_col = 'Año' if 'Año' in df.columns else 'Year'
    if year_col in df.columns:
        return sorted(df[year_col].unique().tolist())
    return []


def get_available_regions(df):
    """Retorna lista de regiones disponibles"""
    region_col = 'Región' if 'Región' in df.columns else 'Region'
    if region_col in df.columns:
        return sorted(df[region_col].unique().tolist())
    return []


def get_available_countries(df):
    """Retorna lista de países disponibles"""
    country_col = 'Pais' if 'Pais' in df.columns else 'País' if 'País' in df.columns else 'Country Name'
    if country_col in df.columns:
        return sorted(df[country_col].unique().tolist())
    return []


def get_top_countries(df, n=10, year=None):
    """
    Retorna top N países con mayor tasa de natalidad
    
    Args:
        df (pd.DataFrame): Dataset
        n (int): Número de países
        year (int): Año específico (opcional)
        
    Returns:
        pd.DataFrame: Top países
    """
    df_temp = df.copy()
    
    year_col = 'Año' if 'Año' in df.columns else 'Year'
    country_col = 'Pais' if 'Pais' in df.columns else 'País' if 'País' in df.columns else 'Country Name'
    birth_col = 'Natalidad' if 'Natalidad' in df.columns else 'Tasa de natalidad' if 'Tasa de natalidad' in df.columns else 'Birth Rate'
    region_col = 'Region' if 'Region' in df.columns else 'Región'
    
    if year and year_col in df.columns:
        df_temp = df_temp[df_temp[year_col] == year]
    
    if birth_col in df.columns and country_col in df.columns:
        cols_to_return = [country_col, birth_col]
        if region_col in df.columns:
            cols_to_return.append(region_col)
        return df_temp.nlargest(n, birth_col)[cols_to_return]
    
    return pd.DataFrame()


def calculate_global_average(df, year=None):
    """
    Calcula promedio global de natalidad
    
    Args:
        df (pd.DataFrame): Dataset
        year (int): Año específico (opcional)
        
    Returns:
        float: Promedio global
    """
    df_temp = df.copy()
    
    year_col = 'Año' if 'Año' in df.columns else 'Year'
    birth_col = 'Natalidad' if 'Natalidad' in df.columns else 'Tasa de natalidad' if 'Tasa de natalidad' in df.columns else 'Birth Rate'
    
    if year and year_col in df.columns:
        df_temp = df_temp[df_temp[year_col] == year]
    
    if birth_col in df.columns:
        return df_temp[birth_col].mean()
    
    return 0


# ============================================
# EXPORTACIÓN
# ============================================

def export_to_csv(df, filename='data_export.csv'):
    """
    Exporta DataFrame a CSV
    
    Args:
        df (pd.DataFrame): Dataset a exportar
        filename (str): Nombre del archivo
        
    Returns:
        str: CSV en formato string
    """
    return df.to_csv(index=False)


if __name__ == "__main__":
    # Pruebas locales
    print("✅ Módulo de funciones cargado correctamente")
    
    # Cargar datos de prueba
    df = load_data()
    if df is not None:
        info = get_data_info(df)
        print(f"\n📊 Dataset cargado:")
        print(f"  - Filas: {info['n_filas']}")
        print(f"  - Columnas: {info['n_columnas']}")
        print(f"  - Países: {info['paises_unicos']}")
        print(f"  - Años: {info['años_disponibles'][:5]}... hasta {info['años_disponibles'][-1]}")