"""
Funciones de procesamiento y pipeline para el proyecto de natalidad
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ============================================
# CARGA DE DATOS
# ============================================

def load_data(filepath):
    """
    Carga el dataset principal
    
    Args:
        filepath (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None


# ============================================
# PREPROCESAMIENTO
# ============================================

def preprocess_data(df):
    """
    Aplica el pipeline de preprocesamiento
    
    Args:
        df (pd.DataFrame): Dataset original
        
    Returns:
        pd.DataFrame: Dataset procesado
    """
    df_clean = df.copy()
    
    # TODO: Agregar tu pipeline aquí
    # Ejemplo:
    # - Manejo de nulos
    # - Encoding de variables categóricas
    # - Feature engineering
    # - etc.
    
    return df_clean


def create_features(df):
    """
    Crea features adicionales
    
    Args:
        df (pd.DataFrame): Dataset base
        
    Returns:
        pd.DataFrame: Dataset con nuevas features
    """
    df_features = df.copy()
    
    # TODO: Agregar feature engineering
    # Ejemplo:
    # df_features['pib_per_capita_log'] = np.log1p(df_features['pib_per_capita'])
    # df_features['tasa_cambio'] = df_features.groupby('pais')['natalidad'].pct_change()
    
    return df_features


def get_feature_names():
    """
    Retorna la lista de features usadas en el modelo
    
    Returns:
        list: Lista de nombres de features
    """
    # TODO: Definir tus features
    features = [
        'pib_per_capita',
        'educacion_femenina',
        'mortalidad_infantil',
        'urbanizacion',
        # ... agregar el resto
    ]
    return features


# ============================================
# SPLIT Y ESCALADO
# ============================================

def prepare_train_test(df, target='natalidad', test_size=0.2, random_state=42):
    """
    Prepara datos para entrenamiento
    
    Args:
        df (pd.DataFrame): Dataset completo
        target (str): Variable objetivo
        test_size (float): Proporción de test
        random_state (int): Semilla aleatoria
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    features = get_feature_names()
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test=None, scaler_path='models/scaler.pkl'):
    """
    Escala features usando StandardScaler
    
    Args:
        X_train: Features de entrenamiento
        X_test: Features de test (opcional)
        scaler_path: Ruta para guardar el scaler
        
    Returns:
        tuple: X_train_scaled, X_test_scaled (si aplica), scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Guardar scaler
    joblib.dump(scaler, scaler_path)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


# ============================================
# UTILIDADES
# ============================================

def get_data_summary(df):
    """
    Retorna un resumen del dataset
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        dict: Diccionario con estadísticas
    """
    summary = {
        'n_filas': len(df),
        'n_columnas': len(df.columns),
        'columnas': df.columns.tolist(),
        'nulos': df.isnull().sum().to_dict(),
        'tipos': df.dtypes.to_dict(),
        'memoria_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    return summary


def filter_by_region(df, region):
    """
    Filtra dataset por región
    
    Args:
        df (pd.DataFrame): Dataset completo
        region (str): Nombre de la región
        
    Returns:
        pd.DataFrame: Dataset filtrado
    """
    # TODO: Ajustar según tu columna de región
    return df[df['region'] == region]


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
    # TODO: Ajustar según tu columna de año
    return df[(df['year'] >= year_start) & (df['year'] <= year_end)]


# ============================================
# EXPORTACIÓN
# ============================================

def export_processed_data(df, filepath='data/processed/data_processed.csv'):
    """
    Exporta dataset procesado
    
    Args:
        df (pd.DataFrame): Dataset a exportar
        filepath (str): Ruta de destino
    """
    df.to_csv(filepath, index=False)
    print(f"Datos exportados a: {filepath}")


if __name__ == "__main__":
    # Pruebas locales
    df = load_data('data/raw/df_con_features_temporales.csv')
    if df is not None:
        print("\nResumen del dataset:")
        summary = get_data_summary(df)
        print(f"Filas: {summary['n_filas']}")
        print(f"Columnas: {summary['n_columnas']}")