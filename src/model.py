"""
Funciones del modelo de Machine Learning
Basado en CuartaPresentacion.ipynb
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb


# ============================================
# CARGA DEL MODELO
# ============================================

@st.cache_resource
def load_model(model_path='models/best_model.pkl'):
    """
    Carga el modelo entrenado con cache
    
    Args:
        model_path (str): Ruta al modelo guardado
        
    Returns:
        model: Modelo cargado
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {e}")
        return None


@st.cache_resource
def load_scaler(scaler_path='models/scaler.pkl'):
    """
    Carga el scaler con cache
    
    Args:
        scaler_path (str): Ruta al scaler guardado
        
    Returns:
        scaler: Scaler cargado
    """
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"❌ Error al cargar scaler: {e}")
        return None


# ============================================
# PREDICCIÓN
# ============================================

def predict_birth_rate(model, scaler, input_features):
    """
    Realiza predicción de tasa de natalidad
    
    Args:
        model: Modelo entrenado
        scaler: Scaler para normalización
        input_features (dict): Diccionario con features de input
        
    Returns:
        float: Predicción de tasa de natalidad
    """
    try:
        # Convertir input a DataFrame
        input_df = pd.DataFrame([input_features])
        
        # Escalar features
        input_scaled = scaler.transform(input_df)
        
        # Predecir
        prediction = model.predict(input_scaled)[0]
        
        return prediction
        
    except Exception as e:
        st.error(f"❌ Error en predicción: {e}")
        return None


def predict_with_confidence(model, scaler, input_features):
    """
    Realiza predicción con intervalo de confianza (si el modelo lo soporta)
    
    Args:
        model: Modelo entrenado
        scaler: Scaler para normalización
        input_features (dict): Diccionario con features
        
    Returns:
        tuple: (predicción, intervalo_inferior, intervalo_superior)
    """
    prediction = predict_birth_rate(model, scaler, input_features)
    
    if prediction is None:
        return None, None, None
    
    # Calcular intervalo aproximado (puedes ajustar según tu modelo)
    # Esto es un placeholder - idealmente usarías el error del modelo de validación
    error_margin = 2.5  # RMSE aproximado de tu modelo
    
    lower_bound = max(0, prediction - error_margin)
    upper_bound = prediction + error_margin
    
    return prediction, lower_bound, upper_bound


# ============================================
# EVALUACIÓN DEL MODELO
# ============================================

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en datos de test
    
    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Target de test
        
    Returns:
        dict: Diccionario con métricas
    """
    try:
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"❌ Error al evaluar modelo: {e}")
        return None


def get_feature_importance(model, feature_names):
    """
    Obtiene importancia de features (si el modelo lo soporta)
    
    Args:
        model: Modelo entrenado
        feature_names (list): Lista de nombres de features
        
    Returns:
        pd.DataFrame: DataFrame con importancias ordenadas
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            df_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return df_importance
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error al obtener importancias: {e}")
        return None


# ============================================
# INTERPRETACIÓN DE PREDICCIONES
# ============================================

def interpret_prediction(prediction, global_average):
    """
    Genera interpretación textual de la predicción
    
    Args:
        prediction (float): Valor predicho
        global_average (float): Promedio global
        
    Returns:
        str: Interpretación en texto
    """
    diff = prediction - global_average
    diff_pct = (diff / global_average) * 100
    
    if abs(diff_pct) < 5:
        level = "similar al promedio global"
    elif diff_pct > 20:
        level = "significativamente superior al promedio global"
    elif diff_pct > 10:
        level = "superior al promedio global"
    elif diff_pct < -20:
        level = "significativamente inferior al promedio global"
    elif diff_pct < -10:
        level = "inferior al promedio global"
    else:
        level = "cercana al promedio global"
    
    interpretation = f"""
    La tasa de natalidad predicha es de **{prediction:.2f} nacimientos por 1000 habitantes**.
    
    Esto es **{level}** ({diff:+.2f} puntos, {diff_pct:+.1f}%).
    
    """
    
    # Agregar contexto según el valor
    if prediction > 30:
        interpretation += "Esta es una tasa de natalidad muy alta, típica de países en desarrollo con acceso limitado a educación y planificación familiar."
    elif prediction > 20:
        interpretation += "Esta es una tasa de natalidad alta, común en economías emergentes con crecimiento poblacional rápido."
    elif prediction > 15:
        interpretation += "Esta es una tasa de natalidad moderada, típica de países de ingresos medios en transición demográfica."
    elif prediction > 10:
        interpretation += "Esta es una tasa de natalidad baja, característica de países desarrollados con población estable o en declive."
    else:
        interpretation += "Esta es una tasa de natalidad muy baja, típica de economías avanzadas con poblaciones envejecidas."
    
    return interpretation


def get_prediction_category(prediction):
    """
    Categoriza la predicción
    
    Args:
        prediction (float): Valor predicho
        
    Returns:
        str: Categoría (Muy Baja, Baja, Moderada, Alta, Muy Alta)
    """
    if prediction < 10:
        return "Muy Baja"
    elif prediction < 15:
        return "Baja"
    elif prediction < 20:
        return "Moderada"
    elif prediction < 30:
        return "Alta"
    else:
        return "Muy Alta"


# ============================================
# COMPARACIONES
# ============================================

def compare_with_similar_countries(prediction, df, input_features, n=5):
    """
    Encuentra países similares según features de input
    
    Args:
        prediction (float): Predicción realizada
        df (pd.DataFrame): Dataset completo
        input_features (dict): Features del input
        n (int): Número de países similares a mostrar
        
    Returns:
        pd.DataFrame: Países similares con sus tasas de natalidad
    """
    # TODO: Implementar lógica de similitud
    # Esto requeriría calcular distancias euclidianas o similares
    # Por ahora retornamos un placeholder
    
    return pd.DataFrame({
        'País': ['Placeholder 1', 'Placeholder 2'],
        'Tasa de Natalidad': [prediction - 1, prediction + 1],
        'Similitud': [0.95, 0.92]
    })


# ============================================
# SIMULACIONES
# ============================================

def simulate_scenarios(model, scaler, base_features, variable_to_change, values):
    """
    Simula diferentes escenarios cambiando una variable
    
    Args:
        model: Modelo entrenado
        scaler: Scaler
        base_features (dict): Features base
        variable_to_change (str): Variable a modificar
        values (list): Lista de valores a probar
        
    Returns:
        pd.DataFrame: Resultados de la simulación
    """
    results = []
    
    for value in values:
        # Copiar features base
        scenario_features = base_features.copy()
        scenario_features[variable_to_change] = value
        
        # Predecir
        prediction = predict_birth_rate(model, scaler, scenario_features)
        
        if prediction is not None:
            results.append({
                variable_to_change: value,
                'Predicción': prediction
            })
    
    return pd.DataFrame(results)


# ============================================
# GUARDADO DE MODELOS
# ============================================

def save_model(model, filepath='models/model.pkl'):
    """
    Guarda el modelo entrenado
    
    Args:
        model: Modelo a guardar
        filepath (str): Ruta de destino
    """
    try:
        joblib.dump(model, filepath)
        print(f"✅ Modelo guardado en: {filepath}")
    except Exception as e:
        print(f"❌ Error al guardar modelo: {e}")


def save_scaler(scaler, filepath='models/scaler.pkl'):
    """
    Guarda el scaler
    
    Args:
        scaler: Scaler a guardar
        filepath (str): Ruta de destino
    """
    try:
        joblib.dump(scaler, filepath)
        print(f"✅ Scaler guardado en: {filepath}")
    except Exception as e:
        print(f"❌ Error al guardar scaler: {e}")


if __name__ == "__main__":
    print(" Módulo de modelo cargado correctamente")