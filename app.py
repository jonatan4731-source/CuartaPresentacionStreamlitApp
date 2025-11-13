"""
Aplicación Streamlit: Predicción de Tasas de Natalidad Global
Basada en CuartaPresentacion.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# Importaciones corregidas
from src.pipeline import ejecutar_pipeline_completo, get_resumen_pipeline, cargar_datos
from src.visualizations import (
    viz_evolucion_temporal_regiones,
    viz_correlaciones_interactivas,
    viz_mapa_mundial_natalidad,
    viz_distribucion_continentes,
    get_available_visualizations
)
from src.model import (
    load_model, 
    load_scaler,
    load_imputer,
    predict_birth_rate,
    predict_batch,
    interpret_prediction,
    get_prediction_category,
    evaluate_model
)

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Predicción de Natalidad Global",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILOS CSS
# ============================================
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #364152;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# CARGA DE DATOS (CON CACHE)
# ============================================

@st.cache_data
def cargar_datos_app():
    """Carga y procesa los datos con cache de Streamlit"""
    # IMPORTANTE: Ruta al CSV ORIGINAL (merged_dataset.csv)
    ruta = 'data/raw/merged_dataset.csv'
    
    # Verificar si existe
    if not os.path.exists(ruta):
        st.error(f"❌ No se encontró el archivo: {ruta}")
        st.info("💡 Asegúrate de tener el archivo merged_dataset.csv en la carpeta data/raw/")
        return None
    
    # Ejecutar pipeline completo (limpieza + features + regiones)
    df_procesado = ejecutar_pipeline_completo(ruta, umbral_faltantes=60)
    
    return df_procesado


# ============================================
# SIDEBAR: NAVEGACIÓN
# ============================================

st.sidebar.title("Navegación")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📊 Visualizaciones", "🧠 Predictor", "📁 Datos"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
"""
**📌 Predicción de Natalidad Global**  

**📌 Ingeniería en Sistemas**

**📌 Ciencia de Datos**

**📌 Dataset Banco Mundial (2000-2023)** 

**📌 Última actualización:** Nov 2024
"""
)

# ============================================
# CARGAR DATOS
# ============================================

with st.spinner("🔄 Cargando y procesando datos..."):
    df = cargar_datos_app()
    
    if df is None:
        st.stop()  # Detener ejecución si no hay datos
    
    # Cargar dataset original solo para el resumen
    df_original = cargar_datos('data/raw/merged_dataset.csv')
    resumen = get_resumen_pipeline(df_original, df)

# ============================================
# PÁGINA: INICIO
# ============================================

if pagina == "🏠 Inicio":
    st.title("Predicción de Tasas de Natalidad Global")
    st.markdown("---")
    
    # Introducción
    st.markdown("""
    ### Bienvenido al Sistema de Análisis y Predicción de Natalidad
    
    Esta aplicación utiliza **Machine Learning** para analizar y predecir las tasas de natalidad 
    a nivel global, considerando múltiples factores socioeconómicos y temporales.
    
    #### Objetivo del Proyecto
    Comprender los factores que influyen en las tasas de natalidad y crear modelos predictivos 
    que ayuden a entender tendencias demográficas globales.
    """)
    
    # Métricas principales
    st.markdown("### 📊 Estadísticas del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Países Analizados",
            f"{resumen['paises_unicos']}",
            help="Número de países con datos completos"
        )
    
    with col2:
        st.metric(
            "Regiones Geográficas",
            f"{resumen['regiones_unicas']}",
            help="Divisiones geográficas para análisis regional"
        )
    
    with col3:
        st.metric(
            "Años de Datos",
            f"{resumen['años_max'] - resumen['años_min'] + 1}",
            f"{resumen['años_min']}-{resumen['años_max']}"
        )
    
    with col4:
        st.metric(
            "Variables Analizadas",
            f"{resumen['columnas_procesado']}",
            help="Features después del procesamiento"
        )
    
    st.markdown("---")
    
    # Información del pipeline
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 Pipeline de Procesamiento")
        st.markdown("""
        1. **Carga de Datos** - Dataset original del Banco Mundial
        2. **Limpieza** - Eliminación de nulos y duplicados
        3. **Eliminación de Leakage** - Variables que generan sesgo
        4. **Feature Engineering** - Creación de features temporales
        5. **Asignación Geográfica** - Continentes y regiones
        6. **Preparación para ML** - Escalado e imputación
        """)
    
    with col2:
        st.markdown("#### 📈 Variables Principales")
        st.markdown("""
        - **Socioeconómicas:** PIB per cápita, Ingreso medio, Desempleo
        - **Educación:** Acceso a educación, Matrícula escolar
        - **Salud:** Esperanza de vida, Acceso a salud, Vacunación
        - **Demografía:** Urbanización, Densidad poblacional
        - **Género:** Participación laboral femenina, Educación femenina
        - **Temporales:** Crisis 2008, Pandemia COVID-19
        """)
    
    st.markdown("---")
    
    # Continentes disponibles
    st.markdown("### 🌍 Continentes en el Dataset")
    continentes_cols = st.columns(len(resumen['continentes']))
    
    for idx, continente in enumerate(resumen['continentes']):
        with continentes_cols[idx]:
            n_paises = df[df['Continente'] == continente]['Pais'].nunique()
            st.info(f"**{continente}**\n\n{n_paises} países")
    
    st.markdown("---")
    
    # Resumen de transformaciones
    with st.expander("ℹ️ Ver detalles del procesamiento de datos"):
        st.markdown(f"""
        **Dataset Original:**
        - Filas: {resumen['filas_original']:,}
        - Columnas: {resumen['columnas_original']}
        
        **Dataset Procesado:**
        - Filas: {resumen['filas_procesado']:,} ({resumen['filas_procesado']/resumen['filas_original']*100:.1f}% conservado)
        - Columnas: {resumen['columnas_procesado']} (eliminadas {resumen['columnas_original'] - resumen['columnas_procesado']} por leakage/nulos)
        
        **Calidad de Datos:**
        - ✅ Sin duplicados
        - ✅ Variables con leakage eliminadas
        - ✅ Features temporales creadas
        - ✅ Regiones geográficas asignadas
        """)

# ============================================
# PÁGINA: VISUALIZACIONES
# ============================================

elif pagina == "📊 Visualizaciones":
    st.title("📊 Visualizaciones Interactivas")
    st.markdown("---")
    
    # Selector de visualización
    vizs = get_available_visualizations()
    
    viz_seleccionada = st.selectbox(
        "Selecciona una visualización:",
        options=[viz['nombre'] for viz in vizs],
        format_func=lambda x: f"📈 {x}"
    )
    
    # Encontrar la viz seleccionada
    viz_actual = next(viz for viz in vizs if viz['nombre'] == viz_seleccionada)
    
    # Mostrar descripción
    st.info(f"**{viz_actual['descripcion']}**")
    
    st.markdown("---")
    
    # Generar y mostrar visualización
    with st.spinner("🎨 Generando visualización..."):
        try:
            chart = None
            
            if viz_actual['id'] == 'evolucion_temporal':
                chart = viz_evolucion_temporal_regiones(df)
            elif viz_actual['id'] == 'correlaciones':
                chart = viz_correlaciones_interactivas(df)
            elif viz_actual['id'] == 'mapa_mundial':
                chart = viz_mapa_mundial_natalidad(df)
            elif viz_actual['id'] == 'distribucion':
                chart = viz_distribucion_continentes(df)
            
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.error("❌ No se pudo generar el gráfico")
            
        except Exception as e:
            st.error(f"❌ Error al generar la visualización: {e}")
            with st.expander("Ver detalles del error"):
                st.exception(e)
    
    # Tips de interacción
    with st.expander("💡 Tips de interacción"):
        st.markdown("""
        - **Zoom:** Rueda del mouse sobre el gráfico
        - **Pan:** Click y arrastra
        - **Tooltip:** Pasa el mouse sobre los elementos
        - **Filtros:** Usa los selectores interactivos
        - **Reset:** Doble click en el gráfico
        """)

# ============================================
# PÁGINA: PREDICTOR INTERACTIVO
# ============================================

elif pagina == "🧠 Predictor":
    st.title("🧠 Predictor de Natalidad")
    st.markdown("---")
    
    # Verificar si existen los modelos
    modelo_existe = os.path.exists('models/best_model.pkl')
    scaler_existe = os.path.exists('models/scaler.pkl')
    imputer_existe = os.path.exists('models/imputer.pkl')
    
    if not modelo_existe or not scaler_existe or not imputer_existe:
        st.warning("⚠️ **Modelos no encontrados**")
        st.markdown("""
        ### 🔧 Configuración Necesaria
        
        Para usar el predictor, necesitas:
        
        1. **Entrenar el modelo** ejecutando el notebook `CuartaPresentacion.ipynb`
        2. **Exportar el modelo** con el código proporcionado en las instrucciones
        3. **Copiar los archivos** a la carpeta `models/`:
           - `best_model.pkl`
           - `scaler.pkl`
           - `imputer.pkl`
        """)
        st.stop()
    
    # ============================================
    # CARGAR MODELO Y PREPARAR DATOS
    # ============================================
    
    with st.spinner("🔄 Cargando modelo y preparando datos..."):
        # Cargar modelo, scaler e imputer
        model = load_model()
        scaler = load_scaler()
        
        try:
            import joblib
            imputer = joblib.load('models/imputer.pkl')
        except:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            st.warning("⚠️ Imputer no encontrado, usando uno por defecto")
        
        # Preparar datos para el modelo
        from src.pipeline import preparar_para_modelo
        
        X_train, X_test, y_train, y_test, scaler_prep, feature_names, imputer_prep = preparar_para_modelo(
            df, 
            año_corte=2021,  # Años <= 2021 para train, >2021 para test
            random_state=42
        )
        
        # Obtener los índices originales para mapear países
        train_mask = df['Año'] <= 2021
        test_mask = df['Año'] > 2021
        
        df_train_original = df[train_mask].copy()
        df_test_original = df[test_mask].copy()
    
    st.success("Modelo cargado correctamente")
    
    # ============================================
    # TABS DE NAVEGACIÓN
    # ============================================
    
    tab1, tab2 = st.tabs(["📊 Evaluación por País", "📈 Métricas Generales"])
    
    # ============================================
    # TAB 1: EVALUACIÓN POR PAÍS
    # ============================================
    
    with tab1:
        st.markdown("### 🌍 Selecciona un País para Evaluar")
        st.markdown("Visualiza cómo el modelo predice la natalidad comparado con los datos reales.")
        
        # Selector de país
        paises_disponibles = sorted(df['Pais'].unique())
        pais_seleccionado = st.selectbox(
            "Selecciona un país:",
            options=paises_disponibles,
            index=paises_disponibles.index('Argentina') if 'Argentina' in paises_disponibles else 0
        )
        
        # Filtrar datos del país seleccionado
        df_pais = df[df['Pais'] == pais_seleccionado].sort_values('Año').copy()
        
        if len(df_pais) == 0:
            st.error(f"No hay datos disponibles para {pais_seleccionado}")
            st.stop()
        
        st.markdown("---")
        
        # ============================================
        # REALIZAR PREDICCIONES PARA EL PAÍS
        # ============================================
        
        with st.spinner(f"🔮 Realizando predicciones para {pais_seleccionado}..."):
            # Separar datos del país en train y test
            df_pais_train = df_pais[df_pais['Año'] <= 2021].copy()
            df_pais_test = df_pais[df_pais['Año'] > 2021].copy()
            
            predicciones_test = []
            años_test = []
            
            # Columnas a excluir para features
            columnas_excluir = ['Natalidad', 'Año', 'Pais', 'CodigoPais', 'Continente', 'Region']
            columnas_excluir_existentes = [col for col in columnas_excluir if col in df_pais_test.columns]
            
            # Hacer predicciones para cada año de test
            for idx, row in df_pais_test.iterrows():
                X_row = row.drop(labels=columnas_excluir_existentes)
                X_row_df = pd.DataFrame([X_row])
                
                # Imputar y escalar
                try:
                    X_row_imputed = imputer.transform(X_row_df)
                    X_row_scaled = scaler.transform(X_row_imputed)
                    
                    # Predecir
                    pred = model.predict(X_row_scaled)[0]
                    predicciones_test.append(pred)
                    años_test.append(row['Año'])
                except Exception as e:
                    st.error(f"Error al predecir para año {row['Año']}: {e}")
                    predicciones_test.append(None)
                    años_test.append(row['Año'])
        
        # ============================================
        # GRÁFICO: REAL VS PREDICHO
        # ============================================
        
        st.markdown("### 📈 Evolución Temporal: Real vs Predicho")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Línea de datos reales (toda la serie)
        fig.add_trace(go.Scatter(
            x=df_pais['Año'],
            y=df_pais['Natalidad'],
            mode='lines+markers',
            name='Datos Reales',
            line=dict(color='steelblue', width=3),
            marker=dict(size=8)
        ))
        
        # Línea de predicciones (solo años test)
        if len(predicciones_test) > 0:
            fig.add_trace(go.Scatter(
                x=años_test,
                y=predicciones_test,
                mode='lines+markers',
                name='Predicciones del Modelo',
                line=dict(color='orange', width=3, dash='dash'),
                marker=dict(size=10, symbol='diamond')
            ))
            
            # Línea vertical separando train/test
            año_corte = 2021
            fig.add_vline(
                x=año_corte,
                line_dash="dot",
                line_color="red",
                annotation_text="Inicio Test",
                annotation_position="top"
            )
        
        fig.update_layout(
            title=f"Tasa de Natalidad: {pais_seleccionado}",
            xaxis_title="Año",
            yaxis_title="Natalidad (nacimientos por 1000 hab)",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ============================================
        # MÉTRICAS DEL PAÍS
        # ============================================
        
        if len(predicciones_test) > 0 and len(df_pais_test) > 0:
            st.markdown("### 📊 Métricas de Predicción")
            
            # Calcular métricas
            y_real_pais = df_pais_test['Natalidad'].values
            y_pred_pais = np.array(predicciones_test)
            
            # Filtrar NaN si hay
            mask = ~np.isnan(y_pred_pais) & ~np.isnan(y_real_pais)
            y_real_pais_clean = y_real_pais[mask]
            y_pred_pais_clean = y_pred_pais[mask]
            
            if len(y_real_pais_clean) > 0:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                rmse_pais = np.sqrt(mean_squared_error(y_real_pais_clean, y_pred_pais_clean))
                mae_pais = mean_absolute_error(y_real_pais_clean, y_pred_pais_clean)
                r2_pais = r2_score(y_real_pais_clean, y_pred_pais_clean)
                mape_pais = np.mean(np.abs((y_real_pais_clean - y_pred_pais_clean) / y_real_pais_clean)) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("RMSE", f"{rmse_pais:.2f}", help="Root Mean Squared Error")
                
                with col2:
                    st.metric("MAE", f"{mae_pais:.2f}", help="Mean Absolute Error")
                
                with col3:
                    st.metric("R² Score", f"{r2_pais:.3f}", help="Coeficiente de Determinación")
                
                with col4:
                    st.metric("MAPE", f"{mape_pais:.1f}%", help="Mean Absolute Percentage Error")
                
                # Interpretación
                st.markdown("---")
                st.markdown("#### 💬 Interpretación")
                
                if r2_pais > 0.9:
                    st.success(f"**Excelente ajuste** - El modelo predice muy bien para {pais_seleccionado} (R² > 0.9)")
                elif r2_pais > 0.7:
                    st.info(f"**Buen ajuste** - El modelo predice razonablemente bien para {pais_seleccionado} (R² > 0.7)")
                elif r2_pais > 0.5:
                    st.warning(f"**Ajuste moderado** - Las predicciones tienen margen de mejora (R² > 0.5)")
                else:
                    st.error(f"**Ajuste débil** - El modelo tiene dificultades con {pais_seleccionado} (R² < 0.5)")
                
                st.markdown(f"""
                - **Error promedio:** {mae_pais:.2f} nacimientos por 1000 habitantes
                - **Error porcentual:** {mape_pais:.1f}% de desviación en promedio
                - **Varianza explicada:** {r2_pais*100:.1f}% de la variabilidad es capturada por el modelo
                """)
                
                # ============================================
                # GRÁFICO: SCATTER REAL VS PREDICHO
                # ============================================
                
                st.markdown("---")
                st.markdown("### 🎯 Precisión de las Predicciones")
                
                fig_scatter = go.Figure()
                
                # Scatter plot
                fig_scatter.add_trace(go.Scatter(
                    x=y_real_pais_clean,
                    y=y_pred_pais_clean,
                    mode='markers',
                    marker=dict(size=12, color='steelblue', opacity=0.6),
                    text=[f"Año: {año}" for año in años_test],
                    hovertemplate='<b>Real:</b> %{x:.2f}<br><b>Predicho:</b> %{y:.2f}<br>%{text}<extra></extra>'
                ))
                
                # Línea de predicción perfecta
                min_val = min(y_real_pais_clean.min(), y_pred_pais_clean.min())
                max_val = max(y_real_pais_clean.max(), y_pred_pais_clean.max())
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Predicción Perfecta',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_scatter.update_layout(
                    title=f"Real vs Predicho: {pais_seleccionado}",
                    xaxis_title="Natalidad Real",
                    yaxis_title="Natalidad Predicha",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
        else:
            st.info("ℹ️ No hay datos de test disponibles para este país (todos los datos son de entrenamiento)")
    
    # ============================================
    # TAB 2: MÉTRICAS GENERALES
    # ============================================
    
    with tab2:
        st.markdown("### 📊 Rendimiento General del Modelo")
        st.markdown("Evaluación del modelo en todo el conjunto de prueba")
        
        with st.spinner("Calculando métricas generales..."):
            # Predecir todo el conjunto de test
            y_pred_test = model.predict(X_test)
            
            # Calcular métricas generales
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            rmse_general = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_general = mean_absolute_error(y_test, y_pred_test)
            r2_general = r2_score(y_test, y_pred_test)
            mape_general = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        # Mostrar métricas
        st.markdown("#### 🎯 Métricas del Conjunto de Test")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "RMSE",
                f"{rmse_general:.2f}",
                help="Error cuadrático medio - Penaliza más los errores grandes"
            )
        
        with col2:
            st.metric(
                "MAE",
                f"{mae_general:.2f}",
                help="Error absoluto medio - Promedio de desviación"
            )
        
        with col3:
            st.metric(
                "R² Score",
                f"{r2_general:.4f}",
                help="Proporción de varianza explicada (0-1, mayor es mejor)"
            )
        
        with col4:
            st.metric(
                "MAPE",
                f"{mape_general:.1f}%",
                help="Error porcentual absoluto medio"
            )
        
        st.markdown("---")
        
        # ============================================
        # GRÁFICO: DISTRIBUCIÓN DE ERRORES
        # ============================================
        
        st.markdown("#### 📉 Distribución de Errores")
        
        residuos = y_test - y_pred_test
        
        fig_residuos = go.Figure()
        
        fig_residuos.add_trace(go.Histogram(
            x=residuos,
            nbinsx=50,
            marker_color='steelblue',
            opacity=0.7,
            name='Residuos'
        ))
        
        fig_residuos.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            annotation_text="Error = 0"
        )
        
        fig_residuos.update_layout(
            title="Distribución de Residuos (Real - Predicho)",
            xaxis_title="Residuo",
            yaxis_title="Frecuencia",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_residuos, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Estadísticas de Residuos:**
            - Media: {residuos.mean():.3f}
            - Mediana: {np.median(residuos):.3f}
            - Desviación Estándar: {residuos.std():.3f}
            """)
        
        with col2:
            # Calcular porcentaje de predicciones dentro de ciertos rangos
            dentro_1 = (np.abs(residuos) <= 1).sum() / len(residuos) * 100
            dentro_2 = (np.abs(residuos) <= 2).sum() / len(residuos) * 100
            dentro_3 = (np.abs(residuos) <= 3).sum() / len(residuos) * 100
            
            st.success(f"""
            **Precisión por Rango:**
            - {dentro_1:.1f}% predicciones con error < 1
            - {dentro_2:.1f}% predicciones con error < 2
            - {dentro_3:.1f}% predicciones con error < 3
            """)
        
        st.markdown("---")
        
        # ============================================
        # GRÁFICO: REAL VS PREDICHO (SCATTER GENERAL)
        # ============================================
        
        st.markdown("#### 🎯 Real vs Predicho (Todo el Test Set)")
        
        # Muestrear si hay muchos puntos
        n_points = len(y_test)
        if n_points > 1000:
            indices = np.random.choice(n_points, 1000, replace=False)
            y_test_sample = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]
            y_pred_sample = y_pred_test[indices]
        else:
            y_test_sample = y_test
            y_pred_sample = y_pred_test
        
        fig_scatter_general = go.Figure()
        
        fig_scatter_general.add_trace(go.Scatter(
            x=y_test_sample,
            y=y_pred_sample,
            mode='markers',
            marker=dict(
                size=6,
                color=np.abs(y_test_sample - y_pred_sample),
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Error Abs"),
                opacity=0.6
            ),
            hovertemplate='<b>Real:</b> %{x:.2f}<br><b>Predicho:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Línea de predicción perfecta
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        fig_scatter_general.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Predicción Perfecta',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig_scatter_general.update_layout(
            title=f"Real vs Predicho - R² = {r2_general:.4f}",
            xaxis_title="Natalidad Real",
            yaxis_title="Natalidad Predicha",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig_scatter_general, use_container_width=True)
        
        # ============================================
        # TOP/BOTTOM PAÍSES POR ERROR
        # ============================================
        
        st.markdown("---")
        st.markdown("#### Países con Mejor y Peor Predicción")
        
        # Calcular errores por país
        df_test_with_pred = df_test_original.copy()
        df_test_with_pred['Prediccion'] = y_pred_test
        df_test_with_pred['Error_Abs'] = np.abs(df_test_with_pred['Natalidad'] - df_test_with_pred['Prediccion'])
        
        # Agrupar por país
        errores_por_pais = df_test_with_pred.groupby('Pais').agg({
            'Error_Abs': 'mean',
            'Natalidad': 'mean'
        }).reset_index()
        errores_por_pais.columns = ['Pais', 'Error_Promedio', 'Natalidad_Promedio']
        errores_por_pais = errores_por_pais.sort_values('Error_Promedio')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Mejores Predicciones:")
            top_5_mejores = errores_por_pais.head(10)
            st.dataframe(
                top_5_mejores[['Pais', 'Error_Promedio', 'Natalidad_Promedio']].style.format({
                    'Error_Promedio': '{:.2f}',
                    'Natalidad_Promedio': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.markdown("##### Peores Predicciones:")
            top_5_peores = errores_por_pais.tail(10)
            st.dataframe(
                top_5_peores[['Pais', 'Error_Promedio', 'Natalidad_Promedio']].style.format({
                    'Error_Promedio': '{:.2f}',
                    'Natalidad_Promedio': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )

# ============================================
# PÁGINA: DATOS
# ============================================

elif pagina == "📁 Datos":
    st.title("📁 Exploración de Datos")
    st.markdown("---")
    
    # Tabs para organizar
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Vista Previa", "📊 Estadísticas", "🔍 Filtros", "⬇️ Descargar"])
    
    with tab1:
        st.subheader("Primeras filas del dataset procesado")
        
        # Selector de número de filas
        n_rows = st.slider("Número de filas a mostrar:", 5, 100, 10)
        
        st.dataframe(
            df.head(n_rows),
            use_container_width=True,
            height=400
        )
        
        st.markdown(f"**Total de filas:** {len(df):,} | **Columnas:** {len(df.columns)}")
    
    with tab2:
        st.subheader("Estadísticas Descriptivas")
        
        # Selector de columnas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        columnas_seleccionadas = st.multiselect(
            "Selecciona columnas:",
            options=columnas_numericas,
            default=columnas_numericas[:5] if len(columnas_numericas) >= 5 else columnas_numericas
        )
        
        if columnas_seleccionadas:
            st.dataframe(
                df[columnas_seleccionadas].describe(),
                use_container_width=True
            )
        else:
            st.warning("⚠️ Selecciona al menos una columna")
    
    with tab3:
        st.subheader("Filtrar Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filtro por continente
            continentes = ['Todos'] + sorted(df['Continente'].unique().tolist())
            continente_filtro = st.selectbox("Continente:", continentes)
            
            # Filtro por año
            años = sorted(df['Año'].unique().tolist())
            año_filtro = st.select_slider("Año:", options=años, value=(años[0], años[-1]))
        
        with col2:
            # Filtro por región
            if continente_filtro != 'Todos':
                regiones = ['Todas'] + sorted(df[df['Continente'] == continente_filtro]['Region'].unique().tolist())
            else:
                regiones = ['Todas'] + sorted(df['Region'].unique().tolist())
            
            region_filtro = st.selectbox("Región:", regiones)
        
        # Aplicar filtros
        df_filtrado = df.copy()
        
        if continente_filtro != 'Todos':
            df_filtrado = df_filtrado[df_filtrado['Continente'] == continente_filtro]
        
        if region_filtro != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Region'] == region_filtro]
        
        df_filtrado = df_filtrado[
            (df_filtrado['Año'] >= año_filtro[0]) & 
            (df_filtrado['Año'] <= año_filtro[1])
        ]
        
        st.markdown(f"**Resultados:** {len(df_filtrado):,} filas")
        
        st.dataframe(df_filtrado, use_container_width=True, height=400)
    
    with tab4:
        st.subheader("Descargar Datos")
        
        st.markdown("""
        Descarga el dataset procesado en formato CSV para análisis externos.
        """)
        
        # Botón de descarga
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Descargar CSV Completo",
            data=csv,
            file_name=f"natalidad_procesado_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Descarga el dataset completo procesado"
        )
        
        st.info(f"📊 El archivo contendrá {len(df):,} filas y {len(df.columns)} columnas")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Predicción de Tasas de Natalidad Global</strong></p>
    <p>Proyecto de Ingeniería en Sistemas | Datos: Banco Mundial | Tecnología: Python + Streamlit</p>
</div>
""", unsafe_allow_html=True)