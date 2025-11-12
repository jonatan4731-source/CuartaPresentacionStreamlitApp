"""
Aplicación Streamlit: Predicción de Tasas de Natalidad Global
Basada en CuartaPresentacion.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline import ejecutar_pipeline_completo, get_resumen_pipeline, cargar_datos
from src.visualizations import (
    viz_evolucion_temporal_regiones,
    viz_correlaciones_interactivas,
    viz_distribucion_continentes,
    get_available_visualizations
)

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Predicción de Natalidad Global",
    page_icon="👶",
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
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# CARGA DE DATOS (CON CACHE)
# ============================================

@st.cache_data
def cargar_datos_app():
    """Carga y procesa los datos con cache de Streamlit"""
    return ejecutar_pipeline_completo('data/raw/merged_dataset.csv')

# ============================================
# SIDEBAR: NAVEGACIÓN
# ============================================

st.sidebar.title("🧭 Navegación")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📊 Visualizaciones", "🤖 Predictor", "📁 Datos"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Proyecto:** Predicción de Natalidad Global  
**Estudiante:** Ingeniería en Sistemas  
**Dataset:** Banco Mundial (2000-2023)  
**Última actualización:** Nov 2024
""")

# ============================================
# CARGAR DATOS
# ============================================

with st.spinner("🔄 Cargando y procesando datos..."):
    df = cargar_datos_app()
    df_original = cargar_datos('data/raw/merged_dataset.csv')
    resumen = get_resumen_pipeline(df_original, df)

# ============================================
# PÁGINA: INICIO
# ============================================

if pagina == "🏠 Inicio":
    st.title("👶 Predicción de Tasas de Natalidad Global")
    st.markdown("---")
    
    # Introducción
    st.markdown("""
    ### Bienvenido al Sistema de Análisis y Predicción de Natalidad
    
    Esta aplicación utiliza **Machine Learning** para analizar y predecir las tasas de natalidad 
    a nivel global, considerando múltiples factores socioeconómicos y temporales.
    
    #### 🎯 Objetivo del Proyecto
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
        - ✅ Sin regiones geográficas agregadas
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
            if viz_actual['id'] == 'evolucion_temporal':
                chart = viz_evolucion_temporal_regiones(df)
            elif viz_actual['id'] == 'correlaciones':
                chart = viz_correlaciones_interactivas(df)
            elif viz_actual['id'] == 'distribucion':
                # Selector de año para distribución
                año_viz = st.slider(
                    "Selecciona el año:",
                    min_value=int(df['Año'].min()),
                    max_value=int(df['Año'].max()),
                    value=int(df['Año'].max())
                )
                chart = viz_distribucion_continentes(df, year=año_viz)
            
            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error al generar la visualización: {e}")
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
# PÁGINA: PREDICTOR
# ============================================

elif pagina == "🤖 Predictor":
    st.title("🤖 Predictor de Natalidad")
    st.markdown("---")
    
    st.info("🚧 **Sección en desarrollo**")
    st.markdown("""
    ### Funcionalidad Planificada
    
    En esta sección podrás:
    1. **Cargar un modelo entrenado** (Random Forest optimizado)
    2. **Ingresar valores** para variables socioeconómicas
    3. **Obtener una predicción** de tasa de natalidad
    4. **Ver interpretación** del resultado
    5. **Comparar** con promedios regionales/globales
    
    #### Para completar esta sección necesitas:
    - Exportar el modelo entrenado del notebook (`best_model.pkl`)
    - Exportar el scaler (`scaler.pkl`)
    - Definir las features exactas usadas en el modelo
    """)
    
    # Placeholder para inputs
    st.markdown("### ⚙️ Parámetros de Predicción (Preview)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("PIB per cápita (USD)", value=25000, step=1000, disabled=True)
        st.number_input("Esperanza de Vida (años)", value=75, step=1, disabled=True)
        st.number_input("Urbanización (%)", value=60, step=5, disabled=True)
    
    with col2:
        st.number_input("Acceso a Educación (%)", value=85, step=5, disabled=True)
        st.number_input("Gasto en Salud (% PIB)", value=5.0, step=0.5, disabled=True)
        st.number_input("Desempleo (%)", value=7.0, step=0.5, disabled=True)
    
    st.button("🔮 Realizar Predicción", disabled=True, help="Funcionalidad en desarrollo")

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