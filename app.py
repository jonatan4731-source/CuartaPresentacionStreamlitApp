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
    predict_birth_rate,
    interpret_prediction,
    get_prediction_category
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
**Grupo: Grupo 07
""")

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
# PÁGINA: PREDICTOR
# ============================================

elif pagina == "🤖 Predictor":
    st.title("🤖 Predictor de Natalidad")
    st.markdown("---")
    
    # Verificar si existen los modelos
    modelo_existe = os.path.exists('models/best_model.pkl')
    scaler_existe = os.path.exists('models/scaler.pkl')
    
    if not modelo_existe or not scaler_existe:
        st.warning("⚠️ **Modelos no encontrados**")
        st.markdown("""
        ### 🔧 Configuración Necesaria
        
        Para usar el predictor, necesitas:
        
        1. **Entrenar el modelo** ejecutando el notebook `CuartaPresentacion.ipynb`
        2. **Exportar el modelo** con este código al final del notebook:
        
        ```python
        import joblib
        
        # Guardar modelo
        joblib.dump(best_rf, 'models/best_model.pkl')
        
        # Guardar scaler (del pipeline)
        joblib.dump(pipeline_info['etapa7_preprocesamiento']['scaler'], 'models/scaler.pkl')
        
        # Guardar imputer
        joblib.dump(pipeline_info['etapa7_preprocesamiento']['imputer'], 'models/imputer.pkl')
        
        print("✅ Modelos guardados!")
        ```
        
        3. **Copiar los archivos** a la carpeta `models/` de esta app
        """)
        
        st.info("📝 **Tip:** Crea la carpeta `models/` si no existe en la raíz del proyecto")
        
    else:
        # Cargar modelo y scaler
        with st.spinner("Cargando modelo..."):
            model = load_model()
            scaler = load_scaler()
        
        if model is None or scaler is None:
            st.error("❌ Error al cargar el modelo o scaler")
            st.stop()
        
        st.success("✅ Modelo cargado correctamente")
        
        st.markdown("### 🎛️ Parámetros de Predicción")
        st.markdown("Ingresa los valores de las variables socioeconómicas:")
        
        # Crear formulario de inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Variables Económicas")
            pib_per_capita = st.number_input(
                "PIB per cápita (USD)", 
                min_value=0, 
                max_value=200000, 
                value=25000, 
                step=1000,
                help="Producto Interno Bruto dividido por la población"
            )
            
            ingreso_medio = st.number_input(
                "Ingreso Medio (USD)", 
                min_value=0, 
                max_value=200000, 
                value=20000, 
                step=1000
            )
            
            desempleo = st.slider(
                "Desempleo (%)", 
                min_value=0.0, 
                max_value=30.0, 
                value=7.0, 
                step=0.5
            )
            
            urbanizacion = st.slider(
                "Urbanización (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=60.0, 
                step=5.0
            )
            
            st.markdown("#### 👩‍🎓 Variables de Educación")
            acceso_educacion = st.slider(
                "Acceso a Educación (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=85.0, 
                step=5.0
            )
            
            matricula_primaria = st.slider(
                "Matrícula Primaria (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=90.0, 
                step=5.0
            )
        
        with col2:
            st.markdown("#### 🏥 Variables de Salud")
            esperanza_vida = st.number_input(
                "Esperanza de Vida (años)", 
                min_value=40, 
                max_value=90, 
                value=75, 
                step=1
            )
            
            gasto_salud = st.slider(
                "Gasto en Salud (% PIB)", 
                min_value=0.0, 
                max_value=20.0, 
                value=5.0, 
                step=0.5
            )
            
            acceso_agua = st.slider(
                "Acceso a Agua Potable (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=85.0, 
                step=5.0
            )
            
            st.markdown("#### 👥 Variables de Género")
            participacion_laboral_fem = st.slider(
                "Participación Laboral Femenina (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=50.0, 
                step=5.0
            )
            
            mujeres_parlamento = st.slider(
                "Mujeres en Parlamento (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=25.0, 
                step=5.0
            )
        
        st.markdown("---")
        
        # Botón de predicción
        if st.button("🔮 Realizar Predicción", type="primary", use_container_width=True):
            with st.spinner("Calculando predicción..."):
                # Preparar input (AJUSTAR según las features exactas de tu modelo)
                input_features = {
                    'PIB_per_capita': pib_per_capita,
                    'IngresoMedio': ingreso_medio,
                    'Desempleo': desempleo,
                    'Urbanizacion': urbanizacion,
                    'AccesoEducacion': acceso_educacion,
                    'MatriculacionPrimaria': matricula_primaria,
                    'EsperanzaVida': esperanza_vida,
                    'GastoSalud': gasto_salud,
                    'AccesoAguaPotable': acceso_agua,
                    'TasaParticipacionLaboralFemenina': participacion_laboral_fem,
                    'MujeresParlamento': mujeres_parlamento,
                    # Agregar features temporales por defecto
                    'AñosDesde2000': 24,  # 2024
                    'Decada': 2020,
                    'CrisisEconomica2008': 0,
                    'PandemiaCOVID': 0
                }
                
                # Realizar predicción
                prediction = predict_birth_rate(model, scaler, input_features)
                
                if prediction is not None:
                    st.markdown("---")
                    st.markdown("### 📊 Resultado de la Predicción")
                    
                    # Mostrar predicción principal
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Tasa de Natalidad Predicha",
                            f"{prediction:.2f}",
                            help="Nacimientos por 1000 habitantes"
                        )
                    
                    with col2:
                        categoria = get_prediction_category(prediction)
                        st.metric(
                            "Categoría",
                            categoria
                        )
                    
                    with col3:
                        # Calcular promedio global del último año
                        promedio_global = df[df['Año'] == df['Año'].max()]['Natalidad'].mean()
                        diferencia = prediction - promedio_global
                        st.metric(
                            "vs Promedio Global",
                            f"{diferencia:+.2f}",
                            f"{(diferencia/promedio_global)*100:+.1f}%"
                        )
                    
                    # Interpretación
                    st.markdown("### 💬 Interpretación")
                    interpretacion = interpret_prediction(prediction, promedio_global)
                    st.markdown(interpretacion)
                else:
                    st.error("❌ Error al realizar la predicción")

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