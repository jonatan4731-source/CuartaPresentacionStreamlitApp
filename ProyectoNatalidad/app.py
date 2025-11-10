import streamlit as st

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Predicción de Natalidad Global",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILOS CSS PERSONALIZADOS
# ============================================
st.markdown("""
    <style>
    /* Estilo del sidebar */
    [data-testid="stSidebar"] {
        background-color: #364152;
    }
    
    /* Títulos */
    h1 {
        color: #5a6773;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# DEFINICIÓN DE PÁGINAS
# ============================================

def page_home():
    """Página principal con información general"""
    st.title("Predicción de Tasas de Natalidad Global")
    st.markdown("---")
    
    # Introducción
    st.markdown("""
    ### Bienvenido al Sistema de Predicción de Natalidad
    
    Esta aplicación utiliza modelos de Machine Learning para predecir y analizar 
    las tasas de natalidad a nivel global, considerando múltiples factores socioeconómicos.
    """)
    
    # Columnas con información
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Visualizaciones**\n\nExplora gráficos interactivos sobre tendencias de natalidad")
    
    with col2:
        st.success("**Predictor**\n\nRealiza predicciones personalizadas con el modelo")
    
    with col3:
        st.warning("**Datos**\n\nConsulta y descarga el dataset completo")
    
    st.markdown("---")
    
    # Información del proyecto
    with st.expander("Acerca del proyecto"):
        st.markdown("""
        **Objetivo:** Predecir tasas de natalidad utilizando variables socioeconómicas
        
        **Variables principales:**
        - PIB per cápita
        - Educación femenina
        - Mortalidad infantil
        - Urbanización
        - Acceso a servicios de salud
        
        **Modelos utilizados:**
        - Random Forest Regressor
        """)
    
    # Métricas de ejemplo
    st.subheader("Estadísticas del Modelo")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", "0.89", "↑ 3%")
    with col2:
        st.metric("RMSE", "2.34", "↓ 0.5")
    with col3:
        st.metric("Países analizados", "195")
    with col4:
        st.metric("Años de datos", "2000-2023")


def page_visualizaciones():
    """Página de visualizaciones con Altair"""
    st.title("Visualizaciones Interactivas")
    st.markdown("---")
    
    st.info("**En construcción:** Aquí se integrarán tus gráficos de Altair")
    
    # Selector de visualización
    viz_option = st.selectbox(
        "Selecciona una visualización:",
        ["Tendencias Temporales", "Comparación por Región", "Correlaciones"]
    )
    
    if viz_option == "Tendencias Temporales":
        st.subheader("Evolución de Natalidad en el Tiempo")
        st.write("Aquí irá tu gráfico de líneas temporal")
        
    elif viz_option == "Comparación por Región":
        st.subheader("Comparación Regional")
        st.write("Aquí irá tu gráfico de barras/mapas")
        
    elif viz_option == "Correlaciones":
        st.subheader("Matriz de Correlaciones")
        st.write("Aquí irá tu heatmap de correlaciones")


def page_predictor():
    """Página del predictor interactivo"""
    st.title("Predictor de Natalidad")
    st.markdown("---")
    
    st.markdown("""
    ### Realiza una predicción personalizada
    Ajusta los parámetros a continuación para obtener una predicción de la tasa de natalidad.
    """)
    
    # Sidebar para inputs
    with st.sidebar:
        st.header("Parámetros de Predicción")
        
        # Inputs de ejemplo (reemplazar con tus variables reales)
        pib = st.slider("PIB per cápita (USD)", 500, 100000, 25000, 500)
        educacion = st.slider("Años de educación femenina", 0, 20, 10)
        mortalidad = st.slider("Mortalidad infantil (por 1000)", 0, 150, 30)
        urbanizacion = st.slider("% Urbanización", 0, 100, 50)
        
        predecir_btn = st.button("Realizar Predicción", type="primary")
    
    # Área de resultados
    if predecir_btn:
        st.success("Predicción realizada con éxito")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.metric(
                "Tasa de Natalidad Predicha",
                "14.7 nacimientos por 1000 hab.",
                "↓ 2.3 vs. promedio global"
            )
        
        with col2:
            st.metric("Confianza del modelo", "87%")
        
        st.info("**Interpretación:** La predicción sugiere una tasa de natalidad moderada-baja...")
    else:
        st.info("Ajusta los parámetros en el sidebar y presiona 'Realizar Predicción'")


def page_datos():
    """Página de exploración de datos"""
    st.title("Exploración de Datos")
    st.markdown("---")
    
    st.markdown("""
    ### Dataset: Tasas de Natalidad Global
    Explora el dataset completo utilizado para entrenar el modelo.
    """)
    
    # Información del dataset
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", "4,500")
    with col2:
        st.metric("Columnas", "25")
    with col3:
        st.metric("Periodo", "2000-2023")
    
    st.markdown("---")
    
    # Tabs para organizar información
    tab1, tab2, tab3 = st.tabs(["Vista Previa", "Estadísticas", "Descargar"])
    
    with tab1:
        st.subheader("Primeras filas del dataset")
        st.info("🚧 Aquí se cargará tu dataframe con `st.dataframe()`")
        
    with tab2:
        st.subheader("Estadísticas descriptivas")
        st.info("🚧 Aquí irá `df.describe()` y otros análisis")
        
    with tab3:
        st.subheader("Descargar datos")
        st.download_button(
            label="Descargar CSV",
            data="dato,ejemplo\n1,2\n3,4",  # Reemplazar con tu CSV real
            file_name="natalidad_data.csv",
            mime="text/csv"
        )


# ============================================
# NAVEGACIÓN PRINCIPAL
# ============================================

# Crear el menú de navegación en el sidebar
st.sidebar.title("Navegación")
st.sidebar.markdown("---")

# Opciones de página
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["Inicio", "Visualizaciones", "Predictor", "Datos"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Proyecto:** Predicción de Natalidad mundial 
**Autor:** Grupo 07 - Ciencia de Datos  
**Última actualización:** Nov 2024
""")

# Renderizar la página seleccionada
if page == "Inicio":
    page_home()
elif page == "Visualizaciones":
    page_visualizaciones()
elif page == "Predictor":
    page_predictor()
elif page == "Datos":
    page_datos()