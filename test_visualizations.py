"""
Script de prueba para src/visualizations.py
Ejecutar desde la raíz del proyecto: python test_visualizations.py
"""

import sys
sys.path.append('.')

from src.functions import load_data
from src.visualizations import *
import altair as alt
import os

print("=" * 60)
print("📊 PRUEBA DE VISUALIZACIONES")
print("=" * 60)

# ============================================
# Cargar datos
# ============================================
print("\n📂 Cargando datos...")
df = load_data('data/raw/merged_dataset.csv', add_geography=True)

if df is None:
    print("❌ No se pudo cargar el dataset")
    sys.exit(1)

print(f"✅ Datos cargados: {df.shape}")

# Crear carpeta de outputs si no existe
os.makedirs('test_outputs', exist_ok=True)

# ============================================
# TEST 1: Evolución temporal
# ============================================
print("\n1️⃣ Probando plot_birth_rate_evolution()...")
try:
    chart1 = plot_birth_rate_evolution(df)
    if chart1 is not None:
        print("   ✅ Gráfico de evolución temporal creado")
        chart1.save('test_outputs/evolution.html')
        print("   💾 Guardado en: test_outputs/evolution.html")
    else:
        print("   ⚠️ No se pudo crear el gráfico")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================
# TEST 2: Comparación regional
# ============================================
print("\n2️⃣ Probando plot_regional_comparison()...")
try:
    chart2 = plot_regional_comparison(df, year=2022)
    if chart2 is not None:
        print("   ✅ Gráfico de comparación regional creado")
        chart2.save('test_outputs/regional.html')
        print("   💾 Guardado en: test_outputs/regional.html")
    else:
        print("   ⚠️ No se pudo crear el gráfico")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================
# TEST 3: Top países
# ============================================
print("\n3️⃣ Probando plot_top_countries()...")
try:
    chart3 = plot_top_countries(df, n=10, year=2022)
    if chart3 is not None:
        print("   ✅ Gráfico de top países creado")
        chart3.save('test_outputs/top_countries.html')
        print("   💾 Guardado en: test_outputs/top_countries.html")
    else:
        print("   ⚠️ No se pudo crear el gráfico")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================
# TEST 4: Bottom países
# ============================================
print("\n4️⃣ Probando plot_top_countries() con ascending=True...")
try:
    chart4 = plot_top_countries(df, n=10, year=2022, ascending=True)
    if chart4 is not None:
        print("   ✅ Gráfico de bottom países creado")
        chart4.save('test_outputs/bottom_countries.html')
        print("   💾 Guardado en: test_outputs/bottom_countries.html")
    else:
        print("   ⚠️ No se pudo crear el gráfico")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================
# TEST 5: Scatter plot (si hay columnas numéricas)
# ============================================
print("\n5️⃣ Probando plot_correlation_scatter()...")
try:
    # Buscar columnas numéricas disponibles
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) >= 2:
        # Intentar con columnas comunes
        x_var = None
        y_var = 'Tasa de natalidad' if 'Tasa de natalidad' in df.columns else 'Birth Rate'
        
        # Buscar una variable X interesante
        for col in ['GDP per capita', 'PIB per capita', 'Life expectancy', 'Esperanza de vida']:
            if col in df.columns:
                x_var = col
                break
        
        if not x_var and len(numeric_cols) > 0:
            x_var = [col for col in numeric_cols if col != y_var][0]
        
        if x_var:
            chart5 = plot_correlation_scatter(df, x_var, y_var, year=2022)
            if chart5 is not None:
                print(f"   ✅ Scatter plot creado ({y_var} vs {x_var})")
                chart5.save('test_outputs/scatter.html')
                print("   💾 Guardado en: test_outputs/scatter.html")
            else:
                print("   ⚠️ No se pudo crear el gráfico")
        else:
            print("   ⏭️ No se encontró columna X adecuada")
    else:
        print("   ⏭️ No hay suficientes columnas numéricas")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================
# TEST 6: Distribución
# ============================================
print("\n6️⃣ Probando plot_distribution()...")
try:
    birth_col = 'Tasa de natalidad' if 'Tasa de natalidad' in df.columns else 'Birth Rate'
    chart6 = plot_distribution(df, variable=birth_col)
    if chart6 is not None:
        print("   ✅ Histograma de distribución creado")
        chart6.save('test_outputs/distribution.html')
        print("   💾 Guardado en: test_outputs/distribution.html")
    else:
        print("   ⚠️ No se pudo crear el gráfico")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================
# TEST 7: Evolución con países específicos
# ============================================
print("\n7️⃣ Probando plot_birth_rate_evolution() con países...")
try:
    countries = ['Argentina', 'Brazil', 'Chile', 'United States', 'China']
    chart7 = plot_birth_rate_evolution(df, countries=countries)
    if chart7 is not None:
        print(f"   ✅ Gráfico con países específicos creado")
        chart7.save('test_outputs/evolution_countries.html')
        print("   💾 Guardado en: test_outputs/evolution_countries.html")
    else:
        print("   ⚠️ No se pudo crear el gráfico")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================
# RESUMEN
# ============================================
print("\n" + "=" * 60)
print("✅ PRUEBAS DE VISUALIZACIÓN COMPLETADAS")
print("=" * 60)

# Contar archivos generados
files = [f for f in os.listdir('test_outputs') if f.endswith('.html')]
print(f"\n📁 {len(files)} gráficos guardados en: test_outputs/")
for f in sorted(files):
    print(f"   • {f}")

print("\n💡 Abre los archivos .html en tu navegador para verlos")
print("\n✨ El módulo visualizations.py está listo para usar en Streamlit!")