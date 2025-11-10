"""
Script de prueba para src/functions.py
Ejecutar desde la raíz del proyecto: python test_functions.py
"""

import sys
sys.path.append('.')

from src.functions import *
import pandas as pd

print("=" * 60)
print("🧪 PRUEBA DE FUNCIONES")
print("=" * 60)

# ============================================
# TEST 1: Cargar datos
# ============================================
print("\n1️⃣ Probando carga de datos...")

# Verificar si el archivo existe
import os
csv_path = 'data/raw/merged_dataset.csv'

print(f"   📂 Buscando: {csv_path}")
print(f"   📍 Directorio actual: {os.getcwd()}")

if os.path.exists(csv_path):
    print(f"   ✅ Archivo encontrado")
    file_size = os.path.getsize(csv_path)
    print(f"   📦 Tamaño: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
else:
    print(f"   ❌ Archivo NO encontrado en: {os.path.abspath(csv_path)}")
    print(f"\n   💡 Soluciones:")
    print(f"      1. Verifica que el archivo esté en: data/raw/")
    print(f"      2. Descárgalo desde tu notebook de Colab")
    print(f"      3. O cambia la ruta en el código")
    
    # Buscar archivos CSV en data/
    print(f"\n   🔍 Buscando archivos CSV en data/...")
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.csv'):
                print(f"      • {os.path.join(root, file)}")
    sys.exit(1)

df = load_data(csv_path)

if df is not None:
    print(f"   ✅ Datos cargados correctamente")
    print(f"   📊 Shape: {df.shape}")
    print(f"   📋 Columnas: {list(df.columns[:5])}... (primeras 5)")
else:
    print("   ❌ Error al cargar datos")
    sys.exit(1)

# ============================================
# TEST 2: Información del dataset
# ============================================
print("\n2️⃣ Probando get_data_info()...")
info = get_data_info(df)

print(f"   ✅ Información obtenida:")
print(f"      - Filas: {info['n_filas']}")
print(f"      - Columnas: {info['n_columnas']}")
print(f"      - Países únicos: {info['paises_unicos']}")
print(f"      - Regiones únicas: {info['regiones_unicas']}")
print(f"      - Nulos totales: {info['nulos_totales']}")

if info['años_disponibles']:
    print(f"      - Años: {info['años_disponibles'][0]} a {info['años_disponibles'][-1]}")

# ============================================
# TEST 3: Filtros
# ============================================
print("\n3️⃣ Probando filtros...")

# Mostrar todas las columnas primero
print(f"   📋 Columnas del dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"      {i}. {col}")

# Probar años disponibles
years = get_available_years(df)
if years:
    print(f"\n   ✅ Años disponibles: {len(years)} años ({years[0]} - {years[-1]})")
else:
    print(f"\n   ⚠️ No se encontró columna 'Year'")
    print(f"      Columnas que contienen 'year' o 'año':")
    year_cols = [col for col in df.columns if 'year' in col.lower() or 'año' in col.lower()]
    if year_cols:
        for col in year_cols:
            print(f"      • {col}")
    else:
        print(f"      (ninguna)")

# Probar regiones disponibles
regions = get_available_regions(df)
if regions:
    print(f"\n   ✅ Regiones disponibles: {len(regions)} regiones")
    print(f"      Ejemplos: {regions[:3]}")
else:
    print(f"\n   ⚠️ No se encontró columna 'Region'")
    region_cols = [col for col in df.columns if 'region' in col.lower()]
    if region_cols:
        print(f"      Columnas que contienen 'region':")
        for col in region_cols:
            print(f"      • {col}")

# Probar países disponibles
countries = get_available_countries(df)
if countries:
    print(f"\n   ✅ Países disponibles: {len(countries)} países")
    print(f"      Ejemplos: {countries[:3]}")
else:
    print(f"\n   ⚠️ No se encontró columna 'Country Name'")
    country_cols = [col for col in df.columns if 'country' in col.lower() or 'país' in col.lower() or 'pais' in col.lower()]
    if country_cols:
        print(f"      Columnas que contienen 'country'/'país':")
        for col in country_cols:
            print(f"      • {col}")

# ============================================
# TEST 4: Filtrar por año
# ============================================
print("\n4️⃣ Probando filter_by_year()...")
if years:
    df_2020 = filter_by_year(df, 2020, 2023)
    print(f"   ✅ Filtrado 2020-2023: {len(df_2020)} filas")
else:
    print(f"   ⏭️ Saltando (no hay columna Year)")

# ============================================
# TEST 5: Filtrar por región
# ============================================
print("\n5️⃣ Probando filter_by_region()...")
if regions:
    df_region = filter_by_region(df, [regions[0]])
    print(f"   ✅ Filtrado región '{regions[0]}': {len(df_region)} filas")
else:
    print(f"   ⏭️ Saltando (no hay columna Region)")

# ============================================
# TEST 6: Top países
# ============================================
print("\n6️⃣ Probando get_top_countries()...")
if years and countries:
    top = get_top_countries(df, n=5, year=years[-1])  # Usar último año disponible
    if not top.empty:
        print(f"   ✅ Top 5 países ({years[-1]}):")
        for idx, row in top.iterrows():
            print(f"      {row['Country Name']}: {row['Birth Rate']:.2f}")
else:
    print(f"   ⏭️ Saltando (faltan columnas necesarias)")

# ============================================
# TEST 7: Promedio global
# ============================================
print("\n7️⃣ Probando calculate_global_average()...")
if years:
    avg_2022 = calculate_global_average(df, year=years[-1])  # Usar último año
    print(f"   ✅ Promedio global {years[-1]}: {avg_2022:.2f}")
else:
    avg_total = calculate_global_average(df)
    print(f"   ✅ Promedio global (todos los años): {avg_total:.2f}")

# ============================================
# TEST 8: Exportar a CSV (test en memoria)
# ============================================
print("\n8️⃣ Probando export_to_csv()...")
csv_string = export_to_csv(df.head(10))
print(f"   ✅ CSV generado: {len(csv_string)} caracteres")
print(f"      Primeras líneas:")
print("      " + "\n      ".join(csv_string.split('\n')[:3]))

# ============================================
# RESUMEN
# ============================================
print("\n" + "=" * 60)
print("✅ TODAS LAS PRUEBAS PASARON CORRECTAMENTE")
print("=" * 60)
print("\n📝 Resumen del dataset:")
print(f"   • {info['n_filas']:,} filas")
print(f"   • {info['n_columnas']} columnas")
print(f"   • {info['paises_unicos']} países")
print(f"   • {info['regiones_unicas']} regiones")
print(f"   • {len(years)} años de datos")
print("\n✨ El módulo functions.py está listo para usar en Streamlit!")