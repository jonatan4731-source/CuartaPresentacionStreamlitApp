"""
Script para debuggear qué columnas tenemos exactamente
"""

import sys
sys.path.append('.')

from src.functions import load_data
import pandas as pd

print("=" * 60)
print("🔍 DEBUG: VERIFICAR COLUMNAS")
print("=" * 60)

# Cargar datos
df = load_data('data/raw/merged_dataset.csv', add_geography=True)

if df is None:
    print("❌ No se pudo cargar el dataset")
    sys.exit(1)

print(f"\n✅ Dataset cargado: {df.shape}")

print("\n" + "=" * 60)
print("📋 TODAS LAS COLUMNAS:")
print("=" * 60)

for i, col in enumerate(df.columns, 1):
    print(f"{i:3}. {col}")

print("\n" + "=" * 60)
print("🔍 BUSCAR COLUMNAS CLAVE:")
print("=" * 60)

# Buscar columnas relacionadas con natalidad
print("\n📊 Columnas con 'natal' o 'birth' o 'tasa':")
birth_cols = [col for col in df.columns if any(word in col.lower() for word in ['natal', 'birth', 'tasa', 'rate'])]
for col in birth_cols:
    print(f"   • {col}")
    # Mostrar ejemplo de valores
    sample = df[col].dropna().head(3).tolist()
    print(f"     Ejemplos: {sample}")

# Buscar columnas temporales
print("\n📅 Columnas con 'año' o 'year':")
year_cols = [col for col in df.columns if any(word in col.lower() for word in ['año', 'year'])]
for col in year_cols:
    print(f"   • {col}")
    print(f"     Valores únicos: {df[col].nunique()}")
    print(f"     Rango: {df[col].min()} - {df[col].max()}")

# Buscar columnas de país
print("\n🌍 Columnas con 'pais' o 'country':")
country_cols = [col for col in df.columns if any(word in col.lower() for word in ['pais', 'país', 'country'])]
for col in country_cols:
    print(f"   • {col}")
    print(f"     Países únicos: {df[col].nunique()}")
    print(f"     Ejemplos: {df[col].unique()[:3].tolist()}")

# Verificar las columnas agregadas
print("\n🗺️ Columnas agregadas por geography_utils:")
if 'Region' in df.columns:
    print(f"   ✅ Region: {df['Region'].nunique()} regiones")
    print(f"      Ejemplos: {df['Region'].unique()[:5].tolist()}")
else:
    print(f"   ❌ Region: NO ENCONTRADA")

if 'Continente' in df.columns:
    print(f"   ✅ Continente: {df['Continente'].nunique()} continentes")
    print(f"      Ejemplos: {df['Continente'].unique().tolist()}")
else:
    print(f"   ❌ Continente: NO ENCONTRADA")

print("\n" + "=" * 60)
print("💡 DIAGNÓSTICO:")
print("=" * 60)

# Verificar nombres esperados vs reales
expected = {
    'Año': 'Año' in df.columns,
    'Year': 'Year' in df.columns,
    'Pais': 'Pais' in df.columns,
    'País': 'País' in df.columns,
    'Country Name': 'Country Name' in df.columns,
    'Tasa de natalidad': 'Tasa de natalidad' in df.columns,
    'Birth Rate': 'Birth Rate' in df.columns,
    'Region': 'Region' in df.columns,
    'Región': 'Región' in df.columns,
}

for name, exists in expected.items():
    status = "✅" if exists else "❌"
    print(f"{status} '{name}'")