from src.pipeline import ejecutar_pipeline_completo
from src.visualizations import get_available_visualizations
import os
import altair as alt

print("=" * 60)
print("📊 TEST DE VISUALIZACIONES")
print("=" * 60)

# 1. Cargar datos procesados
print("\n1️⃣ Ejecutando pipeline...")
df = ejecutar_pipeline_completo('data/raw/merged_dataset.csv')
print(f"   ✅ Datos listos: {df.shape}")

# 2. Crear carpeta de outputs
os.makedirs('test_outputs', exist_ok=True)

# 3. Generar visualizaciones
print("\n2️⃣ Generando visualizaciones...")
vizs = get_available_visualizations()

for viz in vizs:
    print(f"\n   📊 {viz['nombre']}...")
    try:
        chart = viz['funcion'](df)
        filename = f"test_outputs/{viz['id']}.html"
        chart.save(filename)
        print(f"      ✅ Guardado en: {filename}")
    except Exception as e:
        print(f"      ❌ Error: {e}")

print("\n✅ Visualizaciones generadas!")
print("💡 Abre los archivos HTML en test_outputs/ para verlos")