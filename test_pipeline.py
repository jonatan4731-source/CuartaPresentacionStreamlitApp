from src.pipeline import ejecutar_pipeline_completo, get_resumen_pipeline, cargar_datos

print("=" * 60)
print("🧪 TEST DEL PIPELINE COMPLETO")
print("=" * 60)

# Cargar original
print("\n1️⃣ Cargando dataset original...")
df_original = cargar_datos('data/raw/merged_dataset.csv')
print(f"   ✅ Original: {df_original.shape}")

# Ejecutar pipeline
print("\n2️⃣ Ejecutando pipeline completo...")
df_procesado = ejecutar_pipeline_completo('data/raw/merged_dataset.csv')
print(f"   ✅ Procesado: {df_procesado.shape}")

# Resumen
print("\n3️⃣ Resumen:")
resumen = get_resumen_pipeline(df_original, df_procesado)
for key, value in resumen.items():
    print(f"   • {key}: {value}")

print("\n✅ Pipeline funcionando correctamente!")