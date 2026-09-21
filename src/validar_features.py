import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_FILE = BASE_DIR / "data" / "processed" / "lora_features.parquet"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "validacion_features.txt"


# ============================================================
# CARGA
# ============================================================

print("=" * 70)
print("VALIDACIÓN FINAL DEL DATASET DE FEATURES")
print("=" * 70)

print(f"\nArchivo:")
print(FEATURES_FILE)

if not FEATURES_FILE.exists():
    print("\nERROR: no existe el archivo de features.")
    raise SystemExit(1)

df = pd.read_parquet(FEATURES_FILE)

print("\nArchivo cargado correctamente.")


# ============================================================
# REPORTE
# ============================================================

reporte = []

def mostrar(texto=""):
    print(texto)
    reporte.append(str(texto))


# ============================================================
# 1. INFORMACIÓN GENERAL
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("1. INFORMACIÓN GENERAL")
mostrar("=" * 70)

mostrar(f"Filas:    {len(df):,}")
mostrar(f"Columnas: {len(df.columns)}")

mostrar("\nColumnas:")
for i, col in enumerate(df.columns, 1):
    mostrar(f"  {i:02d}. {col}")


# ============================================================
# 2. TIPOS DE DATOS
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("2. TIPOS DE DATOS")
mostrar("=" * 70)

for col, dtype in df.dtypes.items():
    mostrar(f"{col:30s} {dtype}")


# ============================================================
# 3. DUPLICADOS
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("3. DUPLICADOS")
mostrar("=" * 70)

duplicados = df.duplicated().sum()

mostrar(f"Filas duplicadas completas: {duplicados:,}")

if duplicados == 0:
    mostrar("OK: no existen filas duplicadas completas.")
else:
    mostrar("ATENCIÓN: existen filas duplicadas.")


# ============================================================
# 4. VALORES NaN
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("4. VALORES NaN")
mostrar("=" * 70)

nan_counts = df.isna().sum()
nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

total_nan = int(df.isna().sum().sum())

mostrar(f"NaN totales: {total_nan:,}")

if len(nan_counts) == 0:
    mostrar("OK: no existen NaN.")
else:
    mostrar("\nColumnas con NaN:")
    for col, count in nan_counts.items():
        porcentaje = count / len(df) * 100
        mostrar(f"  {col:30s} {count:8,} ({porcentaje:.2f}%)")


# ============================================================
# 5. INFINITOS
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("5. VALORES INFINITOS")
mostrar("=" * 70)

numeric_df = df.select_dtypes(include=[np.number])

inf_mask = np.isinf(numeric_df)

total_inf = int(inf_mask.sum().sum())

mostrar(f"Inf totales: {total_inf:,}")

if total_inf == 0:
    mostrar("OK: no existen valores infinitos.")
else:
    mostrar("\nColumnas con Inf:")
    for col in numeric_df.columns:
        count = int(np.isinf(numeric_df[col]).sum())
        if count > 0:
            mostrar(f"  {col}: {count:,}")


# ============================================================
# 6. TRANSMISIONES
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("6. TRANSMISIONES")
mostrar("=" * 70)

if "transmission_idx" in df.columns:

    total_tx = df["transmission_idx"].nunique()

    rrh_por_tx = df.groupby("transmission_idx")["rrh_idx"].nunique()

    una_rrh = int((rrh_por_tx == 1).sum())
    dos_rrh = int((rrh_por_tx == 2).sum())
    tres_rrh = int((rrh_por_tx == 3).sum())
    cuatro_rrh = int((rrh_por_tx == 4).sum())

    mostrar(f"Transmisiones únicas: {total_tx:,}")

    mostrar("\nRRH por transmisión:")
    mostrar(f"  1 RRH: {una_rrh:,}")
    mostrar(f"  2 RRH: {dos_rrh:,}")
    mostrar(f"  3 RRH: {tres_rrh:,}")
    mostrar(f"  4 RRH: {cuatro_rrh:,}")

    multi = int((rrh_por_tx > 1).sum())

    mostrar(f"\nTransmisiones recibidas por múltiples RRH: {multi:,}")
    mostrar(f"Porcentaje: {multi / total_tx * 100:.2f}%")


# ============================================================
# 7. ESCENARIOS
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("7. ESCENARIOS")
mostrar("=" * 70)

if "area_type" in df.columns:

    escenarios = df["area_type"].value_counts(dropna=False)

    for escenario, cantidad in escenarios.items():
        porcentaje = cantidad / len(df) * 100
        mostrar(
            f"{str(escenario):30s} "
            f"{cantidad:8,} ({porcentaje:.2f}%)"
        )


# ============================================================
# 8. RRH
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("8. DISTRIBUCIÓN POR RRH")
mostrar("=" * 70)

if "rrh_idx" in df.columns:

    rrh_counts = df["rrh_idx"].value_counts().sort_index()

    for rrh, cantidad in rrh_counts.items():
        porcentaje = cantidad / len(df) * 100
        mostrar(
            f"RRH {rrh}: {cantidad:8,} ({porcentaje:.2f}%)"
        )


# ============================================================
# 9. SAMPLE RATE
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("9. SAMPLE RATE")
mostrar("=" * 70)

if "sample_rate" in df.columns:

    fs_counts = df["sample_rate"].value_counts(dropna=False).sort_index()

    for fs, cantidad in fs_counts.items():
        mostrar(f"{fs}: {cantidad:,}")

    valores_fs = set(df["sample_rate"].dropna().unique())

    fs_validos = {250000, 500000}
    fs_desconocidos = valores_fs - fs_validos

    if fs_desconocidos:
        mostrar(f"\nATENCIÓN: sample rates inesperados: {fs_desconocidos}")
    else:
        mostrar("\nOK: todos los sample rates pertenecen a los valores esperados.")


# ============================================================
# 10. BANDWIDTH
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("10. BANDWIDTH")
mostrar("=" * 70)

if "bandwidth" in df.columns:

    bw_counts = df["bandwidth"].value_counts(dropna=False).sort_index()

    for bw, cantidad in bw_counts.items():
        mostrar(f"{bw}: {cantidad:,}")

    valores_bw = set(df["bandwidth"].dropna().unique())
    bw_validos = {125000, 250000}

    inesperados = valores_bw - bw_validos

    if inesperados:
        mostrar(f"\nATENCIÓN: bandwidth inesperado: {inesperados}")
    else:
        mostrar("\nOK: bandwidth consistente.")


# ============================================================
# 11. SPREADING FACTOR
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("11. SPREADING FACTOR")
mostrar("=" * 70)

if "sf" in df.columns:

    sf_counts = df["sf"].value_counts(dropna=False).sort_index()

    for sf, cantidad in sf_counts.items():
        mostrar(f"SF{sf}: {cantidad:,}")

    valores_sf = set(df["sf"].dropna().unique())
    sf_validos = {7, 8, 9, 10, 11, 12}

    inesperados = valores_sf - sf_validos

    if inesperados:
        mostrar(f"\nATENCIÓN: SF inesperados: {inesperados}")
    else:
        mostrar("\nOK: SF dentro del rango esperado de LoRa.")


# ============================================================
# 12. CODING RATE
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("12. CODING RATE")
mostrar("=" * 70)

if "cr" in df.columns:

    cr_counts = df["cr"].value_counts(dropna=False).sort_index()

    for cr, cantidad in cr_counts.items():
        mostrar(f"CR {cr}: {cantidad:,}")


# ============================================================
# 13. FRECUENCIA
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("13. FRECUENCIA")
mostrar("=" * 70)

if "frequency_MHz" in df.columns:

    mostrar(
        f"Mínima: {df['frequency_MHz'].min():.6f} MHz"
    )
    mostrar(
        f"Máxima: {df['frequency_MHz'].max():.6f} MHz"
    )

    mostrar("\nFrecuencias más frecuentes:")

    freq_counts = df["frequency_MHz"].value_counts().head(20)

    for freq, cantidad in freq_counts.items():
        mostrar(f"  {freq:.6f} MHz: {cantidad:,}")


# ============================================================
# 14. SNR
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("14. SNR")
mostrar("=" * 70)

if "snr" in df.columns:

    snr = df["snr"].dropna()

    mostrar(f"Cantidad válida: {len(snr):,}")
    mostrar(f"Mínimo: {snr.min():.4f}")
    mostrar(f"Máximo: {snr.max():.4f}")
    mostrar(f"Media: {snr.mean():.4f}")
    mostrar(f"Mediana: {snr.median():.4f}")
    mostrar(f"Desviación estándar: {snr.std():.4f}")

    mostrar("\nPercentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        mostrar(f"  P{p}: {snr.quantile(p / 100):.4f}")


# ============================================================
# 15. CFO
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("15. CFO")
mostrar("=" * 70)

if "cfo" in df.columns:

    cfo = df["cfo"].dropna()

    mostrar(f"Cantidad válida: {len(cfo):,}")
    mostrar(f"Mínimo: {cfo.min():.4f}")
    mostrar(f"Máximo: {cfo.max():.4f}")
    mostrar(f"Media: {cfo.mean():.4f}")
    mostrar(f"Mediana: {cfo.median():.4f}")
    mostrar(f"Desviación estándar: {cfo.std():.4f}")


# ============================================================
# 16. POTENCIA IQ
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("16. POTENCIA IQ")
mostrar("=" * 70)

if "iq_power" in df.columns:

    power = df["iq_power"].dropna()

    mostrar(f"Cantidad válida: {len(power):,}")
    mostrar(f"Mínimo: {power.min():.8e}")
    mostrar(f"Máximo: {power.max():.8e}")
    mostrar(f"Media: {power.mean():.8e}")
    mostrar(f"Mediana: {power.median():.8e}")


if "iq_power_db" in df.columns:

    power_db = df["iq_power_db"].dropna()

    mostrar("\nPotencia IQ en escala logarítmica:")

    mostrar(f"Mínimo: {power_db.min():.4f}")
    mostrar(f"Máximo: {power_db.max():.4f}")
    mostrar(f"Media: {power_db.mean():.4f}")
    mostrar(f"Mediana: {power_db.median():.4f}")


# ============================================================
# 17. RANGO DE FEATURES
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("17. RANGOS DE FEATURES NUMÉRICAS")
mostrar("=" * 70)

feature_cols = [
    "iq_power",
    "iq_power_db",
    "iq_rms",
    "iq_peak",
    "amplitude_mean",
    "amplitude_std",
    "amplitude_p10",
    "amplitude_p25",
    "amplitude_p50",
    "amplitude_p75",
    "amplitude_p90",
    "crest_factor",
    "i_mean",
    "q_mean",
    "i_std",
    "q_std",
    "power_std",
    "power_cv",
    "power_min_db",
    "power_max_db",
    "power_range_db",
    "spectral_peak_frequency",
    "spectral_peak_power_db",
    "spectral_total_power_db",
]

for col in feature_cols:

    if col not in df.columns:
        continue

    serie = df[col].dropna()

    if len(serie) == 0:
        mostrar(f"{col:30s} SIN DATOS")
        continue

    mostrar(
        f"{col:30s} "
        f"min={serie.min():.6g} "
        f"max={serie.max():.6g} "
        f"mean={serie.mean():.6g}"
    )


# ============================================================
# 18. CONSISTENCIA SAMPLE_COUNT
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("18. SAMPLE COUNT")
mostrar("=" * 70)

if "sample_count" in df.columns:

    counts = df["sample_count"].value_counts(dropna=False)

    mostrar("Valores de sample_count:")

    for value, cantidad in counts.head(20).items():
        mostrar(f"  {value}: {cantidad:,}")


# ============================================================
# 19. VELOCIDAD
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("19. VELOCIDAD")
mostrar("=" * 70)

if "velocity" in df.columns:

    velocity = df["velocity"].dropna()

    mostrar(f"Valores válidos: {len(velocity):,}")
    mostrar(f"Valores NaN: {df['velocity'].isna().sum():,}")

    if len(velocity) > 0:
        mostrar(f"Mínimo: {velocity.min():.4f}")
        mostrar(f"Máximo: {velocity.max():.4f}")
        mostrar(f"Media: {velocity.mean():.4f}")
        mostrar(f"Mediana: {velocity.median():.4f}")


if "target_velocity" in df.columns:

    target = df["target_velocity"].dropna()

    mostrar("\nTarget velocity:")

    mostrar(f"Valores válidos: {len(target):,}")
    mostrar(f"Mínimo: {target.min():.4f}")
    mostrar(f"Máximo: {target.max():.4f}")
    mostrar(f"Media: {target.mean():.4f}")


# ============================================================
# 20. COORDENADAS
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("20. COORDENADAS")
mostrar("=" * 70)

for col in ["latitude", "longitude"]:

    if col not in df.columns:
        continue

    serie = df[col].dropna()

    mostrar(f"\n{col}:")
    mostrar(f"  Válidos: {len(serie):,}")
    mostrar(f"  NaN: {df[col].isna().sum():,}")

    if len(serie) > 0:
        mostrar(f"  Mínimo: {serie.min():.8f}")
        mostrar(f"  Máximo: {serie.max():.8f}")


# ============================================================
# 21. ANOMALÍAS BÁSICAS
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("21. ANOMALÍAS BÁSICAS")
mostrar("=" * 70)

anomalias = {}

# Potencia negativa
if "iq_power" in df.columns:
    n = int((df["iq_power"].dropna() < 0).sum())
    anomalias["iq_power_negativa"] = n
    mostrar(f"iq_power < 0: {n:,}")

# RMS negativa
if "iq_rms" in df.columns:
    n = int((df["iq_rms"].dropna() < 0).sum())
    anomalias["iq_rms_negativo"] = n
    mostrar(f"iq_rms < 0: {n:,}")

# Peak negativo
if "iq_peak" in df.columns:
    n = int((df["iq_peak"].dropna() < 0).sum())
    anomalias["iq_peak_negativo"] = n
    mostrar(f"iq_peak < 0: {n:,}")

# Crest factor negativo
if "crest_factor" in df.columns:
    n = int((df["crest_factor"].dropna() < 0).sum())
    anomalias["crest_factor_negativo"] = n
    mostrar(f"crest_factor < 0: {n:,}")

# Sample count negativo
if "sample_count" in df.columns:
    n = int((df["sample_count"].dropna() <= 0).sum())
    anomalias["sample_count_invalido"] = n
    mostrar(f"sample_count <= 0: {n:,}")

# Sample rate inválido
if "sample_rate" in df.columns:
    n = int(
        (~df["sample_rate"].isin([250000, 500000])).sum()
    )
    anomalias["sample_rate_invalido"] = n
    mostrar(f"sample_rate inesperado: {n:,}")


# ============================================================
# 22. CONCLUSIÓN
# ============================================================

mostrar("\n" + "=" * 70)
mostrar("22. CONCLUSIÓN")
mostrar("=" * 70)

problemas = []

if duplicados > 0:
    problemas.append("duplicados")

if total_inf > 0:
    problemas.append("valores infinitos")

if anomalias.get("iq_power_negativa", 0) > 0:
    problemas.append("potencia IQ negativa")

if anomalias.get("iq_rms_negativo", 0) > 0:
    problemas.append("RMS negativa")

if anomalias.get("sample_count_invalido", 0) > 0:
    problemas.append("sample_count inválido")

if anomalias.get("sample_rate_invalido", 0) > 0:
    problemas.append("sample rate inesperado")


if problemas:
    mostrar("\nATENCIÓN:")
    mostrar("Se encontraron las siguientes condiciones:")
    for problema in problemas:
        mostrar(f"  - {problema}")
else:
    mostrar("\nOK:")
    mostrar("No se encontraron inconsistencias estructurales graves.")


mostrar("\nLos NaN presentes en latitude, longitude y velocity")
mostrar("corresponden a valores faltantes de la metadata original.")
mostrar("No fueron imputados ni modificados.")

mostrar("\nLa potencia IQ se interpreta como potencia relativa")
mostrar("en las unidades del registro IQ; no representa dBm")
mostrar("porque no existe una calibración absoluta disponible.")

mostrar("\nEl dataset queda preservado sin modificaciones y")
mostrar("puede utilizarse como entrada para la siguiente etapa.")


# ============================================================
# GUARDAR REPORTE
# ============================================================

OUTPUT_FILE.write_text(
    "\n".join(reporte),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("REPORTE GUARDADO")
print("=" * 70)
print(OUTPUT_FILE)
print("\nValidación finalizada correctamente.")