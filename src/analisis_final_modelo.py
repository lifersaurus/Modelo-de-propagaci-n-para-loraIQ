import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "data" / "processed" / "lora_features.parquet"
OUTPUT_DIR = BASE_DIR / "results" / "model_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ANÁLISIS FINAL DE PREPARACIÓN PARA EL MODELO")
print("=" * 70)

# ============================================================
# CARGA
# ============================================================

df = pd.read_parquet(INPUT)

print(f"\nRegistros: {len(df):,}")
print(f"Columnas:  {len(df.columns)}")

# ============================================================
# 1. ANÁLISIS DE CFO
# ============================================================

print("\n" + "=" * 70)
print("1. ANÁLISIS DE CFO")
print("=" * 70)

cfo = df["cfo"].dropna()

print(f"\nRegistros válidos: {len(cfo):,}")
print(f"Media:   {cfo.mean():.4f}")
print(f"Mediana: {cfo.median():.4f}")
print(f"Std:     {cfo.std():.4f}")
print(f"Min:     {cfo.min():.4f}")
print(f"Max:     {cfo.max():.4f}")

print("\nPercentiles:")
print(cfo.quantile([0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]))

# Valores extremos
print("\nValores |CFO| > 50:")
print((cfo.abs() > 50).sum())

print("Valores |CFO| > 100:")
print((cfo.abs() > 100).sum())

print("Valores |CFO| > 200:")
print((cfo.abs() > 200).sum())

# Histograma
plt.figure(figsize=(10, 6))
plt.hist(cfo, bins=100)
plt.xlabel("CFO")
plt.ylabel("Frecuencia")
plt.title("Distribución del CFO")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_distribucion_cfo.png", dpi=150)
plt.close()

# CFO por escenario
cfo_scenario = (
    df.groupby("area_type")["cfo"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print("\nCFO por escenario:")
print(cfo_scenario)

cfo_scenario.to_csv(
    OUTPUT_DIR / "cfo_por_escenario.csv"
)

# ============================================================
# 2. MOVILIDAD
# ============================================================

print("\n" + "=" * 70)
print("2. ANÁLISIS DE MOVILIDAD")
print("=" * 70)

# ------------------------------------------------------------
# Velocidad real
# ------------------------------------------------------------

valid_velocity = df.dropna(subset=["velocity", "snr"])

print(f"\nRegistros con velocity + SNR: {len(valid_velocity):,}")

if len(valid_velocity) > 1:
    corr_velocity = valid_velocity["snr"].corr(
        valid_velocity["velocity"]
    )
else:
    corr_velocity = np.nan

print(f"Correlación SNR-velocity: {corr_velocity:.4f}")

# ------------------------------------------------------------
# Target velocity
# ------------------------------------------------------------

valid_target = df.dropna(subset=["target_velocity", "snr"])

corr_target = valid_target["snr"].corr(
    valid_target["target_velocity"]
)

print(f"Correlación SNR-target_velocity: {corr_target:.4f}")

# ------------------------------------------------------------
# Por escenario
# ------------------------------------------------------------

mobility_summary = (
    df.groupby("area_type")
    .agg(
        registros=("snr", "size"),
        snr_mean=("snr", "mean"),
        snr_median=("snr", "median"),
        velocity_mean=("velocity", "mean"),
        velocity_median=("velocity", "median"),
        target_velocity_mean=("target_velocity", "mean")
    )
)

print("\nMovilidad por escenario:")
print(mobility_summary)

mobility_summary.to_csv(
    OUTPUT_DIR / "movilidad_por_escenario.csv"
)

# ------------------------------------------------------------
# SNR vs velocity
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.scatter(
    valid_velocity["velocity"],
    valid_velocity["snr"],
    s=3,
    alpha=0.25
)
plt.xlabel("Velocity")
plt.ylabel("SNR")
plt.title("SNR vs Velocity")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "02_snr_vs_velocity.png",
    dpi=150
)
plt.close()

# ------------------------------------------------------------
# SNR vs target velocity
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.scatter(
    valid_target["target_velocity"],
    valid_target["snr"],
    s=3,
    alpha=0.25
)
plt.xlabel("Target velocity")
plt.ylabel("SNR")
plt.title("SNR vs Target Velocity")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "03_snr_vs_target_velocity.png",
    dpi=150
)
plt.close()

# ============================================================
# 3. ANÁLISIS ESPACIAL
# ============================================================

print("\n" + "=" * 70)
print("3. ANÁLISIS ESPACIAL")
print("=" * 70)

spatial = df.dropna(
    subset=["latitude", "longitude"]
).copy()

print(f"\nRegistros con coordenadas: {len(spatial):,}")

print("\nRango de coordenadas:")

print(
    f"Latitude:  "
    f"{spatial['latitude'].min():.8f} - "
    f"{spatial['latitude'].max():.8f}"
)

print(
    f"Longitude: "
    f"{spatial['longitude'].min():.8f} - "
    f"{spatial['longitude'].max():.8f}"
)

# Número de posiciones únicas
unique_positions = spatial[
    ["latitude", "longitude"]
].drop_duplicates()

print(
    f"\nPosiciones únicas: "
    f"{len(unique_positions):,}"
)

# Coordenadas por transmisión
positions_tx = (
    spatial.groupby("transmission_idx")
    .agg(
        lat_min=("latitude", "min"),
        lat_max=("latitude", "max"),
        lon_min=("longitude", "min"),
        lon_max=("longitude", "max"),
        registros=("latitude", "size")
    )
)

positions_tx["lat_range"] = (
    positions_tx["lat_max"] -
    positions_tx["lat_min"]
)

positions_tx["lon_range"] = (
    positions_tx["lon_max"] -
    positions_tx["lon_min"]
)

print("\nVariación de posición por transmisión:")

print(
    positions_tx[
        ["registros", "lat_range", "lon_range"]
    ].describe()
)

positions_tx.to_csv(
    OUTPUT_DIR / "trayectorias_por_transmision.csv"
)

# ------------------------------------------------------------
# Mapa espacial SNR
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    spatial["longitude"],
    spatial["latitude"],
    c=spatial["snr"],
    s=4,
    alpha=0.5
)

plt.colorbar(scatter, label="SNR")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Distribución espacial del SNR")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_snr_espacial.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# Mapa espacial por escenario
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

for scenario in spatial["area_type"].dropna().unique():

    subset = spatial[
        spatial["area_type"] == scenario
    ]

    plt.scatter(
        subset["longitude"],
        subset["latitude"],
        s=3,
        alpha=0.4,
        label=scenario
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Distribución espacial por escenario")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_escenarios_espaciales.png",
    dpi=150
)

plt.close()

# ============================================================
# 4. ANÁLISIS MULTI-RRH
# ============================================================

print("\n" + "=" * 70)
print("4. ANÁLISIS MULTI-RRH")
print("=" * 70)

rrh_per_tx = (
    df.groupby("transmission_idx")["rrh_idx"]
    .nunique()
)

multi_tx = rrh_per_tx[
    rrh_per_tx > 1
].index

multi = df[
    df["transmission_idx"].isin(multi_tx)
].copy()

print(f"\nTransmisiones multi-RRH: {len(multi_tx):,}")
print(f"Registros involucrados: {len(multi):,}")

# ------------------------------------------------------------
# SNR por transmisión
# ------------------------------------------------------------

snr_rrh = (
    multi.groupby("transmission_idx")["snr"]
    .agg(
        snr_min="min",
        snr_max="max",
        snr_mean="mean",
        snr_std="std"
    )
)

snr_rrh["snr_range"] = (
    snr_rrh["snr_max"] -
    snr_rrh["snr_min"]
)

print("\nVariación de SNR entre RRH:")
print(
    snr_rrh[
        ["snr_mean", "snr_std", "snr_range"]
    ].describe()
)

# ------------------------------------------------------------
# Power por transmisión
# ------------------------------------------------------------

power_rrh = (
    multi.groupby("transmission_idx")["iq_power_db"]
    .agg(
        power_min="min",
        power_max="max",
        power_mean="mean",
        power_std="std"
    )
)

power_rrh["power_range"] = (
    power_rrh["power_max"] -
    power_rrh["power_min"]
)

print("\nVariación de potencia entre RRH:")
print(
    power_rrh[
        ["power_mean", "power_std", "power_range"]
    ].describe()
)

# Guardar resultados
multi_rrh_summary = snr_rrh.join(
    power_rrh,
    how="inner"
)

multi_rrh_summary.to_csv(
    OUTPUT_DIR / "variacion_multi_rrh.csv"
)

# ------------------------------------------------------------
# Histograma diferencia SNR
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    snr_rrh["snr_range"],
    bins=60
)

plt.xlabel("Rango de SNR entre RRH [dB]")
plt.ylabel("Frecuencia")
plt.title("Variación de SNR entre RRH para una misma transmisión")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_variacion_snr_multi_rrh.png",
    dpi=150
)

plt.close()

# ============================================================
# 5. RESUMEN DE VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("5. RESUMEN DE VARIABLES")
print("=" * 70)

for column in df.columns:

    missing = df[column].isna().sum()
    unique = df[column].nunique()

    print(
        f"{column:30s} "
        f"NaN={missing:7,d} "
        f"Unique={unique:7,d}"
    )

# ============================================================
# 6. GUARDAR RESUMEN GENERAL
# ============================================================

summary = {
    "registros": len(df),
    "columnas": len(df.columns),
    "cfo_validos": len(cfo),
    "cfo_abs_mayor_50": int((cfo.abs() > 50).sum()),
    "cfo_abs_mayor_100": int((cfo.abs() > 100).sum()),
    "cfo_abs_mayor_200": int((cfo.abs() > 200).sum()),
    "registros_velocity_snr": len(valid_velocity),
    "corr_snr_velocity": corr_velocity,
    "corr_snr_target_velocity": corr_target,
    "registros_con_coordenadas": len(spatial),
    "posiciones_unicas": len(unique_positions),
    "transmisiones_multi_rrh": len(multi_tx),
    "registros_multi_rrh": len(multi)
}

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    OUTPUT_DIR / "resumen_final.csv",
    index=False
)

print("\n" + "=" * 70)
print("ANÁLISIS FINALIZADO")
print("=" * 70)

print(f"\nResultados guardados en:")
print(OUTPUT_DIR)