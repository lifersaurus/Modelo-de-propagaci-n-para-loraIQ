import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "lora_features.parquet"
RESULTS_DIR = BASE_DIR / "results" / "features_analysis"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("ANÁLISIS EXPLORATORIO DE FEATURES")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)

print(f"\nRegistros: {len(df):,}")
print(f"Columnas:  {len(df.columns)}")



def guardar_figura(nombre):
    ruta = RESULTS_DIR / nombre
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  Guardado: {ruta}")


print("\n" + "=" * 70)
print("1. SNR POR ESCENARIO")
print("=" * 70)

snr_scenario = (
    df.groupby("area_type")["snr"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .sort_values("mean", ascending=False)
)

print(snr_scenario)

plt.figure(figsize=(10, 6))

df.boxplot(
    column="snr",
    by="area_type",
    rot=25
)

plt.title("SNR por escenario")
plt.suptitle("")
plt.xlabel("Escenario")
plt.ylabel("SNR")

guardar_figura("01_snr_por_escenario.png")

print("\n" + "=" * 70)
print("2. SNR POR RRH")
print("=" * 70)

snr_rrh = (
    df.groupby("rrh_idx")["snr"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(snr_rrh)

plt.figure(figsize=(8, 6))

df.boxplot(
    column="snr",
    by="rrh_idx"
)

plt.title("SNR por RRH")
plt.suptitle("")
plt.xlabel("RRH")
plt.ylabel("SNR")

guardar_figura("02_snr_por_rrh.png")

print("\n" + "=" * 70)
print("3. POTENCIA IQ POR ESCENARIO")
print("=" * 70)

power_scenario = (
    df.groupby("area_type")["iq_power_db"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .sort_values("mean", ascending=False)
)

print(power_scenario)

plt.figure(figsize=(10, 6))

df.boxplot(
    column="iq_power_db",
    by="area_type",
    rot=25
)

plt.title("Potencia IQ relativa por escenario")
plt.suptitle("")
plt.xlabel("Escenario")
plt.ylabel("Potencia IQ relativa (dB)")

guardar_figura("03_potencia_por_escenario.png")

print("\n" + "=" * 70)
print("4. POTENCIA IQ POR RRH")
print("=" * 70)

power_rrh = (
    df.groupby("rrh_idx")["iq_power_db"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(power_rrh)

plt.figure(figsize=(8, 6))

df.boxplot(
    column="iq_power_db",
    by="rrh_idx"
)

plt.title("Potencia IQ relativa por RRH")
plt.suptitle("")
plt.xlabel("RRH")
plt.ylabel("Potencia IQ relativa (dB)")

guardar_figura("04_potencia_por_rrh.png")


print("\n" + "=" * 70)
print("5. RELACIÓN SNR - POTENCIA IQ")
print("=" * 70)

correlacion = df[["snr", "iq_power_db"]].corr().iloc[0, 1]

print(f"Correlación Pearson SNR / potencia IQ: {correlacion:.6f}")

plt.figure(figsize=(9, 6))

plt.scatter(
    df["iq_power_db"],
    df["snr"],
    s=3,
    alpha=0.25
)

plt.xlabel("Potencia IQ relativa (dB)")
plt.ylabel("SNR")
plt.title("SNR vs potencia IQ relativa")
plt.grid(True, alpha=0.2)

guardar_figura("05_snr_vs_potencia.png")


print("\n" + "=" * 70)
print("6. SNR VS VELOCIDAD")
print("=" * 70)

tmp = df.dropna(subset=["velocity", "snr"])

print(f"Registros con velocidad y SNR: {len(tmp):,}")

if len(tmp) > 0:

    correlacion_vel = tmp[["velocity", "snr"]].corr().iloc[0, 1]

    print(
        f"Correlación Pearson velocidad / SNR: "
        f"{correlacion_vel:.6f}"
    )

    plt.figure(figsize=(9, 6))

    plt.scatter(
        tmp["velocity"],
        tmp["snr"],
        s=3,
        alpha=0.25
    )

    plt.xlabel("Velocidad")
    plt.ylabel("SNR")
    plt.title("SNR vs velocidad")
    plt.grid(True, alpha=0.2)

    guardar_figura("06_snr_vs_velocidad.png")



print("\n" + "=" * 70)
print("7. SNR VS TARGET VELOCITY")
print("=" * 70)

tmp = df.dropna(subset=["target_velocity", "snr"])

corr_target = tmp[
    ["target_velocity", "snr"]
].corr().iloc[0, 1]

print(
    f"Correlación Pearson target_velocity / SNR: "
    f"{corr_target:.6f}"
)

plt.figure(figsize=(9, 6))

plt.scatter(
    tmp["target_velocity"],
    tmp["snr"],
    s=3,
    alpha=0.25
)

plt.xlabel("Target velocity")
plt.ylabel("SNR")
plt.title("SNR vs target velocity")
plt.grid(True, alpha=0.2)

guardar_figura("07_snr_vs_target_velocity.png")


print("\n" + "=" * 70)
print("8. SNR POR SF")
print("=" * 70)

snr_sf = (
    df.groupby("sf")["snr"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(snr_sf)

plt.figure(figsize=(8, 6))

df.boxplot(
    column="snr",
    by="sf"
)

plt.title("SNR por Spreading Factor")
plt.suptitle("")
plt.xlabel("SF")
plt.ylabel("SNR")

guardar_figura("08_snr_por_sf.png")



print("\n" + "=" * 70)
print("9. POTENCIA IQ POR SF")
print("=" * 70)

power_sf = (
    df.groupby("sf")["iq_power_db"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(power_sf)

plt.figure(figsize=(8, 6))

df.boxplot(
    column="iq_power_db",
    by="sf"
)

plt.title("Potencia IQ relativa por SF")
plt.suptitle("")
plt.xlabel("SF")
plt.ylabel("Potencia IQ relativa (dB)")

guardar_figura("09_potencia_por_sf.png")


print("\n" + "=" * 70)
print("10. CFO")
print("=" * 70)

cfo = df["cfo"].dropna()

print(f"Cantidad: {len(cfo):,}")
print(f"Media: {cfo.mean():.6f}")
print(f"Mediana: {cfo.median():.6f}")
print(f"Std: {cfo.std():.6f}")

plt.figure(figsize=(9, 6))

plt.hist(
    cfo,
    bins=100
)

plt.xlabel("CFO")
plt.ylabel("Cantidad de registros")
plt.title("Distribución de CFO")
plt.grid(True, alpha=0.2)

guardar_figura("10_distribucion_cfo.png")



print("\n" + "=" * 70)
print("11. VARIACIÓN TEMPORAL DE POTENCIA")
print("=" * 70)

power_range = df["power_range_db"].dropna()

print(f"Media: {power_range.mean():.6f}")
print(f"Mediana: {power_range.median():.6f}")
print(f"Máximo: {power_range.max():.6f}")

plt.figure(figsize=(9, 6))

plt.hist(
    power_range,
    bins=100
)

plt.xlabel("Rango de potencia (dB)")
plt.ylabel("Cantidad")
plt.title("Distribución del rango de potencia")
plt.grid(True, alpha=0.2)

guardar_figura("11_power_range.png")


print("\n" + "=" * 70)
print("12. CREST FACTOR")
print("=" * 70)

crest = df["crest_factor"].dropna()

print(f"Media: {crest.mean():.6f}")
print(f"Mediana: {crest.median():.6f}")
print(f"Máximo: {crest.max():.6f}")

plt.figure(figsize=(9, 6))

plt.hist(
    crest,
    bins=100
)

plt.xlabel("Crest factor")
plt.ylabel("Cantidad")
plt.title("Distribución del Crest Factor")
plt.grid(True, alpha=0.2)

guardar_figura("12_crest_factor.png")



print("\n" + "=" * 70)
print("13. DISTRIBUCIÓN ESPACIAL")
print("=" * 70)

geo = df.dropna(
    subset=["latitude", "longitude"]
)

print(f"Registros con coordenadas: {len(geo):,}")

plt.figure(figsize=(9, 7))

plt.scatter(
    geo["longitude"],
    geo["latitude"],
    s=3,
    alpha=0.3
)

plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Distribución espacial de las transmisiones")
plt.grid(True, alpha=0.2)

guardar_figura("13_distribucion_espacial.png")



plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    geo["longitude"],
    geo["latitude"],
    c=geo["snr"],
    s=4,
    alpha=0.5
)

plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Distribución espacial según SNR")

plt.colorbar(
    scatter,
    label="SNR"
)

plt.grid(True, alpha=0.2)

guardar_figura("14_espacial_snr.png")




plt.figure(figsize=(10, 7))

for escenario in sorted(
    geo["area_type"].dropna().unique()
):

    subset = geo[
        geo["area_type"] == escenario
    ]

    plt.scatter(
        subset["longitude"],
        subset["latitude"],
        s=4,
        alpha=0.4,
        label=escenario
    )

plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Distribución espacial por escenario")
plt.legend()

plt.grid(True, alpha=0.2)

guardar_figura("15_espacial_escenario.png")




print("\n" + "=" * 70)
print("16. TRANSMISIONES MULTI-RRH")
print("=" * 70)

rrh_tx = (
    df.groupby("transmission_idx")["rrh_idx"]
    .nunique()
)

multi_tx = rrh_tx[rrh_tx > 1]

print(f"Transmisiones multi-RRH: {len(multi_tx):,}")

print(
    f"Promedio de RRH por transmisión multi-RRH: "
    f"{multi_tx.mean():.4f}"
)



snr_range_tx = (
    df.groupby("transmission_idx")["snr"]
    .agg(lambda x: x.max() - x.min())
)

snr_range_multi = snr_range_tx[
    snr_range_tx.index.isin(multi_tx.index)
]

print(
    f"Diferencia media de SNR entre RRH: "
    f"{snr_range_multi.mean():.4f}"
)

print(
    f"Mediana diferencia SNR entre RRH: "
    f"{snr_range_multi.median():.4f}"
)

plt.figure(figsize=(9, 6))

plt.hist(
    snr_range_multi,
    bins=100
)

plt.xlabel("Rango de SNR entre RRH")
plt.ylabel("Cantidad de transmisiones")
plt.title("Variación de SNR entre RRH para una misma transmisión")
plt.grid(True, alpha=0.2)

guardar_figura("16_snr_diferencia_multi_rrh.png")



print("\n" + "=" * 70)
print("17. MATRIZ DE CORRELACIÓN")
print("=" * 70)

corr_cols = [
    "snr",
    "iq_power_db",
    "iq_rms",
    "iq_peak",
    "amplitude_mean",
    "amplitude_std",
    "crest_factor",
    "power_std",
    "power_cv",
    "power_range_db",
    "spectral_peak_frequency",
    "spectral_peak_power_db",
    "spectral_total_power_db",
    "cfo",
    "velocity",
    "target_velocity",
]

corr_cols = [
    c for c in corr_cols
    if c in df.columns
]

corr = df[corr_cols].corr()

print(corr.round(3))

plt.figure(figsize=(12, 10))

plt.imshow(
    corr,
    aspect="auto"
)

plt.colorbar(label="Correlación")

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.index)),
    corr.index
)

plt.title("Matriz de correlación de variables")

guardar_figura("17_matriz_correlacion.png")




print("\n" + "=" * 70)
print("18. RESUMEN POR ESCENARIO")
print("=" * 70)

resumen = (
    df.groupby("area_type")
    .agg(
        registros=("area_type", "size"),
        snr_media=("snr", "mean"),
        snr_mediana=("snr", "median"),
        potencia_media=("iq_power_db", "mean"),
        potencia_mediana=("iq_power_db", "median"),
        power_range_media=("power_range_db", "mean"),
        cfo_medio=("cfo", "mean"),
        velocidad_objetivo=("target_velocity", "mean"),
    )
    .sort_values("snr_media", ascending=False)
)

print(resumen.round(4))

resumen.to_csv(
    RESULTS_DIR / "resumen_por_escenario.csv"
)




print("\n" + "=" * 70)
print("19. RESUMEN POR RRH")
print("=" * 70)

resumen_rrh = (
    df.groupby("rrh_idx")
    .agg(
        registros=("rrh_idx", "size"),
        snr_media=("snr", "mean"),
        snr_mediana=("snr", "median"),
        potencia_media=("iq_power_db", "mean"),
        potencia_mediana=("iq_power_db", "median"),
        power_range_media=("power_range_db", "mean"),
        cfo_medio=("cfo", "mean"),
    )
)

print(resumen_rrh.round(4))

resumen_rrh.to_csv(
    RESULTS_DIR / "resumen_por_rrh.csv"
)


print("\n" + "=" * 70)
print("ANÁLISIS FINALIZADO")
print("=" * 70)

print(f"\nResultados guardados en:")
print(RESULTS_DIR)