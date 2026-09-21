from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(r"C:\Users\Lenovo\OneDrive\Documentos\LORAIQ")
INPUT = BASE_DIR / "data" / "processed" / "lora_features.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "model_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "snr"
GROUP = "transmission_idx"

REDUNDANT_WITH_SCENARIO = ["sf", "bandwidth", "cr", "target_velocity"]
VELOCITY = "velocity"
SPATIAL = ["latitude", "longitude"]
RRH = "rrh_idx"
SCENARIO = "area_type"

IDENTIFIERS = ["file", "datetime", GROUP, RRH]

FREQ_MHZ = 862.5
EARTH_RADIUS_KM = 6371.0

ITU_ENV_LOSS_DB = {
    "drone_los": 0.0,
    "pedestrian_partial_los": 6.0,
    "pedestrian_nlos": 10.0,
    "drone_nlos": 15.0,
    "indoor": 20.0,
}


print("=" * 70)
print("PREPARACIÓN DEL DATASET PARA MODELADO")
print("=" * 70)

if not INPUT.exists():
    raise FileNotFoundError(f"No existe:\n{INPUT}")

df = pd.read_parquet(INPUT)
print(f"\nArchivo: {INPUT}")
print(f"Filas originales: {len(df):,}")
print(f"Columnas originales: {len(df.columns)}")


required = [TARGET, SCENARIO, RRH, GROUP, VELOCITY] + SPATIAL + IDENTIFIERS
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError("Faltan columnas:\n" + "\n".join(f" - {c}" for c in missing))


print("\n" + "=" * 70)
print("DIAGNÓSTICO DE REDUNDANCIA")
print("=" * 70)

for col in REDUNDANT_WITH_SCENARIO:
    if col in df.columns:
        ct = pd.crosstab(df[SCENARIO], df[col])
        print(f"\n{SCENARIO} × {col}:")
        print(ct)


if SCENARIO in df.columns:
    df[SCENARIO] = df[SCENARIO].astype("category").cat.remove_unused_categories()
if RRH in df.columns:
    df[RRH] = df[RRH].astype("category").cat.remove_unused_categories()

for col in [TARGET, VELOCITY, GROUP]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in SPATIAL:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")


before = len(df)
df = df.dropna(subset=[TARGET]).copy()
print(f"\nFilas eliminadas por SNR faltante: {before - len(df):,}")
print(f"Filas restantes: {len(df):,}")

dup = df.duplicated().sum()
if dup > 0:
    df = df.drop_duplicates().copy()
    print(f"Duplicados eliminados: {dup:,}")


master_columns = list(dict.fromkeys(
    IDENTIFIERS + [TARGET, SCENARIO] + [VELOCITY]
    + REDUNDANT_WITH_SCENARIO + SPATIAL
))
master_columns = [c for c in master_columns if c in df.columns]

model_df = df[master_columns].copy()

master_output = OUTPUT_DIR / "dataset_modelo.parquet"
model_df.to_parquet(master_output, index=False)

print("\n" + "=" * 70)
print("DATASET MAESTRO")
print("=" * 70)
print(f"Filas: {len(model_df):,}")
print(f"Escenarios: {model_df[SCENARIO].value_counts().to_dict()}")


mixed_main_columns = [TARGET, SCENARIO, RRH, GROUP]
mixed_main = model_df[mixed_main_columns].dropna().copy()

mixed_main_output = OUTPUT_DIR / "dataset_mixto.parquet"
mixed_main.to_parquet(mixed_main_output, index=False)

print("\n" + "=" * 70)
print("MODELO MIXTO PRINCIPAL")
print("=" * 70)
print(f"Filas: {len(mixed_main):,}")
print(f"Escenarios: {mixed_main[SCENARIO].value_counts().to_dict()}")
print(f"Transmisiones únicas: {mixed_main[GROUP].nunique():,}")
print(f"RRH: {mixed_main[RRH].nunique()}")


mixed_vel_columns = [TARGET, SCENARIO, RRH, VELOCITY, GROUP]
mixed_vel = model_df[mixed_vel_columns].dropna().copy()

mixed_vel_output = OUTPUT_DIR / "dataset_mixto_velocity.parquet"
mixed_vel.to_parquet(mixed_vel_output, index=False)

print("\n" + "=" * 70)
print("MODELO MIXTO CON VELOCIDAD (subconjunto)")
print("=" * 70)
print(f"Filas: {len(mixed_vel):,}")
print(f"Escenarios: {mixed_vel[SCENARIO].value_counts().to_dict()}")
print(f"Transmisiones únicas: {mixed_vel[GROUP].nunique():,}")


gpr_columns = [TARGET, SCENARIO, RRH, GROUP] + SPATIAL + [VELOCITY]
gpr_columns = [c for c in gpr_columns if c in model_df.columns]

gpr_df = model_df[gpr_columns].dropna(subset=[TARGET] + SPATIAL).copy()

gpr_output = OUTPUT_DIR / "dataset_gpr.parquet"
gpr_df.to_parquet(gpr_output, index=False)

print("\n" + "=" * 70)
print("GPR")
print("=" * 70)
print(f"Filas: {len(gpr_df):,}")
print(f"Escenarios: {gpr_df[SCENARIO].value_counts().to_dict()}")
print(f"Posiciones únicas: {gpr_df[SPATIAL].drop_duplicates().shape[0]:,}")
print("\nPosiciones únicas por escenario:")
print(
    gpr_df.groupby(SCENARIO, observed=True)[SPATIAL]
    .apply(lambda g: g.drop_duplicates().shape[0])
)


xgb_columns = [TARGET, SCENARIO, RRH, GROUP] + SPATIAL + [VELOCITY]
xgb_columns = [c for c in xgb_columns if c in model_df.columns]

xgb_df = model_df[xgb_columns].dropna(subset=[TARGET]).copy()

xgb_output = OUTPUT_DIR / "dataset_xgboost.parquet"
xgb_df.to_parquet(xgb_output, index=False)

print("\n" + "=" * 70)
print("XGBOOST")
print("=" * 70)
print(f"Filas: {len(xgb_df):,}")
print(f"Escenarios: {xgb_df[SCENARIO].value_counts().to_dict()}")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def estimate_rrh_positions(df_spatial, rrh_col="rrh_idx",
                            snr_col="snr", lat_col="latitude",
                            lon_col="longitude"):
    """
    Estima la posición de cada RRH como el centroide ponderado por SNR
    de los puntos donde ese RRH tiene la señal más fuerte.

    Estrategia:
      1. Para cada transmisión, identificar el RRH con mayor SNR.
      2. Acumular las posiciones Tx de esos puntos por RRH.
      3. Calcular centroide (media de lat/lon) por RRH.
    """
    winner_idx = (
        df_spatial
        .sort_values(snr_col, ascending=False)
        .drop_duplicates(subset=["transmission_idx"], keep="first")
    )

    positions = (
        winner_idx
        .groupby(rrh_col, observed=True)
        .agg(
            rrh_lat=(lat_col, "mean"),
            rrh_lon=(lon_col, "mean"),
            n_obs=(lat_col, "size"),
        )
        .reset_index()
    )
    return positions


print("\n" + "=" * 70)
print("ITU — ESTIMACIÓN DE POSICIONES DE RRH")
print("=" * 70)

itu_columns = [TARGET, SCENARIO, RRH, GROUP] + SPATIAL
itu_df = model_df[itu_columns].dropna(subset=[TARGET] + SPATIAL).copy()
itu_df["transmission_idx"] = itu_df[GROUP]

rrh_positions = estimate_rrh_positions(
    itu_df,
    rrh_col=RRH,
    snr_col=TARGET,
    lat_col="latitude",
    lon_col="longitude",
)

print("\nPosiciones estimadas de cada RRH:")
print(rrh_positions.to_string(index=False))

rrh_positions.to_csv(
    RESULTS_DIR / "itu_rrh_positions_estimadas.csv",
    index=False,
)

itu_df = itu_df.merge(
    rrh_positions[[RRH, "rrh_lat", "rrh_lon", "n_obs"]],
    on=RRH,
    how="left",
)

itu_df["distance_km"] = haversine_km(
    itu_df["latitude"].to_numpy(),
    itu_df["longitude"].to_numpy(),
    itu_df["rrh_lat"].to_numpy(),
    itu_df["rrh_lon"].to_numpy(),
)

itu_df = itu_df[itu_df["distance_km"] > 1e-3].copy()

freq_term = 20.0 * np.log10(FREQ_MHZ) + 32.45

itu_df["itu_p525_free_space_db"] = (
    20.0 * np.log10(itu_df["distance_km"]) + freq_term
)

itu_df["itu_env_loss_db"] = (
    itu_df[SCENARIO]
    .map(ITU_ENV_LOSS_DB)
    .astype(float)
    .fillna(10.0)
)

itu_df["itu_p1546_total_loss_db"] = (
    itu_df["itu_p525_free_space_db"]
    + itu_df["itu_env_loss_db"]
)

log_d = np.log10(itu_df["distance_km"].to_numpy())
env = itu_df["itu_env_loss_db"].to_numpy()
y = itu_df[TARGET].to_numpy()

alpha_hat = float(y.mean() + 20.0 * log_d.mean() + env.mean())

itu_df["itu_snr_pred"] = (
    alpha_hat
    - 20.0 * log_d
    - env
)

resid = y - itu_df["itu_snr_pred"].to_numpy()
rmse = float(np.sqrt(np.mean(resid ** 2)))
mae = float(np.mean(np.abs(resid)))
ss_res = float(np.sum(resid ** 2))
ss_tot = float(np.sum((y - y.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

print(f"\nCalibración ITU (P.525 + P.1546 simplificado):")
print(f"  α calibrado: {alpha_hat:.4f}")
print(f"  RMSE: {rmse:.4f} dB")
print(f"  MAE:  {mae:.4f} dB")
print(f"  R²:   {r2:.4f}")
print(f"  n:    {len(itu_df):,}")

itu_output_columns = [
    TARGET, SCENARIO, RRH, GROUP,
    "latitude", "longitude",
    "rrh_lat", "rrh_lon", "n_obs",
    "distance_km",
    "itu_p525_free_space_db",
    "itu_env_loss_db",
    "itu_p1546_total_loss_db",
    "itu_snr_pred",
]

itu_output = OUTPUT_DIR / "dataset_itu.parquet"
itu_df[itu_output_columns].to_parquet(itu_output, index=False)

print(f"\n  ✔ {itu_output.name}")


missing_report = pd.DataFrame({
    "variable": model_df.columns,
    "missing_count": [model_df[col].isna().sum() for col in model_df.columns],
})
missing_report["missing_percentage"] = (
    missing_report["missing_count"] / len(model_df) * 100
)
missing_report = missing_report.sort_values(
    "missing_percentage", ascending=False
)

missing_report.to_csv(
    RESULTS_DIR / "faltantes_dataset_modelo.csv", index=False
)

print("\n" + "=" * 70)
print("VALORES FALTANTES (% del total)")
print("=" * 70)
print(missing_report.to_string(index=False))


summary_lines = [
    "RESUMEN DEL DATASET PARA MODELADO",
    "=" * 70,
    f"Filas originales: {len(df):,}",
    f"Filas dataset maestro: {len(model_df):,}",
    f"Escenarios: {model_df[SCENARIO].nunique()}",
    f"Transmisiones únicas: {model_df[GROUP].nunique():,}",
    f"RRH únicos: {model_df[RRH].nunique():,}",
    "",
    "Datasets específicos:",
    f"  mixto principal:          {len(mixed_main):,} filas",
    f"  mixto + velocity:         {len(mixed_vel):,} filas",
    f"  gpr:                      {len(gpr_df):,} filas",
    f"  xgboost:                  {len(xgb_df):,} filas",
    f"  itu:                      {len(itu_df):,} filas",
    "",
    "Modelo ITU:",
    f"  α calibrado:  {alpha_hat:.4f}",
    f"  RMSE:         {rmse:.4f} dB",
    f"  MAE:          {mae:.4f} dB",
    f"  R²:           {r2:.4f}",
    "",
    "Observaciones metodológicas:",
    "- sf, bandwidth, cr y target_velocity son redundantes con area_type.",
    "- Se EXCLUYEN del modelo principal.",
    "- velocity solo existe en el 32% (drone).",
    "- ITU usa distancia Tx-RRH estimada por centroide ponderado por SNR.",
    "- Correcciones ambientales tomadas de ITU-R P.1546.",
]

summary_output = RESULTS_DIR / "resumen_dataset_modelo.txt"
with open(summary_output, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))


print("\n" + "=" * 70)
print("ARCHIVOS GENERADOS")
print("=" * 70)
for p in [
    master_output,
    mixed_main_output,
    mixed_vel_output,
    gpr_output,
    xgb_output,
    itu_output,
    RESULTS_DIR / "faltantes_dataset_modelo.csv",
    RESULTS_DIR / "itu_rrh_positions_estimadas.csv",
    summary_output,
]:
    print(f" - {p}")

print("\nPreparación finalizada correctamente.")