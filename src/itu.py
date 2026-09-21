from pathlib import Path
import pickle
import warnings
import time

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats as sps

warnings.filterwarnings("ignore")


BASE_DIR = Path(r"C:\Users\Lenovo\OneDrive\Documentos\LORAIQ")
INPUT = BASE_DIR / "data" / "processed" / "dataset_modelo.parquet"
MIXED_PREDICTIONS = (
    BASE_DIR / "results" / "model_analysis"
    / "modelo_mixto_principal_residuos.parquet"
)
RESULTS_DIR = BASE_DIR / "results" / "model_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


FREQ_MHZ = 862.5
EARTH_RADIUS_KM = 6371.0
E_VEG = 0.01
MONTH = 1
D_VEG_MIN_M = 1.0

SCENARIO_GEOMETRY = {
    "drone_los":              (0.10, 45.0),
    "drone_nlos":             (0.30, 30.0),
    "pedestrian_nlos":        (0.50, 10.0),
    "pedestrian_partial_los": (0.40, 15.0),
    "indoor":                 (0.60,  5.0),
}


def compute_B(freq_mhz, month):
    kh = 6.0 - abs(month - 6.5)
    exponent = 0.0013118 - 0.026236 * kh
    B = (0.30281 - 0.003624 * kh) * (freq_mhz / 1000.0) ** exponent
    return B


B_FIXED = compute_B(FREQ_MHZ, MONTH)
K_F = FREQ_MHZ ** B_FIXED

print("=" * 70)
print("ITU-R P.833-10 ADAPTADO — SNR ESPACIAL (5 escenarios)")
print("=" * 70)
print(f"INPUT:                {INPUT.name}")
print(f"Frecuencia:           {FREQ_MHZ} MHz")
print(f"Mes:                  {MONTH}")
print(f"B (determinístico):   {B_FIXED:.6f}")
print(f"K_f = f^B:            {K_F:.4f}")
print(f"E (vegetación):       {E_VEG}")
print(f"N_SPLITS:             {N_SPLITS}")


if not INPUT.exists():
    raise FileNotFoundError(f"No existe:\n{INPUT}")

df = pd.read_parquet(INPUT)
print(f"\nFilas: {len(df):,}")
print(f"Columnas: {list(df.columns)[:10]}...")

TARGET = "snr"
GROUP = "transmission_idx"
SPATIAL = ["latitude", "longitude"]
SCENARIO = "area_type"
RRH = "rrh_idx"

required = [TARGET, GROUP, SCENARIO, RRH] + SPATIAL
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError("Faltan columnas: " + ", ".join(missing))

df = df.dropna(subset=required).copy()

for col in [SCENARIO, RRH]:
    df[col] = df[col].astype("category").cat.remove_unused_categories()

print(f"Filas válidas: {len(df):,}")
print(f"Escenarios: {df[SCENARIO].value_counts().to_dict()}")
print(f"Transmisiones: {df[GROUP].nunique():,}")


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


def estimate_rrh_positions(df_in):
    winner = (
        df_in
        .sort_values(TARGET, ascending=False)
        .drop_duplicates(subset=[GROUP], keep="first")
    )
    return (
        winner
        .groupby(RRH, observed=True)
        .agg(
            rrh_lat=("latitude", "mean"),
            rrh_lon=("longitude", "mean"),
            n_wins=("latitude", "size"),
        )
        .reset_index()
    )


def itu_p833_predict(X, alpha, A, G):
    d_km, d_veg_m, theta_deg = X

    fspl = 20.0 * np.log10(np.maximum(d_km, 1e-3))

    L_veg = (
        A * K_F
        * np.log10(np.maximum(d_veg_m, D_VEG_MIN_M))
        * (theta_deg + E_VEG) ** G
        - 4.0
    )

    return alpha - fspl - L_veg


def _prepare_geometry(df_in):
    frac = df_in[SCENARIO].astype(str).map(
        {k: v[0] for k, v in SCENARIO_GEOMETRY.items()}
    ).astype(float).fillna(0.3).values

    theta = df_in[SCENARIO].astype(str).map(
        {k: v[1] for k, v in SCENARIO_GEOMETRY.items()}
    ).astype(float).fillna(20.0).values

    d_km = df_in["distance_km"].values
    d_veg = np.maximum(frac * d_km * 1000.0, D_VEG_MIN_M)

    return d_km, d_veg, theta


print("\n" + "=" * 70)
print("ESTIMACIÓN DE POSICIONES DE RRH")
print("=" * 70)

rrh_positions = estimate_rrh_positions(df)
print(rrh_positions.to_string(index=False))

df = df.merge(
    rrh_positions[[RRH, "rrh_lat", "rrh_lon", "n_wins"]],
    on=RRH, how="left",
)

df["distance_km"] = haversine_km(
    df["latitude"].to_numpy(),
    df["longitude"].to_numpy(),
    df["rrh_lat"].to_numpy(),
    df["rrh_lon"].to_numpy(),
)

df = df[df["distance_km"] > 1e-3].copy()

print(f"\nDistancias Tx–RRH calculadas: {len(df):,}")
print(f"  min: {df['distance_km'].min():.4f} km")
print(f"  med: {df['distance_km'].median():.4f} km")
print(f"  max: {df['distance_km'].max():.4f} km")


X_dkm, X_dveg, X_theta = _prepare_geometry(df)
y = df[TARGET].to_numpy(dtype=float)
groups = df[GROUP].to_numpy()

X_full = np.vstack([X_dkm, X_dveg, X_theta])


print("\n" + "=" * 70)
print("AJUSTE GLOBAL NLLS (Levenberg-Marquardt)")
print("=" * 70)

p0 = [30.0, 1.0, 0.5]
bounds = (
    [-200.0,   0.0, 0.001],
    [ 200.0, 100.0, 5.000],
)

t0 = time.time()
popt, pcov = curve_fit(
    itu_p833_predict,
    X_full, y,
    p0=p0,
    bounds=bounds,
    method="trf",
    maxfev=50000,
)

alpha_hat, A_hat, G_hat = popt
perr = np.sqrt(np.diag(pcov))

print(f"Entrenado en {time.time() - t0:.2f} s")
print(f"\nParámetros óptimos:")
print(f"  α = {alpha_hat:.6f} ± {perr[0]:.6f}")
print(f"  A = {A_hat:.6f} ± {perr[1]:.6f}")
print(f"  G = {G_hat:.6f} ± {perr[2]:.6f}")

y_pred_full = itu_p833_predict(X_full, *popt)

print(f"\nMétricas en ajuste (n={len(y):,}):")
print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred_full)):.4f} dB")
print(f"  MAE:  {mean_absolute_error(y, y_pred_full):.4f} dB")
print(f"  R²:   {r2_score(y, y_pred_full):.4f}")


print("\n" + "=" * 70)
print(f"CV AGRUPADA POR TRANSMISIÓN ({N_SPLITS} folds)")
print("=" * 70)

gkf = GroupKFold(n_splits=N_SPLITS)
rmse_scores, mae_scores, r2_scores = [], [], []
fold_params = []
fold_predictions = []

for fold, (tr, te) in enumerate(
    gkf.split(X_full.T, y, groups=groups), 1
):
    t0 = time.time()

    X_tr = X_full[:, tr]
    X_te = X_full[:, te]
    y_tr, y_te = y[tr], y[te]

    try:
        popt_fold, _ = curve_fit(
            itu_p833_predict,
            X_tr, y_tr,
            p0=p0,
            bounds=bounds,
            method="trf",
            maxfev=50000,
        )
        pred = itu_p833_predict(X_te, *popt_fold)
    except Exception as e:
        print(f"Fold {fold}: error → {e}")
        continue

    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    fold_params.append({
        "fold": fold,
        "alpha": popt_fold[0],
        "A": popt_fold[1],
        "G": popt_fold[2],
    })

    fold_predictions.append(pd.DataFrame({
        "original_index": df.index[te],
        "fold": fold,
        "area_type": df[SCENARIO].iloc[te].values,
        "snr_real": y_te,
        "snr_pred": pred,
        "distance_km": df["distance_km"].iloc[te].values,
    }))

    print(
        f"Fold {fold}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
        f"(α={popt_fold[0]:.2f}, A={popt_fold[1]:.3f}, G={popt_fold[2]:.3f})  "
        f"({time.time()-t0:.2f} s)"
    )

print("\n" + "=" * 70)
print("RESULTADOS CV")
print("=" * 70)
print(f"RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f} dB")
print(f"MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f} dB")
print(f"R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")


print("\n" + "=" * 70)
print("MÉTRICAS POR ESCENARIO (CV)")
print("=" * 70)

all_pred = pd.concat(fold_predictions, ignore_index=True)
for scenario, g in all_pred.groupby("area_type"):
    rmse_s = np.sqrt(mean_squared_error(g["snr_real"], g["snr_pred"]))
    mae_s = mean_absolute_error(g["snr_real"], g["snr_pred"])
    r2_s = r2_score(g["snr_real"], g["snr_pred"]) if len(g) > 1 else np.nan
    print(
        f"  {scenario:<25} n={len(g):>6}  "
        f"RMSE={rmse_s:>7.4f}  MAE={mae_s:>7.4f}  R²={r2_s:>7.4f}"
    )


print("\n" + "=" * 70)
print("CV ESPACIAL (agrupada por celda lat/lon)")
print("=" * 70)

df["spatial_cell"] = (
    df["latitude"].round(4).astype(str) + "_"
    + df["longitude"].round(4).astype(str)
)
spatial_groups = df["spatial_cell"].to_numpy()
n_cells = df["spatial_cell"].nunique()
print(f"Celdas espaciales: {n_cells:,}")

spatial_rmse, spatial_mae, spatial_r2 = [], [], []

if n_cells >= N_SPLITS:
    sgkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, te) in enumerate(
        sgkf.split(X_full.T, y, groups=spatial_groups), 1
    ):
        t0 = time.time()

        try:
            popt_sp, _ = curve_fit(
                itu_p833_predict,
                X_full[:, tr], y[tr],
                p0=p0,
                bounds=bounds,
                method="trf",
                maxfev=50000,
            )
            pred = itu_p833_predict(X_full[:, te], *popt_sp)
        except Exception as e:
            print(f"Fold {fold}: error → {e}")
            continue

        rmse = np.sqrt(mean_squared_error(y[te], pred))
        mae = mean_absolute_error(y[te], pred)
        r2 = r2_score(y[te], pred)

        spatial_rmse.append(rmse)
        spatial_mae.append(mae)
        spatial_r2.append(r2)

        print(
            f"Fold {fold}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
            f"({time.time()-t0:.2f} s)"
        )

    print(
        f"\nRMSE espacial: {np.mean(spatial_rmse):.4f} "
        f"± {np.std(spatial_rmse):.4f} dB"
    )
    print(
        f"MAE  espacial: {np.mean(spatial_mae):.4f} "
        f"± {np.std(spatial_mae):.4f} dB"
    )
    print(
        f"R²   espacial: {np.mean(spatial_r2):.4f} "
        f"± {np.std(spatial_r2):.4f}"
    )
else:
    print("No hay suficientes celdas para CV espacial.")
    spatial_rmse = [np.nan]
    spatial_mae = [np.nan]
    spatial_r2 = [np.nan]


residual = all_pred["snr_real"] - all_pred["snr_pred"]

print("\n" + "=" * 70)
print("DIAGNÓSTICO DE RESIDUOS (CV)")
print("=" * 70)

sw = sps.shapiro(residual[:5000])
print(f"Shapiro-Wilk: W={sw.statistic:.4f}, p={sw.pvalue:.4e}")
print(f"Asimetría: {sps.skew(residual):.4f}")
print(f"Curtosis:  {sps.kurtosis(residual):.4f}")


print("\n" + "=" * 70)
print("COMPARACIÓN CON MODELO MIXTO")
print("=" * 70)

if MIXED_PREDICTIONS.exists():
    mixed_df = pd.read_parquet(MIXED_PREDICTIONS)

    mixed_target = next(
        (c for c in ["snr", "snr_real", "y_true"] if c in mixed_df.columns),
        None,
    )
    mixed_pred = next(
        (c for c in ["snr_pred", "pred", "y_pred"] if c in mixed_df.columns),
        None,
    )

    if mixed_target and mixed_pred:
        mixed_rmse = np.sqrt(mean_squared_error(
            mixed_df[mixed_target], mixed_df[mixed_pred]))
        mixed_mae = mean_absolute_error(
            mixed_df[mixed_target], mixed_df[mixed_pred])
        mixed_r2 = r2_score(
            mixed_df[mixed_target], mixed_df[mixed_pred])

        print(f"\nModelo Mixto (referencia):")
        print(f"  n = {len(mixed_df):,}")
        print(f"  RMSE = {mixed_rmse:.4f} dB")
        print(f"  MAE  = {mixed_mae:.4f} dB")
        print(f"  R²   = {mixed_r2:.4f}")

        print(f"\nModelo ITU-R P.833-10 adaptado:")
        print(f"  n = {len(all_pred):,}")
        print(f"  RMSE = {np.mean(rmse_scores):.4f} dB")
        print(f"  MAE  = {np.mean(mae_scores):.4f} dB")
        print(f"  R²   = {np.mean(r2_scores):.4f}")

        print(f"\nComparación (Δ = ITU − Mixto):")
        print(f"  ΔRMSE = {np.mean(rmse_scores) - mixed_rmse:+.4f} dB")
        print(f"  ΔMAE  = {np.mean(mae_scores) - mixed_mae:+.4f} dB")
        print(f"  ΔR²   = {np.mean(r2_scores) - mixed_r2:+.4f}")

        comp = pd.DataFrame([
            {"Modelo": "Mixto", "n": len(mixed_df),
             "RMSE": round(mixed_rmse, 4),
             "MAE": round(mixed_mae, 4),
             "R²": round(mixed_r2, 4)},
            {"Modelo": "ITU-P833", "n": len(all_pred),
             "RMSE": round(np.mean(rmse_scores), 4),
             "MAE": round(np.mean(mae_scores), 4),
             "R²": round(np.mean(r2_scores), 4)},
        ])
        comp_path = RESULTS_DIR / "comparacion_mixto_vs_itu.csv"
        comp.to_csv(comp_path, index=False)
        print(f"\n  ✔ {comp_path.name}")
    else:
        print("No se pudieron detectar columnas en el archivo del mixto.")
else:
    print(f"No existe: {MIXED_PREDICTIONS}")
    print("Ejecuta primero modelo_mixto.py si quieres la comparación.")


all_pred.to_parquet(
    RESULTS_DIR / "itu_predicciones.parquet", index=False
)

pd.DataFrame({
    "fold": np.arange(1, len(rmse_scores) + 1),
    "rmse": rmse_scores,
    "mae": mae_scores,
    "r2": r2_scores,
}).to_csv(
    RESULTS_DIR / "itu_validacion_transmission.csv", index=False
)

pd.DataFrame({
    "metric": ["RMSE", "MAE", "R2"],
    "value": [
        np.mean(spatial_rmse),
        np.mean(spatial_mae),
        np.mean(spatial_r2),
    ],
}).to_csv(
    RESULTS_DIR / "itu_validacion_espacial.csv", index=False
)

pd.DataFrame(fold_params).to_csv(
    RESULTS_DIR / "itu_parametros_por_fold.csv", index=False
)

rrh_positions.to_csv(
    RESULTS_DIR / "itu_rrh_positions_estimadas.csv", index=False
)

with open(RESULTS_DIR / "itu_modelo.pkl", "wb") as fp:
    pickle.dump({
        "alpha": alpha_hat,
        "A": A_hat,
        "G": G_hat,
        "B_fixed": B_FIXED,
        "K_f": K_F,
        "E_veg": E_VEG,
        "freq_mhz": FREQ_MHZ,
        "scenario_geometry": SCENARIO_GEOMETRY,
        "rrh_positions": rrh_positions.to_dict("records"),
    }, fp)

with open(RESULTS_DIR / "itu_resumen.txt", "w", encoding="utf-8") as fp:
    fp.write("ITU-R P.833-10 ADAPTADO — SNR ESPACIAL\n" + "=" * 70 + "\n\n")
    fp.write(f"Input: {INPUT.name}\n")
    fp.write(f"Frecuencia: {FREQ_MHZ} MHz\n")
    fp.write(f"Mes: {MONTH}\n")
    fp.write(f"B (determinístico): {B_FIXED:.6f}\n")
    fp.write(f"K_f = f^B: {K_F:.4f}\n")
    fp.write(f"E (vegetación): {E_VEG}\n\n")
    fp.write(f"Observaciones: {len(df):,}\n")
    fp.write(f"Transmisiones: {df[GROUP].nunique():,}\n\n")
    fp.write(f"Parámetros globales:\n")
    fp.write(f"  α = {alpha_hat:.6f} ± {perr[0]:.6f}\n")
    fp.write(f"  A = {A_hat:.6f} ± {perr[1]:.6f}\n")
    fp.write(f"  G = {G_hat:.6f} ± {perr[2]:.6f}\n\n")
    fp.write("CV por transmisión:\n")
    fp.write(f"  RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f} dB\n")
    fp.write(f"  MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f} dB\n")
    fp.write(f"  R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}\n\n")
    fp.write("CV espacial:\n")
    fp.write(f"  RMSE: {np.mean(spatial_rmse):.4f} dB\n")
    fp.write(f"  MAE:  {np.mean(spatial_mae):.4f} dB\n")
    fp.write(f"  R²:   {np.mean(spatial_r2):.4f}\n")

print("\nArchivos guardados en:", RESULTS_DIR)
print("Listo.")