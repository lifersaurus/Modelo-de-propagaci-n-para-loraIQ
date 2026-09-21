from pathlib import Path
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats as sps

warnings.filterwarnings("ignore")

BASE_DIR = Path(r"C:\Users\Lenovo\OneDrive\Documentos\LORAIQ")
INPUT = BASE_DIR / "data" / "processed" / "dataset_xgboost.parquet"
MIXED_PREDICTIONS = (
    BASE_DIR / "results" / "model_analysis"
    / "modelo_mixto_principal_residuos.parquet"
)
RESULTS_DIR = BASE_DIR / "results" / "model_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 70)
print("XGBOOST — SNR ESPACIAL + CATEGÓRICAS")
print("=" * 70)
print(f"INPUT: {INPUT.name}")

if not INPUT.exists():
    raise FileNotFoundError(f"No existe:\n{INPUT}")

df = pd.read_parquet(INPUT)
print(f"\nArchivo: {INPUT}")
print(f"Filas: {len(df):,}")

TARGET = "snr"
GROUP = "transmission_idx"
SPATIAL = ["latitude", "longitude"]

required = [TARGET, GROUP, "area_type", "rrh_idx"] + SPATIAL
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError("Faltan columnas: " + ", ".join(missing))

df = df.dropna(subset=required).copy()
df["area_type"] = df["area_type"].astype("category").cat.remove_unused_categories()
df["rrh_idx"] = df["rrh_idx"].astype("category").cat.remove_unused_categories()

print(f"Filas válidas: {len(df):,}")
print(f"Escenarios: {df['area_type'].value_counts().to_dict()}")
print(f"Transmisiones: {df[GROUP].nunique():,}")


FEATURES = SPATIAL + ["area_type", "rrh_idx"]
X = df[FEATURES].copy()
y = df[TARGET].to_numpy()
groups = df[GROUP].to_numpy()

print(f"\nX: {X.shape}")
print(f"Features: {FEATURES}")
print(f"Tipos:\n{X.dtypes.to_string()}")


def build_model(random_state=RANDOM_STATE):
    return xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=random_state,
        n_jobs=-1,
    )


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    if np.var(y_true) == 0:
        return np.nan
    return r2_score(y_true, y_pred)


def safe_mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-6
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(
        (y_true[mask] - y_pred[mask]) / y_true[mask]
    )) * 100)


print("\n" + "=" * 70)
print("AJUSTE GLOBAL (informativo)")
print("=" * 70)

t0 = time.time()
xgb_full = build_model()
xgb_full.fit(X, y)
print(f"Entrenado en {time.time() - t0:.1f} s")

y_pred_full = xgb_full.predict(X)
rmse_train = np.sqrt(mean_squared_error(y, y_pred_full))
mae_train = mean_absolute_error(y, y_pred_full)
mse_train = mean_squared_error(y, y_pred_full)
mape_train = safe_mape(y, y_pred_full)
r2_train = safe_r2(y, y_pred_full)

print(f"RMSE ajuste: {rmse_train:.4f} dB")
print(f"MAE  ajuste: {mae_train:.4f} dB")
print(f"MSE  ajuste: {mse_train:.4f} dB²")
print(f"MAPE ajuste: {mape_train:.4f} %")
print(f"R²   ajuste: {r2_train:.4f}")

print("\nImportancia de features:")
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "gain": xgb_full.feature_importances_,
}).sort_values("gain", ascending=False)
print(importance_df.to_string(index=False))


print("\n" + "=" * 70)
print(f"CV AGRUPADA POR TRANSMISIÓN ({N_SPLITS} folds)")
print("=" * 70)

gkf = GroupKFold(n_splits=N_SPLITS)
rmse_scores, mae_scores, r2_scores = [], [], []
fold_predictions = []
failed_folds = []

for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
    t0 = time.time()
    try:
        model_fold = build_model(random_state=RANDOM_STATE + fold)
        model_fold.fit(X.iloc[tr], y[tr])
        pred = model_fold.predict(X.iloc[te])
    except Exception as e:
        print(f"Fold {fold}: ERROR → {e}")
        failed_folds.append(fold)
        continue

    rmse = np.sqrt(mean_squared_error(y[te], pred))
    mae = mean_absolute_error(y[te], pred)
    r2 = safe_r2(y[te], pred)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

    fold_predictions.append(pd.DataFrame({
        "original_index": df.index[te],
        "fold": fold,
        "area_type": df["area_type"].iloc[te].values,
        "snr_real": y[te],
        "snr_pred": pred,
    }))

    print(f"Fold {fold}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
          f"({time.time() - t0:.1f} s)")

print("\n" + "=" * 70)
print("RESULTADOS CV (por transmisión)")
print("=" * 70)
if rmse_scores:
    print(f"RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f} dB")
    print(f"MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f} dB")
    print(f"R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
else:
    print("Ningún fold se completó correctamente.")

if failed_folds:
    print(f"\nFolds fallidos: {failed_folds}")


print("\n" + "=" * 70)
print("MÉTRICAS POR ESCENARIO (CV)")
print("=" * 70)

all_pred = pd.concat(fold_predictions, ignore_index=True)
for scenario, g in all_pred.groupby("area_type"):
    rmse_s = np.sqrt(mean_squared_error(g["snr_real"], g["snr_pred"]))
    mae_s = mean_absolute_error(g["snr_real"], g["snr_pred"])
    r2_s = safe_r2(g["snr_real"].values, g["snr_pred"].values)
    print(f"  {scenario:<25} n={len(g):>6}  "
          f"RMSE={rmse_s:>7.4f}  MAE={mae_s:>7.4f}  R²={r2_s:>7.4f}")


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
spatial_predictions = []

if n_cells >= N_SPLITS:
    sgkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, te) in enumerate(
        sgkf.split(X, y, groups=spatial_groups), 1
    ):
        t0 = time.time()
        try:
            model_sp = build_model(random_state=200 + fold)
            model_sp.fit(X.iloc[tr], y[tr])
            pred = model_sp.predict(X.iloc[te])
        except Exception as e:
            print(f"Fold {fold}: ERROR → {e}")
            continue

        rmse = np.sqrt(mean_squared_error(y[te], pred))
        mae = mean_absolute_error(y[te], pred)
        r2 = safe_r2(y[te], pred)

        spatial_rmse.append(rmse)
        spatial_mae.append(mae)
        spatial_r2.append(r2)

        spatial_predictions.append(pd.DataFrame({
            "original_index": df.index[te],
            "fold": fold,
            "area_type": df["area_type"].iloc[te].values,
            "latitude": df["latitude"].iloc[te].values,
            "longitude": df["longitude"].iloc[te].values,
            "snr_real": y[te],
            "snr_pred": pred,
        }))

        print(f"Fold {fold}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
              f"({time.time() - t0:.1f} s)")

    if spatial_rmse:
        print(f"\nRMSE espacial: {np.mean(spatial_rmse):.4f} ± {np.std(spatial_rmse):.4f} dB")
        print(f"MAE  espacial: {np.mean(spatial_mae):.4f} ± {np.std(spatial_mae):.4f} dB")
        print(f"R²   espacial: {np.mean(spatial_r2):.4f} ± {np.std(spatial_r2):.4f}")
else:
    print("No hay suficientes celdas para CV espacial.")


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

if MIXED_PREDICTIONS.exists() and rmse_scores:
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
        mixed_r2 = safe_r2(
            mixed_df[mixed_target].values, mixed_df[mixed_pred].values)

        print(f"\nModelo Mixto (referencia, ajuste in-sample):")
        print(f"  n = {len(mixed_df):,}")
        print(f"  RMSE = {mixed_rmse:.4f} dB")
        print(f"  MAE  = {mixed_mae:.4f} dB")
        print(f"  R²   = {mixed_r2:.4f}")

        print(f"\nModelo XGBoost (CV por transmisión):")
        print(f"  n = {len(all_pred):,}")
        print(f"  RMSE = {np.mean(rmse_scores):.4f} dB")
        print(f"  MAE  = {np.mean(mae_scores):.4f} dB")
        print(f"  R²   = {np.mean(r2_scores):.4f}")

        delta_rmse = np.mean(rmse_scores) - mixed_rmse
        delta_mae = np.mean(mae_scores) - mixed_mae
        delta_r2 = np.mean(r2_scores) - mixed_r2

        print(f"\nDiferencia (Δ = XGBoost − Mixto):")
        print(f"  ΔRMSE = {delta_rmse:+.4f} dB   (negativo = XGBoost mejor)")
        print(f"  ΔMAE  = {delta_mae:+.4f} dB   (negativo = XGBoost mejor)")
        print(f"  ΔR²   = {delta_r2:+.4f}      (positivo = XGBoost mejor)")

        print(
            "\n⚠ NOTA: comparación no estricta — Mixto reporta ajuste, "
            "XGBoost reporta CV. Para comparación justa, "
            "revisar modelo_mixto_principal_resumen.txt."
        )

        comp = pd.DataFrame([
            {"Modelo": "Mixto", "n": len(mixed_df),
             "RMSE": round(mixed_rmse, 4),
             "MAE": round(mixed_mae, 4),
             "R²": round(mixed_r2, 4),
             "Evaluación": "Ajuste (in-sample)"},
            {"Modelo": "XGBoost", "n": len(all_pred),
             "RMSE": round(np.mean(rmse_scores), 4),
             "MAE": round(np.mean(mae_scores), 4),
             "R²": round(np.mean(r2_scores), 4),
             "Evaluación": "CV por transmisión"},
        ])
        comp_path = RESULTS_DIR / "comparacion_mixto_vs_xgboost.csv"
        comp.to_csv(comp_path, index=False)
        print(f"\n  ✔ {comp_path.name}")
    else:
        print("No se pudieron detectar columnas en el archivo del mixto.")
else:
    print(f"No existe: {MIXED_PREDICTIONS}")
    print("Ejecuta primero modelo_mixto.py si quieres la comparación.")


all_pred.to_parquet(
    RESULTS_DIR / "xgboost_predicciones.parquet", index=False
)

pd.DataFrame({
    "fold": np.arange(1, len(rmse_scores) + 1),
    "rmse": rmse_scores,
    "mae": mae_scores,
    "r2": r2_scores,
}).to_csv(
    RESULTS_DIR / "xgboost_validacion_transmission.csv", index=False
)

pd.DataFrame({
    "metric": ["RMSE", "MAE", "R2"],
    "value": [
        np.mean(spatial_rmse) if spatial_rmse else np.nan,
        np.mean(spatial_mae) if spatial_mae else np.nan,
        np.mean(spatial_r2) if spatial_r2 else np.nan,
    ],
}).to_csv(
    RESULTS_DIR / "xgboost_validacion_espacial.csv", index=False
)

importance_df.to_csv(
    RESULTS_DIR / "xgboost_importancia_features.csv", index=False
)

if spatial_predictions:
    pd.concat(spatial_predictions, ignore_index=True).to_parquet(
        RESULTS_DIR / "xgboost_predicciones_espaciales.parquet", index=False
    )

with open(RESULTS_DIR / "xgboost_modelo.pkl", "wb") as fp:
    pickle.dump({"xgb": xgb_full, "features": FEATURES}, fp)

with open(RESULTS_DIR / "xgboost_resumen.txt", "w", encoding="utf-8") as fp:
    fp.write("XGBOOST — SNR ESPACIAL\n" + "=" * 70 + "\n\n")
    fp.write(f"Input: {INPUT.name}\n")
    fp.write(f"Observaciones: {len(df):,}\n")
    fp.write(f"Transmisiones: {df[GROUP].nunique():,}\n")
    fp.write(f"Posiciones: {df[SPATIAL].drop_duplicates().shape[0]:,}\n\n")
    fp.write(f"Features: {FEATURES}\n\n")
    fp.write(f"RMSE ajuste: {rmse_train:.4f} dB\n")
    fp.write(f"MAE  ajuste: {mae_train:.4f} dB\n")
    fp.write(f"MSE  ajuste: {mse_train:.4f} dB²\n")
    fp.write(f"MAPE ajuste: {mape_train:.4f} %\n")
    fp.write(f"R²   ajuste: {r2_train:.4f}\n\n")
    fp.write("CV por transmisión:\n")
    if rmse_scores:
        fp.write(f"  RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f} dB\n")
        fp.write(f"  MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f} dB\n")
        fp.write(f"  R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}\n\n")
    fp.write("CV espacial:\n")
    if spatial_rmse:
        fp.write(f"  RMSE: {np.mean(spatial_rmse):.4f} dB\n")
        fp.write(f"  MAE:  {np.mean(spatial_mae):.4f} dB\n")
        fp.write(f"  R²:   {np.mean(spatial_r2):.4f}\n\n")
    fp.write("IMPORTANCIA DE FEATURES\n")
    fp.write(importance_df.to_string(index=False))

print("\nArchivos guardados en:", RESULTS_DIR)
print("Listo.")