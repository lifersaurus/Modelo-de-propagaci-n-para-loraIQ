
from pathlib import Path
import pickle
import warnings
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import stats as sps

warnings.filterwarnings("ignore")


BASE_DIR = Path(r"C:\Users\Lenovo\OneDrive\Documentos\LORAIQ")

INPUT = BASE_DIR / "data" / "processed" / "dataset_gpr.parquet"

MIXED_PREDICTIONS = (
    BASE_DIR / "results" / "model_analysis"
    / "modelo_mixto_principal_residuos.parquet"
)

RESULTS_DIR = BASE_DIR / "results" / "model_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
MAX_TRAIN_GPR = 3000
RANDOM_STATE = 42
N_RESTARTS_GLOBAL = 2
N_RESTARTS_CV = 0

np.random.seed(RANDOM_STATE)

print("=" * 70)
print("GPR — SNR ESPACIAL (5 escenarios)")
print("=" * 70)
print(f"INPUT:            {INPUT.name}")
print(f"MAX_TRAIN_GPR:    {MAX_TRAIN_GPR}")
print(f"N_RESTARTS_GLOBAL: {N_RESTARTS_GLOBAL}")
print(f"N_RESTARTS_CV:     {N_RESTARTS_CV}")


if not INPUT.exists():
    raise FileNotFoundError(f"No existe:\n{INPUT}")

df = pd.read_parquet(INPUT)
print(f"\nFilas: {len(df):,}")
print(f"Columnas: {list(df.columns)[:10]}...")

TARGET = "snr"
GROUP = "transmission_idx"
SPATIAL = ["latitude", "longitude"]

required = [TARGET, GROUP, "area_type", "rrh_idx"] + SPATIAL
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError("Faltan columnas: " + ", ".join(missing))

df = df.dropna(subset=required).copy()

for col in ["area_type", "rrh_idx"]:
    df[col] = df[col].astype("category").cat.remove_unused_categories()

print(f"Filas válidas: {len(df):,}")
print(f"Escenarios: {df['area_type'].value_counts().to_dict()}")
print(f"Transmisiones: {df[GROUP].nunique():,}")
print(f"Posiciones únicas: {df[SPATIAL].drop_duplicates().shape[0]:,}")

print("\nPosiciones únicas por escenario:")
print(
    df.groupby("area_type", observed=True)[SPATIAL]
    .apply(lambda g: g.drop_duplicates().shape[0])
)


CATEGORICAL = ["area_type", "rrh_idx"]

X_raw = df[SPATIAL + CATEGORICAL].copy()
y = df[TARGET].to_numpy()
groups = df[GROUP].to_numpy()

print(f"\nX_raw: {X_raw.shape}, y: {y.shape}")


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), SPATIAL),
            ("cat", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
    )


kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))
)


def fit_gpr(
    X_train,
    y_train,
    random_state=42,
    max_train=MAX_TRAIN_GPR,
    n_restarts=0,
):
    if len(X_train) > max_train:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_train), size=max_train, replace=False)
        X_fit, y_fit = X_train[idx], y_train[idx]
    else:
        X_fit, y_fit = X_train, y_train

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=random_state,
    )
    model.fit(X_fit, y_fit)
    return model


def predict_in_batches(model, X, batch_size=5000, return_std=False):
    n = len(X)
    preds = []
    stds = [] if return_std else None

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        if return_std:
            p, s = model.predict(X[i:end], return_std=True)
            stds.append(s)
        else:
            p = model.predict(X[i:end])
        preds.append(p)

    if return_std:
        return np.concatenate(preds), np.concatenate(stds)
    return np.concatenate(preds)


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    if np.var(y_true) == 0:
        return np.nan
    return r2_score(y_true, y_pred)


print("\n" + "=" * 70)
print("AJUSTE GPR GLOBAL (informativo)")
print("=" * 70)

pre_g = build_preprocessor()
X_full = pre_g.fit_transform(X_raw)

t0 = time.time()
gpr = fit_gpr(
    X_full, y,
    random_state=RANDOM_STATE,
    n_restarts=N_RESTARTS_GLOBAL,
)
print(f"Entrenado en {time.time() - t0:.1f} s")

print("Kernel optimizado:")
print(gpr.kernel_)

n_train = min(len(X_full), MAX_TRAIN_GPR)
rng = np.random.default_rng(RANDOM_STATE)
idx_eval = rng.choice(len(X_full), size=n_train, replace=False)

y_pred_sub, y_std_sub = predict_in_batches(
    gpr, X_full[idx_eval], batch_size=2000, return_std=True
)
y_sub = y[idx_eval]

rmse_train = np.sqrt(mean_squared_error(y_sub, y_pred_sub))
mae_train = mean_absolute_error(y_sub, y_pred_sub)
r2_train = safe_r2(y_sub, y_pred_sub)

print(f"\nMétricas sobre sub-muestra de ajuste (n={n_train}):")
print(f"  RMSE: {rmse_train:.4f} dB")
print(f"  MAE:  {mae_train:.4f} dB")
print(f"  R²:   {r2_train:.4f}")


print("\n" + "=" * 70)
print(f"CV AGRUPADA POR TRANSMISIÓN ({N_SPLITS} folds)")
print("=" * 70)

gkf = GroupKFold(n_splits=N_SPLITS)
rmse_scores, mae_scores, r2_scores = [], [], []
fold_predictions = []
failed_folds = []

for fold, (tr, te) in enumerate(gkf.split(X_raw, y, groups=groups), 1):
    t0 = time.time()

    try:
        pre_fold = build_preprocessor()
        X_tr = pre_fold.fit_transform(X_raw.iloc[tr])
        X_te = pre_fold.transform(X_raw.iloc[te])
        y_tr, y_te = y[tr], y[te]

        model_fold = fit_gpr(
            X_tr, y_tr,
            random_state=RANDOM_STATE + fold,
            n_restarts=N_RESTARTS_CV,
        )
        pred, pred_std = predict_in_batches(
            model_fold, X_te, batch_size=5000, return_std=True
        )
    except Exception as e:
        print(f"Fold {fold}: ERROR → {e}")
        failed_folds.append(fold)
        continue

    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae = mean_absolute_error(y_te, pred)
    r2 = safe_r2(y_te, pred)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

    fold_predictions.append(pd.DataFrame({
        "original_index": df.index[te],
        "fold": fold,
        "area_type": df["area_type"].iloc[te].values,
        "snr_real": y_te,
        "snr_pred": pred,
        "prediction_std": pred_std,
    }))

    print(
        f"Fold {fold}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
        f"({time.time() - t0:.1f} s)"
    )

print("\n" + "=" * 70)
print("RESULTADOS CV")
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
        sgkf.split(X_raw, y, groups=spatial_groups), 1
    ):
        t0 = time.time()

        try:
            pre_fold = build_preprocessor()
            X_tr = pre_fold.fit_transform(X_raw.iloc[tr])
            X_te = pre_fold.transform(X_raw.iloc[te])
            y_tr, y_te = y[tr], y[te]

            model_sp = fit_gpr(
                X_tr, y_tr,
                random_state=200 + fold,
                n_restarts=N_RESTARTS_CV,
            )
            pred = predict_in_batches(model_sp, X_te, batch_size=5000)
        except Exception as e:
            print(f"Fold {fold}: ERROR → {e}")
            continue

        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)
        r2 = safe_r2(y_te, pred)

        spatial_rmse.append(rmse)
        spatial_mae.append(mae)
        spatial_r2.append(r2)

        print(
            f"Fold {fold}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
            f"({time.time() - t0:.1f} s)"
        )

    if spatial_rmse:
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

        print(f"\nModelo Mixto (referencia, métricas de ajuste):")
        print(f"  n = {len(mixed_df):,}")
        print(f"  RMSE = {mixed_rmse:.4f} dB")
        print(f"  MAE  = {mixed_mae:.4f} dB")
        print(f"  R²   = {mixed_r2:.4f}")

        print(f"\nModelo GPR (CV por transmisión):")
        print(f"  n = {len(all_pred):,}")
        print(f"  RMSE = {np.mean(rmse_scores):.4f} dB")
        print(f"  MAE  = {np.mean(mae_scores):.4f} dB")
        print(f"  R²   = {np.mean(r2_scores):.4f}")

        delta_rmse = np.mean(rmse_scores) - mixed_rmse
        delta_mae = np.mean(mae_scores) - mixed_mae
        delta_r2 = np.mean(r2_scores) - mixed_r2

        print(f"\nDiferencia (Δ = GPR − Mixto):")
        print(f"  ΔRMSE = {delta_rmse:+.4f} dB   (negativo = GPR mejor)")
        print(f"  ΔMAE  = {delta_mae:+.4f} dB   (negativo = GPR mejor)")
        print(f"  ΔR²   = {delta_r2:+.4f}      (positivo = GPR mejor)")

        print(
            "\n⚠ NOTA: comparación no estricta — Mixto reporta ajuste, "
            "GPR reporta CV. Para comparación justa, "
            "revisar modelo_mixto_principal_resumen.txt."
        )

        comp = pd.DataFrame([
            {"Modelo": "Mixto", "n": len(mixed_df),
             "RMSE": round(mixed_rmse, 4),
             "MAE": round(mixed_mae, 4),
             "R²": round(mixed_r2, 4),
             "Evaluación": "Ajuste (in-sample)"},
            {"Modelo": "GPR", "n": len(all_pred),
             "RMSE": round(np.mean(rmse_scores), 4),
             "MAE": round(np.mean(mae_scores), 4),
             "R²": round(np.mean(r2_scores), 4),
             "Evaluación": "CV por transmisión"},
        ])
        comp_path = RESULTS_DIR / "comparacion_mixto_vs_gpr.csv"
        comp.to_csv(comp_path, index=False)
        print(f"\n  ✔ {comp_path.name}")
    else:
        print("No se pudieron detectar columnas en el archivo del mixto.")
else:
    print(f"No existe: {MIXED_PREDICTIONS}")
    print("Ejecuta primero modelo_mixto.py si quieres la comparación.")


all_pred.to_parquet(
    RESULTS_DIR / "gpr_predicciones.parquet", index=False
)

pd.DataFrame({
    "fold": np.arange(1, len(rmse_scores) + 1),
    "rmse": rmse_scores,
    "mae": mae_scores,
    "r2": r2_scores,
}).to_csv(
    RESULTS_DIR / "gpr_validacion_transmission.csv", index=False
)

pd.DataFrame({
    "metric": ["RMSE", "MAE", "R2"],
    "value": [
        np.mean(spatial_rmse) if spatial_rmse else np.nan,
        np.mean(spatial_mae) if spatial_mae else np.nan,
        np.mean(spatial_r2) if spatial_r2 else np.nan,
    ],
}).to_csv(
    RESULTS_DIR / "gpr_validacion_espacial.csv", index=False
)

with open(RESULTS_DIR / "gpr_modelo.pkl", "wb") as fp:
    pickle.dump({
        "gpr": gpr,
        "preprocessor": pre_g,
        "features_used": SPATIAL + CATEGORICAL,
        "max_train": MAX_TRAIN_GPR,
        "n_restarts_global": N_RESTARTS_GLOBAL,
    }, fp)

with open(RESULTS_DIR / "gpr_resumen.txt", "w", encoding="utf-8") as fp:
    fp.write("GPR — SNR ESPACIAL (5 escenarios)\n" + "=" * 70 + "\n\n")
    fp.write(f"Input: {INPUT.name}\n")
    fp.write(f"Observaciones: {len(df):,}\n")
    fp.write(f"Transmisiones: {df[GROUP].nunique():,}\n")
    fp.write(f"Posiciones: {df[SPATIAL].drop_duplicates().shape[0]:,}\n\n")
    fp.write(f"MAX_TRAIN_GPR: {MAX_TRAIN_GPR}\n")
    fp.write(f"N_RESTARTS_GLOBAL: {N_RESTARTS_GLOBAL}\n")
    fp.write(f"N_RESTARTS_CV: {N_RESTARTS_CV}\n\n")
    fp.write(f"Features: {SPATIAL + CATEGORICAL}\n")
    fp.write("NOTA: indoor tiene 1 sola posición única.\n\n")
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
    fp.write(f"Kernel: {gpr.kernel_}\n")

print("\nArchivos guardados en:", RESULTS_DIR)
print("Listo.")