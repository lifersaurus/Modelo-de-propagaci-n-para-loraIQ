
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import patsy
from scipy import stats
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(r"C:\Users\Lenovo\OneDrive\Documentos\LORAIQ")
INPUT = BASE_DIR / "data" / "processed" / "dataset_mixto.parquet"
RESULTS_DIR = BASE_DIR / "results" / "model_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CV_SPLITS = 5

print("=" * 70)
print("MODELO MIXTO PRINCIPAL — SNR (5 escenarios)")
print("=" * 70)


if not INPUT.exists():
    raise FileNotFoundError(f"No existe:\n{INPUT}")

df = pd.read_parquet(INPUT)
print(f"\nArchivo: {INPUT}")
print(f"Filas: {len(df):,}")

required = ["snr", "area_type", "rrh_idx", "transmission_idx"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError("Faltan columnas: " + ", ".join(missing))

model_df = df[required].copy()

for col in ["area_type", "rrh_idx"]:
    model_df[col] = (
        model_df[col].astype("category").cat.remove_unused_categories()
    )

model_df["snr"] = pd.to_numeric(model_df["snr"], errors="coerce")
model_df = model_df.dropna(subset=required).copy()

print(f"Filas tras limpieza: {len(model_df):,}")
print(f"Escenarios: {model_df['area_type'].value_counts().to_dict()}")
print(f"RRH: {model_df['rrh_idx'].nunique()}")
print(f"Transmisiones: {model_df['transmission_idx'].nunique():,}")


formula = "snr ~ C(area_type)"
vc_formula = {"rrh": "0 + C(rrh_idx)"}

print(f"\nFórmula: {formula}")
print(f"RE: (1|transmission_idx) + (1|rrh_idx)")

y_pre, X_pre = patsy.dmatrices(formula, model_df, return_type="dataframe")
rank = np.linalg.matrix_rank(X_pre.values)
print(f"\nDiseño: {X_pre.shape[1]} columnas, rango {rank}")
if rank < X_pre.shape[1]:
    raise RuntimeError("Diseño singular.")


def fit_model(data, reml, maxiter=2000):
    model = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data["transmission_idx"],
        re_formula="1",
        vc_formula=vc_formula,
    )
    return model, model.fit(
        reml=reml, method="lbfgs", maxiter=maxiter, disp=False
    )


print("\n" + "=" * 70)
print("AJUSTE ML")
print("=" * 70)
model_ml, result_ml = fit_model(model_df, reml=False)
print("OK")

print("\n" + "=" * 70)
print("AJUSTE REML")
print("=" * 70)
model_reml, result_reml = fit_model(model_df, reml=True)
print("OK")

print("\n" + result_reml.summary().as_text())


print("\n" + "=" * 70)
print("COMPONENTES DE VARIANZA")
print("=" * 70)

cov_re = result_reml.cov_re
print("Covarianza de efectos aleatorios (transmission):")
print(cov_re)

var_tx = float(cov_re.iloc[0, 0])

vcomp = result_reml.vcomp
print(f"\nVariance components (vc_formula): {vcomp}")

if vcomp is not None and len(vcomp) > 0:
    var_rrh = float(np.asarray(vcomp).ravel()[0])
else:
    var_rrh = float("nan")

var_resid = float(result_reml.scale)
var_total = var_tx + var_rrh + var_resid

print(f"\nVarianza transmission: {var_tx:.4f}  ({var_tx/var_total*100:.1f}%)")
print(f"Varianza rrh:          {var_rrh:.4f}  ({var_rrh/var_total*100:.1f}%)")
print(f"Varianza residual:     {var_resid:.4f}  ({var_resid/var_total*100:.1f}%)")
print(f"Varianza total:        {var_total:.4f}")

icc = (
    (var_tx + var_rrh) / var_total
    if var_total > 0
    else np.nan
)

fe = result_reml.fe_params
X = result_reml.model.exog
var_fixed = float(np.var(X @ fe.values, ddof=1))
denom = var_fixed + var_tx + var_rrh + var_resid
r2_marg = var_fixed / denom if denom > 0 else np.nan
r2_cond = (var_fixed + var_tx + var_rrh) / denom if denom > 0 else np.nan

print(f"\nICC (transmission + rrh): {icc:.4f}")
print(f"R² marginal:              {r2_marg:.4f}")
print(f"R² condicional:           {r2_cond:.4f}")


print("\n" + "=" * 70)
print("EFECTOS FIJOS")
print("=" * 70)

fixed_effects = pd.DataFrame({
    "coef": result_reml.fe_params,
    "std_error": result_reml.bse_fe,
    "z": result_reml.tvalues,
    "p_value": result_reml.pvalues,
})

conf_int = result_reml.conf_int()
fixed_effects["ci_lower"] = conf_int.iloc[:, 0]
fixed_effects["ci_upper"] = conf_int.iloc[:, 1]

print(fixed_effects.to_string(float_format=lambda x: f"{x:.4f}"))


print("\n" + "=" * 70)
print("MÉTRICAS EN AJUSTE")
print("=" * 70)

model_df["snr_pred"] = result_reml.fittedvalues
model_df["residual"] = model_df["snr"] - model_df["snr_pred"]

rmse_train = np.sqrt((model_df["residual"] ** 2).mean())
mae_train = model_df["residual"].abs().mean()
ss_res = (model_df["residual"] ** 2).sum()
ss_tot = ((model_df["snr"] - model_df["snr"].mean()) ** 2).sum()
r2_train = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

print(f"RMSE: {rmse_train:.4f} dB")
print(f"MAE:  {mae_train:.4f} dB")
print(f"R²:   {r2_train:.4f}")


print("\n" + "=" * 70)
print(f"CV AGRUPADA POR TRANSMISIÓN ({N_CV_SPLITS} folds)")
print("=" * 70)

gkf = GroupKFold(n_splits=N_CV_SPLITS)
groups = model_df["transmission_idx"].values

rmse_folds, mae_folds, r2_folds = [], [], []

for fold, (tr, te) in enumerate(gkf.split(model_df, groups=groups), 1):
    train = model_df.iloc[tr].copy()
    test = model_df.iloc[te].copy()

    for col in ["area_type", "rrh_idx"]:
        train[col] = train[col].cat.remove_unused_categories()

    try:
        _, res_fold = fit_model(train, reml=True, maxiter=1000)
        preds = res_fold.predict(test)
        resid = test["snr"].values - preds.values

        rmse_folds.append(np.sqrt(np.mean(resid ** 2)))
        mae_folds.append(np.mean(np.abs(resid)))
        ss_res_f = np.sum(resid ** 2)
        ss_tot_f = np.sum((test["snr"].values - test["snr"].mean()) ** 2)
        r2_folds.append(1 - ss_res_f / ss_tot_f)

        print(
            f"Fold {fold}: RMSE={rmse_folds[-1]:.4f}  "
            f"MAE={mae_folds[-1]:.4f}  R²={r2_folds[-1]:.4f}"
        )
    except Exception as e:
        print(f"Fold {fold}: error -> {e}")

print("\n" + "=" * 70)
print("RESULTADOS CV")
print("=" * 70)
if rmse_folds:
    print(f"RMSE: {np.mean(rmse_folds):.4f} ± {np.std(rmse_folds):.4f} dB")
    print(f"MAE:  {np.mean(mae_folds):.4f} ± {np.std(mae_folds):.4f} dB")
    print(f"R²:   {np.mean(r2_folds):.4f} ± {np.std(r2_folds):.4f}")
else:
    print("Ningún fold se completó.")


print("\n" + "=" * 70)
print("DIAGNÓSTICO DE RESIDUOS")
print("=" * 70)

residuals = model_df["residual"].values
sw = stats.shapiro(residuals[:5000])
print(f"Shapiro-Wilk: W={sw.statistic:.4f}, p={sw.pvalue:.4e}")
print(f"Asimetría: {stats.skew(residuals):.4f}")
print(f"Curtosis:  {stats.kurtosis(residuals):.4f}")
corr_abs = np.corrcoef(
    np.abs(residuals), model_df["snr_pred"].values
)[0, 1]
print(f"Correlación |residual| vs predicho: {corr_abs:.4f}")


print("\n" + "=" * 70)
print("COMPARACIÓN CON MODELO NULO")
print("=" * 70)

null_formula = "snr ~ 1"
model_null = smf.mixedlm(
    formula=null_formula,
    data=model_df,
    groups=model_df["transmission_idx"],
    re_formula="1",
    vc_formula=vc_formula,
)
result_null = model_null.fit(
    reml=False, method="lbfgs", maxiter=1000, disp=False
)

lr_stat = 2 * (result_ml.llf - result_null.llf)
df_diff = len(result_ml.fe_params) - len(result_null.fe_params)
lr_p = stats.chi2.sf(lr_stat, max(df_diff, 1))

print(f"LogLik completo (ML): {result_ml.llf:.4f}")
print(f"LogLik nulo (ML):     {result_null.llf:.4f}")
print(f"LR stat: {lr_stat:.4f}, df={df_diff}, p={lr_p:.4e}")


fixed_effects.to_csv(
    RESULTS_DIR / "modelo_mixto_principal_efectos_fijos.csv", index=True
)

random_effects_rows = []
for gname, gseries in result_reml.random_effects.items():
    random_effects_rows.append({
        "group": str(gname),
        "random_intercept": float(np.asarray(gseries).ravel()[0]),
    })
pd.DataFrame(random_effects_rows).to_csv(
    RESULTS_DIR / "modelo_mixto_principal_efectos_aleatorios.csv",
    index=False,
)

model_df.to_parquet(
    RESULTS_DIR / "modelo_mixto_principal_residuos.parquet", index=False
)

with open(RESULTS_DIR / "modelo_mixto_principal.pkl", "wb") as fp:
    pickle.dump({"model": model_reml, "result": result_reml}, fp)

with open(
    RESULTS_DIR / "modelo_mixto_principal_resumen.txt",
    "w", encoding="utf-8",
) as fp:
    fp.write("MODELO MIXTO PRINCIPAL — SNR\n")
    fp.write("=" * 70 + "\n\n")
    fp.write(f"Observaciones: {len(model_df):,}\n")
    fp.write(f"Escenarios: {model_df['area_type'].nunique()}\n")
    fp.write(f"Transmisiones: {model_df['transmission_idx'].nunique():,}\n")
    fp.write(f"RRH: {model_df['rrh_idx'].nunique()}\n\n")
    fp.write(f"Fórmula: {formula}\n")
    fp.write(f"RE: (1|transmission_idx) + (1|rrh_idx)\n\n")
    fp.write(f"AIC (ML): {result_ml.aic:.4f}\n")
    fp.write(f"BIC (ML): {result_ml.bic:.4f}\n")
    fp.write(f"LogLik (ML): {result_ml.llf:.4f}\n\n")
    fp.write(f"ICC: {icc:.4f}\n")
    fp.write(f"R² marginal:    {r2_marg:.4f}\n")
    fp.write(f"R² condicional: {r2_cond:.4f}\n\n")
    fp.write(f"Varianza transmission: {var_tx:.4f}\n")
    fp.write(f"Varianza rrh:          {var_rrh:.4f}\n")
    fp.write(f"Varianza residual:     {var_resid:.4f}\n\n")
    fp.write(f"RMSE ajuste: {rmse_train:.4f} dB\n")
    fp.write(f"MAE  ajuste: {mae_train:.4f} dB\n")
    fp.write(f"R²   ajuste: {r2_train:.4f}\n\n")
    if rmse_folds:
        fp.write(f"RMSE CV: {np.mean(rmse_folds):.4f} ± {np.std(rmse_folds):.4f} dB\n")
        fp.write(f"MAE  CV: {np.mean(mae_folds):.4f} ± {np.std(mae_folds):.4f} dB\n")
        fp.write(f"R²   CV: {np.mean(r2_folds):.4f} ± {np.std(r2_folds):.4f}\n\n")
    fp.write("EFECTOS FIJOS\n" + "-" * 70 + "\n")
    fp.write(fixed_effects.to_string(float_format=lambda x: f"{x:.4f}"))

print("\nArchivos guardados en:", RESULTS_DIR)
print("Listo.")