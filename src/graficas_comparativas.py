from pathlib import Path
import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE_DIR = Path(r"C:\Users\Lenovo\OneDrive\Documentos\LORAIQ")
RESULTS_DIR = BASE_DIR / "results" / "model_analysis"
FIGURES_DIR = BASE_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

sns.set_palette("deep")

COLORS = {
    "Mixto":   "#1f77b4",
    "GPR":     "#2ca02c",
    "XGBoost": "#d62728",
    "ITU-P833": "#ff7f0e",
}

SCENARIO_ORDER = [
    "drone_los", "drone_nlos", "indoor",
    "pedestrian_nlos", "pedestrian_partial_los"
]


def safe_savefig(fig, path, max_retries=5, delay=1.0, dpi=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()

    last_err = None
    for attempt in range(max_retries):
        try:
            with open(path, "wb") as f:
                f.write(data)
            return True
        except OSError as e:
            last_err = e
            time.sleep(delay * (attempt + 1))

    raise OSError(
        f"No se pudo guardar {path} tras {max_retries} intentos: {last_err}"
    )


def load_predictions(model_name, path):
    if not path.exists():
        print(f"  ✗ {model_name}: no existe {path.name}")
        return None

    df = pd.read_parquet(path)

    target_col = None
    for c in ["snr", "snr_real", "y_true"]:
        if c in df.columns:
            target_col = c
            break

    pred_col = None
    for c in ["snr_pred", "pred", "y_pred", "prediction"]:
        if c in df.columns:
            pred_col = c
            break

    if target_col is None or pred_col is None:
        print(f"  ✗ {model_name}: faltan columnas")
        return None

    out = pd.DataFrame({
        "model": model_name,
        "snr_real": df[target_col],
        "snr_pred": df[pred_col],
    })

    for c in ["area_type", "transmission_idx",
              "latitude", "longitude", "original_index"]:
        if c in df.columns:
            out[c] = df[c].values

    out["residual"] = out["snr_real"] - out["snr_pred"]
    return out


print("=" * 70)
print("PANELES DIAGNÓSTICOS POR MODELO")
print("=" * 70)

predictions = {}
for name, filename in [
    ("Mixto",   "modelo_mixto_principal_residuos.parquet"),
    ("GPR",     "gpr_predicciones.parquet"),
    ("XGBoost", "xgboost_predicciones.parquet"),
    ("ITU-P833", "itu_predicciones.parquet"),
]:
    p = load_predictions(name, RESULTS_DIR / filename)
    if p is not None:
        predictions[name] = p
        print(f"  ✔ {name}: {len(p):,} predicciones")

if not predictions:
    raise RuntimeError("No hay predicciones disponibles para graficar.")


def make_model_panel(model_name, df, color):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    r2 = r2_score(df["snr_real"], df["snr_pred"])
    rmse = np.sqrt(mean_squared_error(df["snr_real"], df["snr_pred"]))
    mae = mean_absolute_error(df["snr_real"], df["snr_pred"])

    ax = axes[0]
    lims = [-25, 60]
    hb = ax.hexbin(
        df["snr_real"], df["snr_pred"],
        gridsize=50,
        cmap="Blues" if model_name != "XGBoost" else "Reds",
        mincnt=1,
        extent=[lims[0], lims[1], lims[0], lims[1]],
    )
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Nº de observaciones")

    ax.plot(
        lims, lims,
        color="black", linestyle="--", linewidth=1.5,
        label="y = x  (predicción perfecta)",
        zorder=10,
    )
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("SNR real (dB)")
    ax.set_ylabel("SNR predicho (dB)")
    ax.set_title("(A) Predicción vs realidad")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax.text(
        0.97, 0.03,
        f"RMSE = {rmse:.2f} dB\nMAE  = {mae:.2f} dB\nR²   = {r2:.3f}",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="gray"),
    )

    ax = axes[1]
    hb = ax.hexbin(
        df["snr_pred"], df["residual"],
        gridsize=50,
        cmap="Blues" if model_name != "XGBoost" else "Reds",
        mincnt=1,
    )
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Nº de observaciones")

    ax.axhline(
        0, color="black", linewidth=1.2, linestyle="--",
        label="Residuo = 0  (predicción perfecta)",
        zorder=10,
    )
    ax.set_xlabel("SNR predicho (dB)")
    ax.set_ylabel("Residuo (real − predicho) [dB]")
    ax.set_title("(B) Residuos vs predicho")
    ax.set_ylim(-40, 40)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    bias = df["residual"].mean()
    sigma = df["residual"].std()
    ax.text(
        0.03, 0.03,
        f"Bias = {bias:+.2f} dB\nσ    = {sigma:.2f} dB",
        transform=ax.transAxes, va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="gray"),
    )

    ax = axes[2]
    if "area_type" in df.columns:
        df_plot = df.copy()
        df_plot["area_type"] = pd.Categorical(
            df_plot["area_type"], categories=SCENARIO_ORDER, ordered=True
        )
        df_plot = df_plot.dropna(subset=["area_type"])

        sns.boxplot(
            data=df_plot, x="area_type", y="residual",
            ax=ax, color=color, showfliers=False, width=0.6,
        )
        ax.axhline(
            0, color="black", linewidth=1.2, linestyle="--",
            label="Residuo = 0  (sin error)",
            zorder=10,
        )
        ax.set_ylim(-30, 30)
        ax.set_xlabel("")
        ax.set_ylabel("Residuo (dB)")
        ax.set_title("(C) Residuos por escenario")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "sin area_type", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle(f"Diagnóstico — {model_name}", fontsize=13, y=1.02)
    fig.tight_layout()

    out = FIGURES_DIR / f"panel_{model_name.lower().replace('-', '_')}.png"
    safe_savefig(fig, out)
    plt.close(fig)
    print(f"  ✔ {out.name}")


print()
for name, df in predictions.items():
    make_model_panel(name, df, COLORS.get(name, "#333333"))


def fig_final_why():
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.25)

    feature_map = {
        "Mixto":    "area_type + RE",
        "GPR":      "lat/lon + area + rrh",
        "XGBoost":  "lat/lon + area + rrh",
        "ITU-P833": "dist + vegetación + área",
    }

    metrics = []
    for name, df in predictions.items():
        metrics.append({
            "model": name,
            "rmse": np.sqrt(mean_squared_error(df["snr_real"], df["snr_pred"])),
            "mae": mean_absolute_error(df["snr_real"], df["snr_pred"]),
            "r2_global": r2_score(df["snr_real"], df["snr_pred"]),
            "n_features": feature_map.get(name, "-"),
        })
    mdf = pd.DataFrame(metrics).sort_values("rmse")

    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(
        mdf["model"], mdf["rmse"],
        color=[COLORS.get(m, "#333333") for m in mdf["model"]],
        edgecolor="black", linewidth=0.5,
    )
    ax.set_ylabel("RMSE CV (dB)")
    ax.set_title("(A) Ranking de modelos — menor es mejor", fontsize=11)
    ax.set_ylim(0, max(mdf["rmse"]) * 1.3)

    for bar, val in zip(bars, mdf["rmse"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{val:.2f}",
            ha="center", va="bottom", fontweight="bold",
        )

    ax.text(
        0.5, -0.30,
        "Mixto solo ve etiquetas categóricas.\n"
        "ITU/GPR/XGBoost añaden información geométrica.",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, style="italic", color="#444",
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#e6c200"),
    )

    ax = fig.add_subplot(gs[0, 1])

    data_r2 = []
    for name, df in predictions.items():
        data_r2.append((name, "Global", r2_score(df["snr_real"], df["snr_pred"])))
        if "area_type" in df.columns:
            for sc in SCENARIO_ORDER:
                sub = df[df["area_type"] == sc]
                if len(sub) > 1:
                    data_r2.append(
                        (name, sc, r2_score(sub["snr_real"], sub["snr_pred"]))
                    )

    r2df = pd.DataFrame(data_r2, columns=["model", "scope", "r2"])
    intra = (
        r2df[r2df["scope"] != "Global"]
        .groupby("model")["r2"].mean()
        .reset_index()
    )
    intra = intra.merge(
        r2df[r2df["scope"] == "Global"][["model", "r2"]].rename(
            columns={"r2": "r2_global"}
        ),
        on="model",
    )
    intra = intra.set_index("model").reindex(mdf["model"]).reset_index()

    x = np.arange(len(intra))
    w = 0.35

    ax.bar(
        x - w/2, intra["r2_global"], w,
        label="R² global",
        color=[COLORS.get(m, "#333333") for m in intra["model"]],
        edgecolor="black", linewidth=0.5,
    )
    ax.bar(
        x + w/2, intra["r2"], w,
        label="R² medio intra-escenario",
        color=[COLORS.get(m, "#333333") for m in intra["model"]],
        alpha=0.5, edgecolor="black", linewidth=0.5, hatch="//",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(intra["model"])
    ax.set_ylabel("R²")
    ax.axhline(0, color="gray", linewidth=0.8,
               label="R² = 0 (equivalente a la media)")
    ax.set_title("(B) El R² global es engañoso", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.1, 1.05)
    ax.text(
        0.5, -0.30,
        "La mayor parte del R² viene de acertar el escenario.\n"
        "Dentro de un mismo escenario, el modelo explica menos.",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, style="italic", color="#444",
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#e6c200"),
    )

    ax = fig.add_subplot(gs[1, 0])

    times = {"Mixto": 60, "GPR": 195, "XGBoost": 4, "ITU-P833": 2}

    for name in mdf["model"]:
        rmse = mdf[mdf["model"] == name]["rmse"].values[0]
        t = times.get(name, 10)
        ax.scatter(
            t, rmse,
            s=250, color=COLORS.get(name, "#333333"),
            edgecolor="black", linewidth=1,
            zorder=3,
        )
        ax.annotate(
            name, (t, rmse),
            xytext=(10, 5), textcoords="offset points",
            fontsize=10, fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Tiempo por fold (s, escala log)")
    ax.set_ylabel("RMSE CV (dB)")
    ax.set_title("(C) Coste computacional vs precisión", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    ax.text(
        0.5, -0.30,
        "XGBoost e ITU: mejor compromiso rapidez/precisión.\n"
        "GPR solo se justifica si el objetivo es extrapolación espacial.",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, style="italic", color="#444",
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#e6c200"),
    )

    ax = fig.add_subplot(gs[1, 1])

    all_res = []
    for name, df in predictions.items():
        sub = df[["residual"]].copy()
        sub["model"] = name
        all_res.append(sub)
    res_df = pd.concat(all_res)

    order = [m for m in ["Mixto", "GPR", "XGBoost", "ITU-P833"] if m in predictions]

    sns.violinplot(
        data=res_df, x="model", y="residual",
        order=order,
        hue="model",
        palette=COLORS,
        legend=False,
        ax=ax, inner="quartile", cut=0,
    )
    ax.axhline(
        0, color="black", linewidth=1.2, linestyle="--",
        label="Residuo = 0  (sin error)",
        zorder=10,
    )
    ax.set_ylim(-30, 30)
    ax.set_ylabel("Residuo (dB)")
    ax.set_xlabel("")
    ax.set_title("(D) Distribución de residuos por modelo", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    ax.text(
        0.5, -0.30,
        "Mixto: residuos anchos y asimétricos.\n"
        "ITU/GPR/XGBoost: concentrados alrededor de 0.",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, style="italic", color="#444",
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#e6c200"),
    )

    fig.suptitle(
        "Por qué unos modelos superan a otros",
        fontsize=14, y=0.99, fontweight="bold",
    )

    out = FIGURES_DIR / "fig_final_porque.png"
    safe_savefig(fig, out)
    plt.close(fig)
    print(f"\n  ✔ {out.name}")


if len(predictions) >= 2:
    fig_final_why()
else:
    print("\n  (Se necesitan al menos 2 modelos para la figura final)")


def plot_residual_map(df, model_name):
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print(f"  ✗ {model_name}: sin lat/lon para mapa de residuos")
        return

    df_plot = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(df_plot) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.abs(df_plot["residual"]).quantile(0.99)

    sc = ax.scatter(
        df_plot["longitude"], df_plot["latitude"],
        c=df_plot["residual"],
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        s=6, alpha=0.6,
    )
    plt.colorbar(sc, ax=ax, label="Residuo (real − predicho) [dB]")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(f"Mapa de residuos — {model_name}")
    ax.set_aspect("equal")

    out = FIGURES_DIR / f"mapa_residuos_{model_name.lower().replace('-', '_')}.png"
    safe_savefig(fig, out)
    plt.close(fig)
    print(f"  ✔ {out.name}")


def plot_cdf_error(predictions_dict):
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in predictions_dict.items():
        err = np.sort(np.abs(df["residual"].values))
        cdf = np.arange(1, len(err) + 1) / len(err)
        ax.plot(
            err, cdf, label=name,
            color=COLORS.get(name, "#333333"), linewidth=2,
        )

    ax.axhline(0.95, color="gray", linestyle=":",
               label="95% de predicciones")
    ax.axvline(5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Error absoluto (dB)")
    ax.set_ylabel("Fracción acumulada")
    ax.set_title("CDF del error absoluto por modelo")
    ax.set_xlim(0, 30)
    ax.legend()
    ax.grid(alpha=0.3)

    out = FIGURES_DIR / "cdf_error.png"
    safe_savefig(fig, out)
    plt.close(fig)
    print(f"  ✔ {out.name}")


def plot_bootstrap_ci(predictions_dict, n_boot=1000):
    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(42)

    names = []
    for i, (name, df) in enumerate(predictions_dict.items()):
        residuals = df["residual"].values
        n = len(residuals)

        boot_rmse = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            boot_rmse[b] = np.sqrt((residuals[idx] ** 2).mean())

        mean = boot_rmse.mean()
        ci_low, ci_high = np.percentile(boot_rmse, [2.5, 97.5])

        ax.errorbar(
            i, mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="o", markersize=12, capsize=8, capthick=2,
            color=COLORS.get(name, "#333333"), linewidth=2,
        )
        ax.text(
            i, ci_high + 0.15,
            f"{mean:.2f}\n[{ci_low:.2f}, {ci_high:.2f}]",
            ha="center", va="bottom", fontsize=9,
        )
        names.append(name)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("RMSE con IC 95% (bootstrap, n=1000)")
    ax.grid(alpha=0.3, axis="y")

    out = FIGURES_DIR / "bootstrap_rmse.png"
    safe_savefig(fig, out)
    plt.close(fig)
    print(f"  ✔ {out.name}")


def plot_error_correlation(predictions_dict):
    residuals = {}
    for name, df in predictions_dict.items():
        if "original_index" in df.columns:
            residuals[name] = df.set_index("original_index")["residual"]

    if len(residuals) < 2:
        print("  ✗ correlación de residuos: faltan índices")
        return

    rdf = pd.DataFrame(residuals).dropna()
    corr = rdf.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="coolwarm",
        vmin=-1, vmax=1, ax=ax, square=True,
        cbar_kws={"label": "Correlación"},
    )
    ax.set_title("Correlación de residuos entre modelos")

    out = FIGURES_DIR / "correlacion_residuos.png"
    safe_savefig(fig, out)
    plt.close(fig)
    print(f"  ✔ {out.name}")


print("\n" + "=" * 70)
print("GRÁFICAS ADICIONALES")
print("=" * 70)

for name, df in predictions.items():
    plot_residual_map(df, name)

plot_cdf_error(predictions)
plot_bootstrap_ci(predictions)
plot_error_correlation(predictions)


print("\n" + "=" * 70)
print("RESUMEN DE MÉTRICAS")
print("=" * 70)

rows = []
for name, df in predictions.items():
    rows.append({
        "Modelo": name,
        "n":      len(df),
        "RMSE":   round(np.sqrt(mean_squared_error(df["snr_real"], df["snr_pred"])), 3),
        "MAE":    round(mean_absolute_error(df["snr_real"], df["snr_pred"]), 3),
        "R²":     round(r2_score(df["snr_real"], df["snr_pred"]), 3),
        "Bias":   round(df["residual"].mean(), 3),
        "σ_res":  round(df["residual"].std(), 3),
    })

summary = pd.DataFrame(rows).sort_values("RMSE")
print(summary.to_string(index=False))

summary_path = RESULTS_DIR / "resumen_metricas_modelos.csv"
try:
    summary.to_csv(summary_path, index=False)
    print(f"\n  ✔ {summary_path.name}")
except OSError:
    for attempt in range(3):
        try:
            time.sleep(1.5 * (attempt + 1))
            summary.to_csv(summary_path, index=False)
            break
        except OSError:
            continue

print("\n" + "=" * 70)
print(f"Figuras guardadas en: {FIGURES_DIR}")
print("=" * 70)