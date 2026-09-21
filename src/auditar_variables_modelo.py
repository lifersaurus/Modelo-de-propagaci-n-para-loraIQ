import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "data" / "processed" / "lora_features.parquet"
OUTPUT_DIR = BASE_DIR / "results" / "model_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("AUDITORÍA DE VARIABLES PARA DATASET MODEL-READY")
print("=" * 70)

df = pd.read_parquet(INPUT)

print(f"\nRegistros: {len(df):,}")
print(f"Columnas:  {len(df.columns)}")

# ============================================================
# 1. INVESTIGACIÓN DE CFO EXTREMOS
# ============================================================

print("\n" + "=" * 70)
print("1. CFO EXTREMOS")
print("=" * 70)

cfo_extreme = df[df["cfo"].abs() > 50].copy()

print(f"\nRegistros con |CFO| > 50: {len(cfo_extreme)}")

if len(cfo_extreme) > 0:

    cols = [
        "file",
        "datetime",
        "transmission_idx",
        "rrh_idx",
        "area_type",
        "name",
        "sample_rate",
        "bandwidth",
        "sf",
        "cfo",
        "snr",
        "iq_power_db",
        "latitude",
        "longitude",
        "velocity",
        "target_velocity"
    ]

    print("\nRegistros extremos:")
    print(
        cfo_extreme[cols]
        .sort_values("cfo")
        .to_string(index=False)
    )

    cfo_extreme[cols].to_csv(
        OUTPUT_DIR / "cfo_extremos.csv",
        index=False
    )

# ============================================================
# 2. CFO POR RRH
# ============================================================

print("\n" + "=" * 70)
print("2. CFO POR RRH")
print("=" * 70)

cfo_rrh = (
    df.groupby("rrh_idx")["cfo"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max"
    )
)

print(cfo_rrh)

cfo_rrh.to_csv(
    OUTPUT_DIR / "cfo_por_rrh.csv"
)

# ============================================================
# 3. VARIABLES NUMÉRICAS
# ============================================================

print("\n" + "=" * 70)
print("3. VARIABLES NUMÉRICAS")
print("=" * 70)

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

print(f"\nVariables numéricas: {len(numeric_cols)}")

# ============================================================
# 4. CORRELACIÓN
# ============================================================

print("\n" + "=" * 70)
print("4. CORRELACIONES ALTAS")
print("=" * 70)

corr = df[numeric_cols].corr()

pairs = []

for i in range(len(corr.columns)):

    for j in range(i + 1, len(corr.columns)):

        a = corr.columns[i]
        b = corr.columns[j]

        value = corr.iloc[i, j]

        if pd.notna(value):

            pairs.append({
                "variable_1": a,
                "variable_2": b,
                "correlation": value,
                "abs_correlation": abs(value)
            })

corr_pairs = pd.DataFrame(pairs)

corr_pairs = corr_pairs.sort_values(
    "abs_correlation",
    ascending=False
)

print("\nTop 30 correlaciones:")

print(
    corr_pairs.head(30).to_string(index=False)
)

corr_pairs.to_csv(
    OUTPUT_DIR / "correlaciones_variables.csv",
    index=False
)

# ============================================================
# 5. REDUNDANCIA MUY ALTA
# ============================================================

print("\n" + "=" * 70)
print("5. REDUNDANCIA |r| >= 0.90")
print("=" * 70)

high_corr = corr_pairs[
    corr_pairs["abs_correlation"] >= 0.90
]

print(
    high_corr.to_string(index=False)
)

high_corr.to_csv(
    OUTPUT_DIR / "redundancia_alta.csv",
    index=False
)

# ============================================================
# 6. INFORMACIÓN DE CADA VARIABLE
# ============================================================

print("\n" + "=" * 70)
print("6. AUDITORÍA DE VARIABLES")
print("=" * 70)

audit = []

for column in df.columns:

    dtype = str(df[column].dtype)

    missing = int(df[column].isna().sum())

    missing_pct = (
        missing / len(df) * 100
    )

    unique = int(
        df[column].nunique(dropna=True)
    )

    row = {
        "variable": column,
        "dtype": dtype,
        "missing": missing,
        "missing_pct": missing_pct,
        "unique": unique,
        "decision": "",
        "reason": ""
    }

    # --------------------------------------------------------
    # Identificadores
    # --------------------------------------------------------

    if column in [
        "file",
        "datetime"
    ]:

        row["decision"] = "AUXILIAR"
        row["reason"] = (
            "Identificación/ordenamiento; "
            "no usar directamente como feature."
        )

    elif column in [
        "transmission_idx",
        "rrh_idx"
    ]:

        row["decision"] = "CONSERVAR"
        row["reason"] = (
            "Necesario para conservar la estructura "
            "transmisión-RRH y realizar agrupamientos."
        )

    # --------------------------------------------------------
    # Configuración
    # --------------------------------------------------------

    elif column in [
        "datatype",
        "frequency_MHz",
        "fc",
        "cr"
    ]:

        row["decision"] = "AUXILIAR"
        row["reason"] = (
            "Configuración prácticamente constante "
            "en el dataset."
        )

    elif column in [
        "sample_rate",
        "bandwidth",
        "sf"
    ]:

        row["decision"] = "CONSERVAR"
        row["reason"] = (
            "Parámetros de configuración que cambian "
            "entre registros."
        )

    # --------------------------------------------------------
    # Escenario
    # --------------------------------------------------------

    elif column in [
        "area",
        "area_type",
        "name"
    ]:

        row["decision"] = "CONSERVAR"
        row["reason"] = (
            "Información contextual del escenario "
            "de propagación."
        )

    # --------------------------------------------------------
    # Posición
    # --------------------------------------------------------

    elif column in [
        "latitude",
        "longitude"
    ]:

        row["decision"] = "CONSERVAR"
        row["reason"] = (
            "Posición asociada al transmisor; "
            "útil para análisis espacial."
        )

    # --------------------------------------------------------
    # Movilidad
    # --------------------------------------------------------

    elif column in [
        "velocity",
        "target_velocity"
    ]:

        row["decision"] = "CONSERVAR"
        row["reason"] = (
            "Variables relacionadas con movilidad."
        )

    # --------------------------------------------------------
    # CFO
    # --------------------------------------------------------

    elif column == "cfo":

        row["decision"] = "INVESTIGAR"
        row["reason"] = (
            "Presenta outliers extremos; "
            "requiere revisión antes de utilizarse."
        )

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    elif column == "snr":

        row["decision"] = "DEPENDE_DEL_MODELO"
        row["reason"] = (
            "Puede ser variable objetivo o feature; "
            "no decidir hasta definir el objetivo."
        )

    # --------------------------------------------------------
    # Features IQ
    # --------------------------------------------------------

    elif column in [
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
        "amplitude_p90"
    ]:

        row["decision"] = "CANDIDATA"
        row["reason"] = (
            "Feature derivada directamente de la "
            "señal IQ."
        )

    # --------------------------------------------------------
    # Features temporales
    # --------------------------------------------------------

    elif column in [
        "power_std",
        "power_cv",
        "power_min_db",
        "power_max_db",
        "power_range_db"
    ]:

        row["decision"] = "CANDIDATA"
        row["reason"] = (
            "Describe variación temporal de la "
            "potencia dentro del segmento IQ."
        )

    # --------------------------------------------------------
    # Features espectrales
    # --------------------------------------------------------

    elif column in [
        "spectral_peak_frequency",
        "spectral_peak_power_db",
        "spectral_total_power_db"
    ]:

        row["decision"] = "CANDIDATA"
        row["reason"] = (
            "Feature derivada del contenido espectral."
        )

    # --------------------------------------------------------
    # Bloques
    # --------------------------------------------------------

    elif column.startswith(
        "block_power_"
    ):

        row["decision"] = "CANDIDATA"
        row["reason"] = (
            "Potencia relativa por bloque temporal."
        )

    # --------------------------------------------------------
    # Segmentación
    # --------------------------------------------------------

    elif column in [
        "sample_start",
        "sample_count",
        "n_samples",
        "freq_lower_edge",
        "freq_upper_edge"
    ]:

        row["decision"] = "AUXILIAR"
        row["reason"] = (
            "Información de segmentación/metadatos "
            "del registro IQ."
        )

    else:

        row["decision"] = "REVISAR"
        row["reason"] = (
            "Variable no clasificada automáticamente."
        )

    audit.append(row)

audit_df = pd.DataFrame(audit)

print(
    audit_df[
        [
            "variable",
            "missing_pct",
            "unique",
            "decision",
            "reason"
        ]
    ].to_string(index=False)
)

audit_df.to_csv(
    OUTPUT_DIR / "auditoria_variables.csv",
    index=False
)

# ============================================================
# 7. RESUMEN DE DECISIONES
# ============================================================

print("\n" + "=" * 70)
print("7. RESUMEN DE DECISIONES")
print("=" * 70)

decision_counts = (
    audit_df["decision"]
    .value_counts()
)

print(decision_counts)

decision_counts.to_csv(
    OUTPUT_DIR / "resumen_decisiones.csv"
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("AUDITORÍA FINALIZADA")
print("=" * 70)

print("\nArchivos generados:")

for file in sorted(
    OUTPUT_DIR.glob("*.csv")
):

    print(f"  {file.name}")