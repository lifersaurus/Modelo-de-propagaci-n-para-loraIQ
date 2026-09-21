from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "lorawan_metadata.parquet"
)


# ============================================================
# COLUMNAS ESPERADAS
# ============================================================

EXPECTED_COLUMNS = [
    "file",
    "datetime",
    "datatype",
    "frequency_MHz",
    "sample_rate",
    "transmission_idx",
    "rrh_idx",
    "bandwidth",
    "sf",
    "cr",
    "fc",
    "cfo",
    "snr",
    "latitude",
    "longitude",
    "velocity",
    "target_velocity",
    "area",
    "area_type",
    "name",
    "sample_start",
    "sample_count",
    "freq_lower_edge",
    "freq_upper_edge",
]


# ============================================================
# FUNCIÓN AUXILIAR
# ============================================================

def imprimir_seccion(titulo):

    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


# ============================================================
# VALIDACIÓN
# ============================================================

def main():

    print("=" * 70)
    print("VALIDACIÓN DEL DATASET LoRaWAN")
    print("=" * 70)

    print(f"\nArchivo:")
    print(METADATA_PATH)

    # --------------------------------------------------------
    # EXISTENCIA
    # --------------------------------------------------------

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"No existe el archivo:\n{METADATA_PATH}"
        )

    # --------------------------------------------------------
    # CARGAR DATASET
    # --------------------------------------------------------

    df = pd.read_parquet(
        METADATA_PATH
    )

    print(
        f"\nFilas: {len(df):,}"
    )

    print(
        f"Columnas: {len(df.columns)}"
    )

    # ========================================================
    # COLUMNAS
    # ========================================================

    imprimir_seccion(
        "1. COLUMNAS"
    )

    print(
        "Columnas encontradas:"
    )

    for column in df.columns:

        print(
            f"  ✓ {column}"
        )

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print(
            "\n⚠ Columnas faltantes:"
        )

        for column in missing_columns:

            print(
                f"  ✗ {column}"
            )

    else:

        print(
            "\n✓ Todas las columnas esperadas están presentes."
        )

    # ========================================================
    # TIPOS DE DATOS
    # ========================================================

    imprimir_seccion(
        "2. TIPOS DE DATOS"
    )

    print(
        df.dtypes.to_string()
    )

    # ========================================================
    # TRANSMISIONES
    # ========================================================

    imprimir_seccion(
        "3. TRANSMISIONES"
    )

    if "transmission_idx" in df.columns:

        total_tx = (
            df["transmission_idx"]
            .nunique()
        )

        print(
            f"Transmisiones únicas: {total_tx:,}"
        )

        rrh_per_tx = (
            df.groupby(
                "transmission_idx"
            )["rrh_idx"]
            .nunique()
        )

        distribution = (
            rrh_per_tx
            .value_counts()
            .sort_index()
        )

        print(
            "\nNúmero de RRH por transmisión:"
        )

        for n_rrh, count in distribution.items():

            print(
                f"  {n_rrh} RRH: "
                f"{count:,} transmisiones"
            )

        multi_rrh = (
            rrh_per_tx > 1
        ).sum()

        print(
            f"\nTransmisiones recibidas por "
            f"más de un RRH: {multi_rrh:,}"
        )

        print(
            f"Porcentaje multi-RRH: "
            f"{multi_rrh / total_tx * 100:.2f}%"
        )

    # ========================================================
    # RRH
    # ========================================================

    imprimir_seccion(
        "4. RRH"
    )

    if "rrh_idx" in df.columns:

        rrh_counts = (
            df["rrh_idx"]
            .value_counts()
            .sort_index()
        )

        print(
            rrh_counts.to_string()
        )

    # ========================================================
    # ESCENARIOS
    # ========================================================

    imprimir_seccion(
        "5. ESCENARIOS"
    )

    if "area_type" in df.columns:

        scenario_counts = (
            df["area_type"]
            .value_counts()
        )

        print(
            scenario_counts.to_string()
        )

    # ========================================================
    # PARÁMETROS LoRa
    # ========================================================

    imprimir_seccion(
        "6. PARÁMETROS LoRa"
    )

    for column in [
        "bandwidth",
        "sf",
        "cr",
        "frequency_MHz",
        "sample_rate",
    ]:

        if column not in df.columns:
            continue

        print(
            f"\n{column}:"
        )

        print(
            df[column]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    # ========================================================
    # SNR
    # ========================================================

    imprimir_seccion(
        "7. SNR"
    )

    if "snr" in df.columns:

        print(
            df["snr"]
            .describe()
            .to_string()
        )

    # ========================================================
    # CFO
    # ========================================================

    imprimir_seccion(
        "8. CFO"
    )

    if "cfo" in df.columns:

        print(
            df["cfo"]
            .describe()
            .to_string()
        )

    # ========================================================
    # POSICIÓN
    # ========================================================

    imprimir_seccion(
        "9. POSICIÓN"
    )

    for column in [
        "latitude",
        "longitude",
    ]:

        if column in df.columns:

            print(
                f"\n{column}:"
            )

            print(
                df[column]
                .describe()
                .to_string()
            )

    # ========================================================
    # MOVIMIENTO
    # ========================================================

    imprimir_seccion(
        "10. MOVIMIENTO"
    )

    for column in [
        "velocity",
        "target_velocity",
    ]:

        if column in df.columns:

            print(
                f"\n{column}:"
            )

            print(
                df[column]
                .describe()
                .to_string()
            )

    # ========================================================
    # MUESTRAS IQ
    # ========================================================

    imprimir_seccion(
        "11. SEGMENTOS IQ"
    )

    for column in [
        "sample_start",
        "sample_count",
        "freq_lower_edge",
        "freq_upper_edge",
    ]:

        if column in df.columns:

            print(
                f"\n{column}:"
            )

            print(
                df[column]
                .describe()
                .to_string()
            )

    # ========================================================
    # DATATYPE
    # ========================================================

    imprimir_seccion(
        "12. FORMATO IQ"
    )

    if "datatype" in df.columns:

        print(
            df["datatype"]
            .value_counts(dropna=False)
            .to_string()
        )

        if (
            df["datatype"]
            .astype(str)
            .eq("cf32_le")
            .all()
        ):

            print(
                "\n✓ Todos los registros utilizan cf32_le."
            )

        else:

            print(
                "\n⚠ Existen otros formatos IQ."
            )

    # ========================================================
    # VALORES NULOS
    # ========================================================

    imprimir_seccion(
        "13. VALORES NULOS"
    )

    nulls = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    nulls = nulls[
        nulls > 0
    ]

    if len(nulls) == 0:

        print(
            "✓ No hay valores nulos."
        )

    else:

        print(
            nulls.to_string()
        )

    # ========================================================
    # INFINITOS
    # ========================================================

    imprimir_seccion(
        "14. VALORES INFINITOS"
    )

    numeric = df.select_dtypes(
        include=np.number
    )

    inf_count = np.isinf(
        numeric.to_numpy()
    ).sum()

    print(
        f"Valores infinitos: {inf_count}"
    )

    if inf_count == 0:

        print(
            "✓ No existen valores infinitos."
        )

    # ========================================================
    # DUPLICADOS
    # ========================================================

    imprimir_seccion(
        "15. DUPLICADOS"
    )

    duplicates = df.duplicated().sum()

    print(
        f"Filas duplicadas: {duplicates:,}"
    )

    # ========================================================
    # ARCHIVOS SIGMF
    # ========================================================

    imprimir_seccion(
        "16. ARCHIVOS SIGMF"
    )

    if "file" in df.columns:

        unique_files = (
            df["file"]
            .nunique()
        )

        print(
            f"Archivos .sigmf-meta referenciados: "
            f"{unique_files:,}"
        )

        missing_file_names = (
            df["file"]
            .isna()
            .sum()
        )

        print(
            f"Registros sin archivo: "
            f"{missing_file_names:,}"
        )

    # ========================================================
    # ALTITUD / ALTURA
    # ========================================================

    imprimir_seccion(
        "17. ALTITUD / ALTURA"
    )

    altitude_candidates = [
        column
        for column in df.columns
        if any(
            word in column.lower()
            for word in [
                "altitude",
                "altitud",
                "height",
                "altura",
                "elevation",
                "elevacion",
                "z"
            ]
        )
    ]

    if altitude_candidates:

        print(
            "Posibles columnas relacionadas "
            "con altura:"
        )

        for column in altitude_candidates:

            print(
                f"  ✓ {column}"
            )

    else:

        print(
            "⚠ No se encontró una columna "
            "explícita de altitud/altura."
        )

    # ========================================================
    # COORDENADAS POR TRANSMISIÓN
    # ========================================================

    imprimir_seccion(
        "18. CONSISTENCIA DE POSICIONES"
    )

    if {
        "transmission_idx",
        "latitude",
        "longitude"
    }.issubset(df.columns):

        positions = (
            df.groupby(
                "transmission_idx"
            )[[
                "latitude",
                "longitude"
            ]]
            .nunique()
        )

        same_position = (
            (positions["latitude"] == 1)
            &
            (positions["longitude"] == 1)
        )

        print(
            f"Transmisiones con una única "
            f"posición: {same_position.sum():,}"
        )

        print(
            f"Transmisiones con posiciones "
            f"variables: {(~same_position).sum():,}"
        )

    # ========================================================
    # RESUMEN FINAL
    # ========================================================

    imprimir_seccion(
        "RESUMEN"
    )

    checks = {
        "Archivo existe":
            METADATA_PATH.exists(),

        "Dataset no vacío":
            len(df) > 0,

        "Columnas completas":
            len(missing_columns) == 0,

        "Sin infinitos":
            inf_count == 0,

        "Tiene transmission_idx":
            "transmission_idx" in df.columns,

        "Tiene rrh_idx":
            "rrh_idx" in df.columns,

        "Tiene sample_start":
            "sample_start" in df.columns,

        "Tiene sample_count":
            "sample_count" in df.columns,
    }

    for name, status in checks.items():

        symbol = "✓" if status else "✗"

        print(
            f"{symbol} {name}"
        )

    print(
        "\nValidación finalizada."
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()