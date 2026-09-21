from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data" / "metadata" / "lorawan_metadata.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "lora_features.parquet"
INCOMPLETE_PATH = ROOT / "data" / "processed" / "iq_incompletos.csv"
ERRORS_PATH = ROOT / "data" / "processed" / "iq_errores.csv"
REPORT_PATH = ROOT / "results" / "audit" / "comparacion_metadata_features.csv"


def index_por_fila(df):
    """Índice único por (file, rrh_idx) — no solo file."""
    return set(
        zip(
            df["file"].astype(str),
            df["rrh_idx"].astype(int),
        )
    )


def main():

    print("=" * 70)
    print("COMPARACIÓN METADATA VS FEATURES")
    print("=" * 70)

    metadata = pd.read_parquet(METADATA_PATH)
    features = pd.read_parquet(FEATURES_PATH)

    print(f"\nMetadata: {len(metadata):,} registros")
    print(f"Features: {len(features):,} registros")

    idx_meta = index_por_fila(metadata)
    idx_feat = index_por_fila(features)

    missing = sorted(idx_meta - idx_feat)
    extra = sorted(idx_feat - idx_meta)

    print("\n" + "=" * 70)
    print("REGISTROS FALTANTES")
    print("=" * 70)

    print(
        f"\nFilas (file, rrh) en metadata "
        f"pero no en features: {len(missing):,}"
    )

    for file, rrh in missing[:20]:
        print(f"  {file}  (RRH {rrh})")

    if len(missing) > 20:
        print(f"  ... y {len(missing) - 20:,} más")

    print("\n" + "=" * 70)
    print("REGISTROS EXTRA")
    print("=" * 70)

    print(
        f"\nFilas (file, rrh) en features "
        f"pero no en metadata: {len(extra):,}"
    )

    for file, rrh in extra[:20]:
        print(f"  {file}  (RRH {rrh})")


    if INCOMPLETE_PATH.exists():

        df_inc = pd.read_csv(INCOMPLETE_PATH)
        print(f"\nIncompletos documentados: {len(df_inc):,}")

    else:
        print("\n⚠️  No hay archivo de incompletos")

    if ERRORS_PATH.exists():

        df_err = pd.read_csv(ERRORS_PATH)
        print(f"Errores documentados: {len(df_err):,}")

    else:
        print("⚠️  No hay archivo de errores")



    print("\n" + "=" * 70)
    print("DIFERENCIAS POR RRH")
    print("=" * 70)

    meta_rrh = metadata["rrh_idx"].value_counts().sort_index()
    feat_rrh = features["rrh_idx"].value_counts().sort_index()

    diff_rrh = pd.DataFrame({
        "metadata": meta_rrh,
        "features": feat_rrh,
    }).fillna(0).astype(int)

    diff_rrh["perdidas"] = (
        diff_rrh["metadata"] - diff_rrh["features"]
    )
    diff_rrh["pct"] = (
        diff_rrh["perdidas"] / diff_rrh["metadata"] * 100
    ).round(2)

    print(diff_rrh.to_string())

    print("\n" + "=" * 70)
    print("DIFERENCIAS POR ESCENARIO")
    print("=" * 70)

    meta_area = metadata["area_type"].value_counts()
    feat_area = features["area_type"].value_counts()

    diff_area = pd.DataFrame({
        "metadata": meta_area,
        "features": feat_area,
    }).fillna(0).astype(int)

    diff_area["perdidas"] = (
        diff_area["metadata"] - diff_area["features"]
    )
    diff_area["pct"] = (
        diff_area["perdidas"] / diff_area["metadata"] * 100
    ).round(2)

    print(diff_area.to_string())


    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if missing:

        pd.DataFrame(
            missing,
            columns=["file", "rrh_idx"]
        ).to_csv(REPORT_PATH, index=False)

        print(f"\nReporte guardado en:\n{REPORT_PATH}")

    print("\n" + "=" * 70)
    print("COMPARACIÓN FINALIZADA")
    print("=" * 70)


if __name__ == "__main__":
    main()