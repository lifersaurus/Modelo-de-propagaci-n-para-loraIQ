from pathlib import Path
from zipfile import ZipFile
import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = Path.home() / "Downloads" / "sigmfs.zip"

METADATA_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "lorawan_metadata.parquet"
)

TARGET = "sigmfs/25_01_15-10_17/rrh1/10.sigmf-meta"

df = pd.read_parquet(METADATA_PATH)

row = df[df["file"] == TARGET]


print("=" * 70)
print("DIAGNÓSTICO DEL REGISTRO FALTANTE")
print("=" * 70)

if row.empty:
    print("\nERROR: el registro no existe en lorawan_metadata.parquet")
    raise SystemExit

row = row.iloc[0]

print("\nMETADATA:")
print("-" * 70)

for column in df.columns:
    print(f"{column}: {row[column]}")


print("\n" + "=" * 70)
print("ARCHIVOS IQ")
print("=" * 70)

with ZipFile(ZIP_PATH, "r") as zf:

    names = set(zf.namelist())

    meta_name = TARGET

    data_name = TARGET.replace(
        ".sigmf-meta",
        ".sigmf-data"
    )

    print("\nMetadata:")
    print(meta_name)

    print("\nIQ esperado:")
    print(data_name)

    print(
        "\nMetadata existe:",
        meta_name in names
    )

    print(
        "IQ existe:",
        data_name in names
    )

    if data_name in names:

        info = zf.getinfo(data_name)

        print("\nTamaño del IQ dentro del ZIP:")
        print(f"{info.file_size:,} bytes")

        sample_start = int(row["sample_start"])
        sample_count = int(row["sample_count"])
        bytes_per_sample = 8

        offset = sample_start * bytes_per_sample
        requested_bytes = sample_count * bytes_per_sample

        print("\nLectura solicitada:")
        print(f"Sample start: {sample_start:,}")

        print(f"Sample count: {sample_count:,}")

        print(f"Offset: {offset:,} bytes")

        print(f"Bytes solicitados: {requested_bytes:,}")

        print(f"Bytes disponibles: {info.file_size:,}")

        with zf.open(data_name) as f:

            f.seek(offset)

            raw = f.read(requested_bytes)

        print("\nBytes realmente leídos:", len(raw))

        iq = np.frombuffer(raw, dtype="<c8")

        print("Muestras IQ obtenidas:", len(iq))

        if len(iq) > 0:

            print("\nPrimeras 10 muestras IQ:")
            print(iq[:10])

            power = np.abs(iq) ** 2

            print("\nPotencia media:", np.mean(power))
            print("Potencia máxima:", np.max(power))

    if meta_name in names:

        print("\n" + "=" * 70)
        print("SIGMF-META ORIGINAL")
        print("=" * 70)

        with zf.open(meta_name) as f:

            meta = json.load(f)

        print(
            json.dumps(
                meta,
                indent=2,
                ensure_ascii=False
            )
        )


print("\n" + "=" * 70)
print("DIAGNÓSTICO FINALIZADO")
print("=" * 70)