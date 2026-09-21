import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = Path.home() / "Downloads" / "sigmfs.zip"

METADATA_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "lorawan_metadata.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "iq"
)

FS = 250_000

BYTES_PER_SAMPLE = 8

TRANSMISSION_ID = (
    int(sys.argv[1])
    if len(sys.argv) > 1
    else 0
)


def to_data_name(meta_name):

    name = meta_name.replace("\\", "/")

    if name.endswith(".sigmf-meta"):
        return (
            name[: -len(".sigmf-meta")]
            + ".sigmf-data"
        )

    if name.endswith(".sigmf-data"):
        return name

    return name + ".sigmf-data"


def leer_iq(zf, data_file, sample_start, sample_count):

    name = to_data_name(data_file)

    offset = int(sample_start) * BYTES_PER_SAMPLE
    size = int(sample_count) * BYTES_PER_SAMPLE

    with zf.open(name) as f:

        f.seek(offset)

        raw = f.read(size)

    if len(raw) != size:

        raise ValueError(
            f"Lectura incompleta en {name}: "
            f"{len(raw)} de {size} bytes "
            f"(offset={offset})"
        )

    return np.frombuffer(
        raw,
        dtype="<c8"
    )


def main():

    if not ZIP_PATH.exists():

        raise FileNotFoundError(
            f"No se encontró el ZIP:\n{ZIP_PATH}"
        )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"No se encontró la metadata:\n"
            f"{METADATA_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_parquet(
        METADATA_PATH
    )

    tx = df[
        df["transmission_idx"]
        == TRANSMISSION_ID
    ]

    if tx.empty:

        raise ValueError(
            f"No existe transmission_idx="
            f"{TRANSMISSION_ID}"
        )

    print("=" * 70)
    print(
        f"TRANSMISIÓN {TRANSMISSION_ID} — "
        f"{len(tx)} RRH"
    )
    print("=" * 70)

    with ZipFile(ZIP_PATH, "r") as zf:

        for _, row in tx.iterrows():

            iq = leer_iq(
                zf,
                row["file"],
                row["sample_start"],
                row["sample_count"]
            )

            time_ms = (
                np.arange(len(iq))
                / FS
                * 1000
            )

            amplitude = np.abs(iq)

            fig, axes = plt.subplots(
                3,
                1,
                figsize=(12, 8),
                sharex=True
            )

            axes[0].plot(
                time_ms,
                iq.real,
                linewidth=0.6
            )

            axes[0].set_ylabel("I")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(
                time_ms,
                iq.imag,
                linewidth=0.6
            )

            axes[1].set_ylabel("Q")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(
                time_ms,
                amplitude,
                linewidth=0.6
            )

            axes[2].set_ylabel("|IQ|")
            axes[2].set_xlabel("Tiempo [ms]")
            axes[2].grid(True, alpha=0.3)

            duration_ms = (
                time_ms[-1]
                if len(time_ms) > 0
                else 0.0
            )

            fig.suptitle(
                f"TX {TRANSMISSION_ID} — "
                f"RRH {int(row['rrh_idx'])} — "
                f"n={len(iq)} muestras "
                f"({duration_ms:.2f} ms)"
            )

            fig.tight_layout()

            output = (
                OUTPUT_DIR
                / f"tx_{TRANSMISSION_ID}"
                f"_rrh_{int(row['rrh_idx'])}.png"
            )

            fig.savefig(
                output,
                dpi=150
            )

            plt.close(fig)

            print(f"Generado: {output.name}")

    print(
        f"\nFiguras guardadas en:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()