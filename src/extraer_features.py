import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = Path.home() / "Downloads" / "sigmfs.zip"

METADATA_PATH = (
    ROOT / "data" / "metadata" / "lorawan_metadata.parquet"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "lora_features.parquet"
)

INCOMPLETE_PATH = (
    ROOT / "data" / "processed" / "iq_incompletos.csv"
)

ERRORS_PATH = (
    ROOT / "data" / "processed" / "iq_errores.csv"
)

N_BLOCKS = 8
BYTES_PER_SAMPLE = 8

MAX_ERROR_RATE = 0.05


def to_data_name(meta_name):

    name = str(meta_name).replace("\\", "/")

    if name.endswith(".sigmf-meta"):
        return name[: -len(".sigmf-meta")] + ".sigmf-data"

    if name.endswith(".sigmf-data"):
        return name

    return name + ".sigmf-data"


def calcular_features(iq, fs, n_blocks=N_BLOCKS):

    if iq is None or len(iq) == 0:
        return None

    amplitude = np.abs(iq)
    power = amplitude ** 2

    mean_power = float(np.mean(power))
    rms = float(np.sqrt(mean_power))
    max_amplitude = float(np.max(amplitude))
    mean_amplitude = float(np.mean(amplitude))
    std_amplitude = float(np.std(amplitude))

    crest_factor = (
        max_amplitude / rms
        if rms > 0
        else 0.0
    )

    iq_power_db = float(
        10 * np.log10(mean_power + 1e-30)
    )

    amplitude_p10 = float(np.percentile(amplitude, 10))
    amplitude_p25 = float(np.percentile(amplitude, 25))
    amplitude_p50 = float(np.percentile(amplitude, 50))
    amplitude_p75 = float(np.percentile(amplitude, 75))
    amplitude_p90 = float(np.percentile(amplitude, 90))

    i_mean = float(np.mean(iq.real))
    q_mean = float(np.mean(iq.imag))
    i_std = float(np.std(iq.real))
    q_std = float(np.std(iq.imag))

    power_std = float(np.std(power))

    power_cv = (
        power_std / mean_power
        if mean_power > 0
        else 0.0
    )

    block_features = {}

    blocks = np.array_split(power, n_blocks)

    block_powers_db = []

    for i, block in enumerate(blocks, start=1):

        if len(block) == 0:
            block_power_db = np.nan

        else:
            block_power = float(np.mean(block))
            block_power_db = float(
                10 * np.log10(block_power + 1e-30)
            )

        block_features[
            f"block_power_{i}_db"
        ] = block_power_db

        block_powers_db.append(block_power_db)

    valid_block_powers = [
        x
        for x in block_powers_db
        if np.isfinite(x)
    ]

    if valid_block_powers:
        power_min_db = float(np.min(valid_block_powers))
        power_max_db = float(np.max(valid_block_powers))
        power_range_db = power_max_db - power_min_db
    else:
        power_min_db = np.nan
        power_max_db = np.nan
        power_range_db = np.nan

    spectrum = np.fft.fftshift(np.fft.fft(iq))

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(len(iq), d=1 / float(fs))
    )

    spectrum_power = np.abs(spectrum) ** 2

    peak_index = int(np.argmax(spectrum_power))

    spectral_peak_frequency = float(frequencies[peak_index])

    spectral_peak_power_db = float(
        10 * np.log10(spectrum_power[peak_index] + 1e-30)
    )

    n_samples = int(len(iq))

    spectral_total_power_db = float(
        10 * np.log10(np.sum(spectrum_power) + 1e-30)
    )

    spectral_mean_power_db = float(
        10 * np.log10(
            np.mean(spectrum_power) + 1e-30
        )
    )

    features = {

        "iq_power": mean_power,
        "iq_power_db": iq_power_db,
        "iq_rms": rms,
        "iq_peak": max_amplitude,
        "amplitude_mean": mean_amplitude,
        "amplitude_std": std_amplitude,
        "amplitude_p10": amplitude_p10,
        "amplitude_p25": amplitude_p25,
        "amplitude_p50": amplitude_p50,
        "amplitude_p75": amplitude_p75,
        "amplitude_p90": amplitude_p90,
        "crest_factor": crest_factor,
        "i_mean": i_mean,
        "q_mean": q_mean,
        "i_std": i_std,
        "q_std": q_std,
        "power_std": power_std,
        "power_cv": power_cv,
        "power_min_db": power_min_db,
        "power_max_db": power_max_db,
        "power_range_db": power_range_db,
        "spectral_peak_frequency": spectral_peak_frequency,
        "spectral_peak_power_db": spectral_peak_power_db,
        "spectral_total_power_db": spectral_total_power_db,
        "spectral_mean_power_db": spectral_mean_power_db,
        "n_samples": n_samples
    }

    features.update(block_features)

    return features


def main():

    print("=" * 70)
    print("EXTRACCIÓN DE CARACTERÍSTICAS IQ")
    print("=" * 70)

    print(f"ZIP:      {ZIP_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Salida:   {OUTPUT_PATH}")

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el ZIP:\n{ZIP_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la metadata:\n{METADATA_PATH}"
        )

    df_metadata = pd.read_parquet(METADATA_PATH)

    print("\nMetadata cargada")
    print(f"Filas: {len(df_metadata):,}")

    required_columns = [
        "file",
        "transmission_idx",
        "rrh_idx",
        "sample_start",
        "sample_count",
        "sample_rate"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df_metadata.columns
    ]

    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    print("\nSample rate del dataset:")
    print(
        df_metadata["sample_rate"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    results = []
    errors = []
    incomplete = []

    total = len(df_metadata)
    t_start = time.time()

    with ZipFile(ZIP_PATH, "r") as zf:

        for position, (_, row) in enumerate(
            df_metadata.iterrows(),
            start=1
        ):

            data_file_meta = row["file"]

            sample_start = int(row["sample_start"])
            sample_count = int(row["sample_count"])
            fs = float(row["sample_rate"])

            try:

                data_file = to_data_name(data_file_meta)

                info = zf.getinfo(data_file)

                file_size = info.file_size

                offset = sample_start * BYTES_PER_SAMPLE
                bytes_requested = sample_count * BYTES_PER_SAMPLE
                bytes_available = file_size - offset

                if bytes_available < bytes_requested:

                    incomplete.append({
                        "file": data_file_meta,
                        "data_file": data_file,
                        "sample_start": sample_start,
                        "sample_count": sample_count,
                        "file_size": file_size,
                        "bytes_available": max(bytes_available, 0),
                        "expected_bytes": bytes_requested,
                        "samples_available": max(
                            bytes_available // BYTES_PER_SAMPLE,
                            0
                        ),
                        "samples_expected": sample_count
                    })

                    continue

                with zf.open(data_file) as f:
                    f.seek(offset)
                    raw = f.read(bytes_requested)

                iq = np.frombuffer(raw, dtype="<c8")

                if len(iq) != sample_count:

                    incomplete.append({
                        "file": data_file_meta,
                        "data_file": data_file,
                        "sample_start": sample_start,
                        "sample_count": sample_count,
                        "file_size": file_size,
                        "bytes_available": bytes_available,
                        "expected_bytes": bytes_requested,
                        "samples_available": len(iq),
                        "samples_expected": sample_count
                    })

                    continue

                features = calcular_features(
                    iq,
                    fs=fs,
                    n_blocks=N_BLOCKS
                )

                if features is None:
                    continue

                result = row.to_dict()
                result.update(features)

                results.append(result)

            except Exception as error:

                errors.append({
                    "file": data_file_meta,
                    "error": str(error)
                })

                if (
                    len(errors) > 100
                    and len(errors) / position > MAX_ERROR_RATE
                ):

                    print(
                        f"\n⚠️  Tasa de errores > "
                        f"{MAX_ERROR_RATE:.0%} "
                        f"({len(errors)}/{position}). "
                        f"Abortando."
                    )
                    print(
                        f"Primer error: "
                        f"{errors[0]['file']} → "
                        f"{errors[0]['error']}"
                    )
                    sys.exit(1)

            if (
                position % 1000 == 0
                or position == total
            ):

                elapsed = time.time() - t_start
                rate = position / elapsed if elapsed > 0 else 0

                eta = (
                    (total - position) / rate
                    if rate > 0
                    else float("inf")
                )

                print(
                    f"Procesados: {position:,}/{total:,}  "
                    f"({position/total:5.1%})  "
                    f"|  {rate:6.1f} filas/s  "
                    f"|  ETA: {eta/60:5.1f} min"
                )

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)

    print(f"Metadata:    {len(df_metadata):,}")
    print(f"Procesados:  {len(df):,}")
    print(f"Incompletos: {len(incomplete):,}")
    print(f"Errores:     {len(errors):,}")
    print(f"Columnas:    {len(df.columns)}")

    if incomplete:

        print("\nRegistros IQ incompletos:")

        for item in incomplete[:10]:

            print(f"\n  {item['file']}")
            print(
                f"    Muestras esperadas: "
                f"{item['samples_expected']:,}"
            )
            print(
                f"    Muestras disponibles: "
                f"{item['samples_available']:,}"
            )

        if len(incomplete) > 10:
            print(
                f"\n  ... y "
                f"{len(incomplete) - 10:,} "
                f"registros adicionales."
            )

    if errors:

        print("\nErrores encontrados:")

        for error in errors[:10]:
            print(f"  {error['file']}: {error['error']}")

    if len(df) > 0:

        numeric_columns = (
            df.select_dtypes(include=np.number).columns
        )

        nan_count = int(
            df[numeric_columns].isna().sum().sum()
        )

        inf_count = int(
            np.isinf(
                df[numeric_columns].to_numpy()
            ).sum()
        )

        print(f"\nNaN numéricos: {nan_count:,}")
        print(f"Inf numéricos: {inf_count:,}")

        rrh_count = (
            df.groupby("transmission_idx")["rrh_idx"].nunique()
        )

        print("\nTransmisiones:")
        print(
            f"  Total: "
            f"{df['transmission_idx'].nunique():,}"
        )
        print(
            f"  Con múltiples RRH: "
            f"{(rrh_count > 1).sum():,}"
        )

        if "area_type" in df.columns:
            print("\nEscenarios:")
            print(
                df["area_type"]
                .value_counts()
                .to_string()
            )

        print("\nRegistros por RRH:")
        print(
            df["rrh_idx"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("\nSample rate procesado:")
        print(
            df["sample_rate"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    if incomplete:
        pd.DataFrame(incomplete).to_csv(
            INCOMPLETE_PATH, index=False
        )

    if errors:
        pd.DataFrame(errors).to_csv(
            ERRORS_PATH, index=False
        )

    print("\nArchivo guardado en:")
    print(OUTPUT_PATH)

    if incomplete:
        print(
            f"\nDiagnóstico incompletos:\n"
            f"{INCOMPLETE_PATH}"
        )

    if errors:
        print(
            f"\nDiagnóstico errores:\n"
            f"{ERRORS_PATH}"
        )

    print("\nProceso finalizado correctamente.")


if __name__ == "__main__":
    main()