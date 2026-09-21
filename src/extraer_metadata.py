import json
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = Path.home() / "Downloads" / "sigmfs.zip"

OUTPUT = (
    ROOT
    / "data"
    / "metadata"
    / "lorawan_metadata.parquet"
)

def parse_comment(comment):
    """
    Convierte el contenido tabulado de core:comment
    en un diccionario.
    """

    data = {}

    if not comment:
        return data

    for line in comment.splitlines():

        if "\t" not in line:
            continue

        key, value = line.split("\t", 1)

        data[key.strip()] = value.strip()

    return data

def main():

    print("=" * 70)
    print("EXTRACCIÓN DE METADATA SIGMF")
    print("=" * 70)

    print(f"ZIP:    {ZIP_PATH}")
    print(f"Salida: {OUTPUT}")

    #

    if not ZIP_PATH.exists():

        raise FileNotFoundError(
            f"No se encontró el archivo ZIP:\n{ZIP_PATH}"
        )

    results = []


    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zf:

        meta_files = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".sigmf-meta")
        ]

        print(
            f"\nMetadata encontrados: "
            f"{len(meta_files):,}"
        )


        for name in tqdm(
            meta_files,
            desc="Procesando metadata"
        ):

            try:

                with zf.open(name) as f:

                    meta = json.load(f)

                global_data = meta.get(
                    "global",
                    {}
                )

                annotations = meta.get(
                    "annotations",
                    []
                )

                if not annotations:
                    continue

                ann = annotations[0]

                comment = parse_comment(
                    ann.get(
                        "core:comment",
                        ""
                    )
                )

          

                row = {

             

                    "file": name,

             

                    "datetime":
                        global_data.get(
                            "core:datetime"
                        ),

                    "datatype":
                        global_data.get(
                            "core:datatype"
                        ),

                    "frequency_MHz":
                        global_data.get(
                            "core:frequency"
                        ),

                    "sample_rate":
                        global_data.get(
                            "core:sample_rate"
                        ),

              

                    "transmission_idx":
                        comment.get(
                            "transmission_idx"
                        ),

                    "rrh_idx":
                        comment.get(
                            "rrh_idx"
                        ),

                    "bandwidth":
                        comment.get(
                            "bandwidth"
                        ),

                    "sf":
                        comment.get(
                            "sf"
                        ),

                    "cr":
                        comment.get(
                            "cr"
                        ),

                    "fc":
                        comment.get(
                            "fc"
                        ),

                    "cfo":
                        comment.get(
                            "cfo"
                        ),

                    "snr":
                        comment.get(
                            "snr"
                        ),

                   

                    "latitude":
                        comment.get(
                            "latitude"
                        ),

                    "longitude":
                        comment.get(
                            "longitude"
                        ),

                 
                    "velocity":
                        comment.get(
                            "velocity"
                        ),

                    "target_velocity":
                        comment.get(
                            "target_velocity"
                        ),

            

                    "area":
                        comment.get(
                            "area"
                        ),

                    "area_type":
                        comment.get(
                            "area_type"
                        ),

                    "name":
                        comment.get(
                            "name"
                        ),

                   

                    "sample_start":
                        ann.get(
                            "core:sample_start"
                        ),

                    "sample_count":
                        ann.get(
                            "core:sample_count"
                        ),

                    "freq_lower_edge":
                        ann.get(
                            "core:freq_lower_edge"
                        ),

                    "freq_upper_edge":
                        ann.get(
                            "core:freq_upper_edge"
                        )
                }

                results.append(row)

            except Exception as error:

                print(
                    f"\nError en {name}: {error}"
                )



    df = pd.DataFrame(results)


    numeric_columns = [

        "transmission_idx",
        "rrh_idx",

        "frequency_MHz",
        "sample_rate",

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

        "sample_start",
        "sample_count",

        "freq_lower_edge",
        "freq_upper_edge"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )



    if {
        "transmission_idx",
        "rrh_idx"
    }.issubset(df.columns):

        df = df.sort_values(
            [
                "transmission_idx",
                "rrh_idx"
            ]
        ).reset_index(
            drop=True
        )

  

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT,
        index=False
    )


    print("\n" + "=" * 70)
    print("PROCESAMIENTO TERMINADO")
    print("=" * 70)

    print(
        f"Filas: {len(df):,}"
    )

    print(
        f"Columnas: {len(df.columns)}"
    )

    print("\nColumnas:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    print("\nPrimeras filas:")

    print(
        df.head().to_string()
    )



    if "area_type" in df.columns:

        print("\nEscenarios:")

        print(
            df["area_type"]
            .value_counts()
            .to_string()
        )

    if "rrh_idx" in df.columns:

        print("\nRRH:")

        print(
            df["rrh_idx"]
            .value_counts()
            .sort_index()
            .to_string()
        )

  

    if "transmission_idx" in df.columns:

        print(
            "\nTransmisiones únicas:",
            df["transmission_idx"].nunique()
        )


    print("\nValores faltantes:")

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing[missing > 0]
        .to_string()
        if (missing > 0).any()
        else "No hay valores faltantes."
    )

    print(
        "\nGuardado en:"
    )

    print(OUTPUT)



if __name__ == "__main__":
    main()