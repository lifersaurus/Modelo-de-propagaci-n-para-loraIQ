import zipfile
import json
import re
from collections import defaultdict

ZIP_PATH = r"C:\Users\Lenovo\Downloads\sigmfs.zip"

# Palabras que podrían identificar coordenadas o ubicación del receptor
KEYWORDS = [
    "rrh_lat",
    "rrh_lon",
    "rrh_latitude",
    "rrh_longitude",
    "rx_lat",
    "rx_lon",
    "rx_latitude",
    "rx_longitude",
    "receiver",
    "receiver_lat",
    "receiver_lon",
    "receiver_latitude",
    "receiver_longitude",
    "antenna",
    "position",
    "latitude",
    "longitude",
    "location",
    "coordinates",
]

def search_recursive(obj, path=""):
    """
    Busca claves relacionadas con posición dentro de cualquier
    estructura JSON.
    """
    results = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()

            if any(keyword in key_lower for keyword in KEYWORDS):
                results.append((path + "/" + str(key), value))

            results.extend(
                search_recursive(value, path + "/" + str(key))
            )

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            results.extend(
                search_recursive(value, path + f"[{i}]")
            )

    return results


print("========================================")
print("BUSCANDO INFORMACIÓN DE POSICIÓN DE RRH")
print("========================================")

with zipfile.ZipFile(ZIP_PATH, "r") as zf:

    meta_files = [
        name for name in zf.namelist()
        if name.lower().endswith(".sigmf-meta")
    ]

    print("Archivos metadata:", len(meta_files))

    encontrados = defaultdict(list)

    # Solo revisaremos una muestra de metadata inicialmente.
    # Si encontramos algo interesante, hacemos una búsqueda completa.
    muestra = meta_files[:1000]

    for name in muestra:

        try:
            with zf.open(name) as f:
                meta = json.load(f)

            results = search_recursive(meta)

            if results:
                for path, value in results:
                    encontrados[path].append(
                        (name, value)
                    )

        except Exception as e:
            print("Error:", name, e)


print("\n========================================")
print("CAMPOS RELACIONADOS CON POSICIÓN")
print("========================================")

if not encontrados:
    print("No se encontraron campos adicionales en la muestra.")

else:
    for path, values in encontrados.items():

        print(f"\nCAMPO: {path}")
        print(f"Ocurrencias: {len(values)}")

        for filename, value in values[:5]:
            print("  Archivo:", filename)
            print("  Valor:", value)


print("\n========================================")
print("BUSCANDO INFORMACIÓN EN NOMBRES DE ARCHIVO")
print("========================================")

with zipfile.ZipFile(ZIP_PATH, "r") as zf:

    names = zf.namelist()

    # Mostrar carpetas principales
    folders = set()

    for name in names:
        parts = name.split("/")

        if len(parts) >= 3:
            folders.add("/".join(parts[:3]))

    for folder in sorted(folders):
        print(folder)

print("\n========================================")
print("FIN")
print("========================================")