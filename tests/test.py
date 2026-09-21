from pathlib import Path
import ast

CARPETA = Path(
    r"C:\Users\Lenovo\OneDrive\Documentos\LORAIQ"
)

# Archivos que ya revisamos y NO necesitamos volver a mostrar
YA_REVISADOS = {
    "revisar_scripts.py",
    "test.py",
    "pros.py",
    "inspect_zip_old.py",
    "extraer_iq_prueba.py",
    "analizar_tx.py",
}

# ============================================================
# ANALIZAR UN ARCHIVO
# ============================================================

def analizar_archivo(ruta):

    contenido = ruta.read_text(
        encoding="utf-8",
        errors="replace"
    )

    resultado = {
        "archivo": ruta.name,
        "lineas": len(contenido.splitlines()),
        "tamaño_kb": ruta.stat().st_size / 1024,
        "imports": [],
        "funciones": [],
        "clases": [],
        "variables": [],
        "parseo_ok": True,
        "error": None,
    }

    try:

        tree = ast.parse(contenido)

        for nodo in ast.walk(tree):

            # -----------------------------
            # IMPORTS
            # -----------------------------

            if isinstance(nodo, ast.Import):

                for nombre in nodo.names:
                    resultado["imports"].append(
                        nombre.name
                    )

            elif isinstance(nodo, ast.ImportFrom):

                if nodo.module:
                    resultado["imports"].append(
                        nodo.module
                    )

            # -----------------------------
            # FUNCIONES
            # -----------------------------

            elif isinstance(
                nodo,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            ):

                resultado["funciones"].append(
                    nodo.name
                )

            # -----------------------------
            # CLASES
            # -----------------------------

            elif isinstance(nodo, ast.ClassDef):

                resultado["clases"].append(
                    nodo.name
                )

            # -----------------------------
            # VARIABLES
            # -----------------------------

            elif isinstance(nodo, ast.Assign):

                for target in nodo.targets:

                    if isinstance(
                        target,
                        ast.Name
                    ):
                        resultado["variables"].append(
                            target.id
                        )

    except SyntaxError as e:

        resultado["parseo_ok"] = False
        resultado["error"] = str(e)

    return resultado, contenido


# ============================================================
# BUSCAR OTROS PY
# ============================================================

archivos = sorted(
    ruta
    for ruta in CARPETA.glob("*.py")
    if ruta.name not in YA_REVISADOS
)

print("=" * 90)
print("REVISIÓN DE LOS SCRIPTS RESTANTES")
print("=" * 90)

print()
print("Carpeta:")
print(CARPETA)

print()
print("Archivos encontrados:")
print(len(archivos))


# ============================================================
# REVISIÓN
# ============================================================

for numero, ruta in enumerate(
    archivos,
    start=1
):

    resultado, contenido = analizar_archivo(ruta)

    print()
    print("=" * 90)
    print(
        f"[{numero}/{len(archivos)}] "
        f"{resultado['archivo']}"
    )
    print("=" * 90)

    print(
        f"Tamaño: "
        f"{resultado['tamaño_kb']:.2f} KB"
    )

    print(
        f"Líneas: "
        f"{resultado['lineas']}"
    )

    print(
        "Sintaxis: ",
        "OK"
        if resultado["parseo_ok"]
        else "ERROR"
    )

    if resultado["error"]:
        print(
            "Error:",
            resultado["error"]
        )

    # ========================================================
    # IMPORTS
    # ========================================================

    print()
    print("IMPORTS:")

    imports = sorted(
        set(resultado["imports"])
    )

    if imports:

        for item in imports:
            print(f"  - {item}")

    else:
        print("  Ninguno")

    # ========================================================
    # FUNCIONES
    # ========================================================

    print()
    print("FUNCIONES:")

    if resultado["funciones"]:

        for item in resultado["funciones"]:
            print(f"  - {item}")

    else:
        print("  Ninguna")

    # ========================================================
    # CLASES
    # ========================================================

    print()
    print("CLASES:")

    if resultado["clases"]:

        for item in resultado["clases"]:
            print(f"  - {item}")

    else:
        print("  Ninguna")

    # ========================================================
    # VARIABLES
    # ========================================================

    print()
    print("VARIABLES PRINCIPALES:")

    variables = sorted(
        set(resultado["variables"])
    )

    if variables:

        for item in variables[:50]:
            print(f"  - {item}")

        if len(variables) > 50:
            print(
                f"  ... y "
                f"{len(variables) - 50} más"
            )

    else:
        print("  Ninguna")

    # ========================================================
    # CONTENIDO COMPLETO
    # ========================================================

    print()
    print("-" * 90)
    print("CONTENIDO COMPLETO")
    print("-" * 90)

    for linea_num, linea in enumerate(
        contenido.splitlines(),
        start=1
    ):

        print(
            f"{linea_num:4}: {linea}"
        )


# ============================================================
# FIN
# ============================================================

print()
print("=" * 90)
print("FIN DE LA REVISIÓN")
print("=" * 90)