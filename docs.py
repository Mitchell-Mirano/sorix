import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# rutas fuente
dp_sorix_path = os.path.join(BASE_PATH, "sorix")

# rutas destino
docs_api_path = os.path.join(BASE_PATH, "docs", "api")

def create_docs(src_path, target_path, module_root):
    """
    Recorre src_path, crea estructura equivalente en target_path,
    y por cada archivo .py genera un .md con referencia mkdocstrings.
    """
    for root, _, files in os.walk(src_path):
        if "__pycache__" in root:
            continue

        # Crear carpeta equivalente en docs
        relative_dir = os.path.relpath(root, src_path)
        target_dir = os.path.join(target_path, relative_dir)
        os.makedirs(target_dir, exist_ok=True)

        # Procesar archivos .py
        for file in files:
            if file.endswith(".py") and not file.startswith("__") and not "cupy" in file:
                module_name = file[:-3]  # remover .py
                md_file_path = os.path.join(target_dir, f"{module_name}.md")

                # Construir nombre completo del módulo (import)
                relative_module = (
                    module_root +
                    ("." + relative_dir.replace(os.sep, ".")) if relative_dir != "." else module_root
                )
                full_module_path = f"{relative_module}.{module_name}"

                # Escribir archivo markdown
                with open(md_file_path, "w", encoding="utf-8") as md:
                    md.write(f"# {full_module_path}\n\n")
                    md.write(f"::: {full_module_path}\n")

                print(f"📄 Creado → {md_file_path}")


# Ejecutar para cada módulo
create_docs(dp_sorix_path, docs_api_path, "sorix")

print("🚀 Generación de documentación automática completada.")
