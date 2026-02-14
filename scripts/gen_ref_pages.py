from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("sorix").rglob("*.py")):
    module_path = path.with_suffix("")
    doc_path = path.relative_to("sorix").with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts.pop()
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # Clean navigation parts (remove root package name if it's 'sorix')
    nav_parts = list(parts)
    if nav_parts[0] == "sorix":
        nav_parts.pop(0)
    
    # If it's the root __init__, it becomes empty, so we name it after the package
    if not nav_parts:
        nav_parts = ["sorix"]

    nav[tuple(nav_parts)] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

generated_nav = nav.build_literate_nav()
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(generated_nav)
