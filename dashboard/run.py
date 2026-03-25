import os
import sys
import subprocess

# Ruta al frontend donde estará el manage.py
ruta_frontend = os.path.join(os.path.dirname(__file__), "frontend")


def _interpreter_has_django(ruta_python):
    if not ruta_python or not os.path.exists(ruta_python):
        return False

    proceso = subprocess.run(
        [ruta_python, "-c", "import django"],
        capture_output=True,
        text=True,
    )
    return proceso.returncode == 0


def resolver_python_django():
    """
    Resuelve un intérprete con Django para iniciar el server.

    Prioridad:
    1) Variable de entorno NODEQUANT_PYTHON_DJANGO
    2) venv11 local del repositorio
    3) Intérprete actual
    """
    ruta_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidatos = []
    exe_env = os.environ.get("NODEQUANT_PYTHON_DJANGO", "").strip()
    if exe_env:
        candidatos.append(exe_env)

    candidatos.append(os.path.join(ruta_repo, "venv11", "Scripts", "python.exe"))
    candidatos.append(sys.executable)

    vistos = set()
    candidatos_unicos = []
    for c in candidatos:
        c_norm = os.path.normpath(c)
        if c_norm not in vistos:
            vistos.add(c_norm)
            candidatos_unicos.append(c_norm)

    for candidato in candidatos_unicos:
        if _interpreter_has_django(candidato):
            return candidato

    detalles = "\n".join(f"- {c}" for c in candidatos_unicos)
    raise RuntimeError(
        "No se encontró un intérprete con Django para iniciar el dashboard.\n"
        "Configura NODEQUANT_PYTHON_DJANGO o instala Django en venv11.\n"
        f"Candidatos evaluados:\n{detalles}"
    )

if __name__ == "__main__":
    os.chdir(ruta_frontend)
    python_django = resolver_python_django()
    print(f"Usando intérprete Django: {python_django}")
    subprocess.run([python_django, "manage.py", "runserver"], check=False)