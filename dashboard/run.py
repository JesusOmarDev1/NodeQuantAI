import os
import sys
import subprocess

# Ruta al frontend donde estará el manage.py
ruta_frontend = os.path.join(os.path.dirname(__file__), "frontend")

if __name__ == "__main__":
    os.chdir(ruta_frontend)
    subprocess.run([sys.executable, "manage.py", "runserver"])