import os
import json
import joblib
import numpy as np
import pandas as pd
import subprocess
import sys
from typing import List, Tuple

# --- AJUSTE DE RUTAS ---
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ruta_raiz not in sys.path:
    sys.path.insert(0, ruta_raiz)

# Importación absoluta desde la raíz del proyecto
from regression.scripts.modelo_regresion import crear_features_derivadas


class MotorInferenciaNodeQuant:
    def __init__(self, ruta_reg, ruta_clf):
        """
        Carga los modelos de ambas carpetas de forma independiente.
        """
        self.ruta_reg = ruta_reg
        self.ruta_clf = ruta_clf
        self.modelos_regresion = {}
        self.modelo_clasificacion = None
        self.python_radiomics_exe = self._resolver_python_radiomics()

        print("Cargando cerebro de NodeQuant AI...")
        print(f"  [OK] Intérprete radiomics seleccionado: {self.python_radiomics_exe}")

        # 1. Cargar los 3 modelos de regresión
        targets = ["volumen", "eje_corto", "eje_largo"]
        for t in targets:
            ruta_joblib = os.path.join(self.ruta_reg, f"modelo_{t}.joblib")
            ruta_json = os.path.join(self.ruta_reg, f"metadata_{t}.json")

            if os.path.exists(ruta_joblib) and os.path.exists(ruta_json):
                with open(ruta_json, 'r') as f:
                    metadata = json.load(f)

                self.modelos_regresion[t] = {
                    "modelo": joblib.load(ruta_joblib),
                    "metadata": metadata
                }
                print(f"  [OK] Modelo regresión {t} cargado.")
            else:
                print(f"  [ALERTA] Faltan archivos para {t} en {self.ruta_reg}")

        # 2. Cargar modelo de Clasificación de Riesgo (Independiente del bucle anterior)
        ruta_clf_joblib = os.path.join(self.ruta_clf, "modelo_riesgo.joblib")
        ruta_clf_json = os.path.join(self.ruta_clf, "metadata_riesgo.json")

        if os.path.exists(ruta_clf_joblib) and os.path.exists(ruta_clf_json):
            with open(ruta_clf_json, 'r') as f:
                meta_clf = json.load(f)
            self.modelo_clasificacion = {
                "modelo": joblib.load(ruta_clf_joblib),
                "metadata": meta_clf
            }
            print("  [OK] Modelo clasificación de riesgo cargado.")
        else:
            print(f"  [ALERTA] Faltan archivos para clasificación en {self.ruta_clf}")

    def _probar_radiomics_en_interprete(self, ruta_python: str) -> Tuple[bool, str]:
        """Verifica que un intérprete pueda importar radiomics."""
        if not ruta_python or not os.path.exists(ruta_python):
            return False, "No existe"

        prueba = subprocess.run(
            [ruta_python, "-c", "import radiomics"],
            capture_output=True,
            text=True
        )

        if prueba.returncode == 0:
            return True, "OK"

        detalle = (prueba.stderr or prueba.stdout or "Import fallido").strip()
        return False, detalle

    def _resolver_python_radiomics(self) -> str:
        """
        Resuelve el intérprete con fallback para equipos con venv o Anaconda.

        Prioridad:
        1) Variable de entorno NODEQUANT_PYTHON_RADIOMICS
        2) venv9 del repositorio
        3) Python del entorno Conda activo
        4) Rutas comunes de Anaconda/Miniconda
        5) Intérprete actual (sys.executable)
        """
        candidatos: List[str] = []

        # 1) Override explícito por variable de entorno (evita interferir entre equipos)
        exe_env = os.environ.get("NODEQUANT_PYTHON_RADIOMICS", "").strip()
        if exe_env:
            candidatos.append(exe_env)

        # 2) venv9 local del repositorio
        candidatos.append(os.path.join(ruta_raiz, "sic_9", "Scripts", "python.exe"))

        # 3) Entorno Conda activo
        conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
        if conda_prefix:
            candidatos.append(os.path.join(conda_prefix, "python.exe"))

        # 4) Rutas comunes de Anaconda/Miniconda por usuario actual
        user_home = os.path.expanduser("~")
        bases_conda = ["anaconda3", "miniconda3"]
        envs_conda = ["sic_9", "py39", "venv9", "nodequant", "radiomics", "base"]
        for base in bases_conda:
            candidatos.append(os.path.join(user_home, base, "python.exe"))
            for env_name in envs_conda:
                candidatos.append(os.path.join(user_home, base, "envs", env_name, "python.exe"))

        # 5) Intérprete actual como último recurso
        candidatos.append(sys.executable)

        # Mantener orden y eliminar duplicados
        candidatos_unicos: List[str] = []
        vistos = set()
        for c in candidatos:
            c_norm = os.path.normpath(c)
            if c_norm not in vistos:
                vistos.add(c_norm)
                candidatos_unicos.append(c_norm)

        errores = []
        for candidato in candidatos_unicos:
            ok, detalle = self._probar_radiomics_en_interprete(candidato)
            if ok:
                return candidato
            errores.append(f"- {candidato}: {detalle}")

        raise RuntimeError(
            "No se encontró un intérprete válido para PyRadiomics. "
            "Configura NODEQUANT_PYTHON_RADIOMICS con la ruta de python.exe.\n"
            + "\n".join(errores)
        )

    def generar_visualizacion(self, ruta_imagen, ruta_mascara):
        """
        Genera imágenes CT (original + overlay) mediante un subproceso con Python 3.9.
        Retorna dict con imágenes base64 y metadata, o None si falla.
        """
        script_viz = os.path.join(os.path.dirname(__file__), "visualizador_nifti.py")

        if not os.path.exists(script_viz):
            print("  [WARN] visualizador_nifti.py no encontrado, omitiendo visualización.")
            return None

        try:
            proceso = subprocess.run(
                [self.python_radiomics_exe, script_viz, ruta_imagen, ruta_mascara],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.path.dirname(__file__)
            )

            if proceso.returncode != 0:
                print(f"  [WARN] Visualización falló: {proceso.stderr[:200]}")
                return None

            resultado = json.loads(proceso.stdout)
            if "error" in resultado:
                print(f"  [WARN] Visualización error: {resultado['error']}")
                return None

            return resultado

        except subprocess.TimeoutExpired:
            print("  [WARN] Visualización excedió timeout de 60s.")
            return None
        except Exception as e:
            print(f"  [WARN] Visualización excepción: {e}")
            return None

    def extraer_radiomica(self, ruta_imagen, ruta_mascara):
        """
        Llama al extractor de PyRadiomics mediante un subproceso.
        """
        script_extractor = os.path.join(os.path.dirname(__file__), "extractor_py39.py")

        if not os.path.exists(script_extractor):
            raise FileNotFoundError(f"No existe el script extractor: {script_extractor}")
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"No existe la imagen de entrada: {ruta_imagen}")
        if not os.path.exists(ruta_mascara):
            raise FileNotFoundError(f"No existe la máscara de entrada: {ruta_mascara}")

        proceso = subprocess.run(
            [self.python_radiomics_exe, script_extractor, ruta_imagen, ruta_mascara],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )

        if proceso.returncode != 0:
            raise Exception(
                "Fallo en el extractor de radiomics.\n"
                f"[Python]: {self.python_radiomics_exe}\n"
                f"[STDERR]: {proceso.stderr}\n"
                f"[STDOUT]: {proceso.stdout}"
            )

        try:
            resultado_json = json.loads(proceso.stdout)
        except json.JSONDecodeError:
            raise Exception(f"Salida no válida de PyRadiomics: {proceso.stdout}")

        if "error" in resultado_json:
            raise Exception(f"Error interno en PyRadiomics: {resultado_json['error']}")

        return pd.DataFrame([resultado_json])

    def predecir_paciente(self, ruta_imagen, ruta_mascara):
        """Ejecuta todo el pipeline: Extracción -> Derivadas -> Regresión -> Clasificación"""

        # 1. Extracción y Derivadas
        df_crudo = self.extraer_radiomica(ruta_imagen, ruta_mascara)
        df_procesado = crear_features_derivadas(df_crudo)

        resultados_finales = {}

        # 2. Predicciones de Regresión
        for target, info in self.modelos_regresion.items():
            modelo = info["modelo"]
            meta = info["metadata"]
            X_input = df_procesado.reindex(columns=meta["features_entrada"], fill_value=0)

            pred_raw = modelo.predict(X_input)[0]
            if meta.get("use_log", False):
                pred_final = np.maximum(0, np.expm1(pred_raw))
            else:
                pred_final = np.maximum(0, pred_raw)

            resultados_finales[target] = {
                "valor": round(pred_final, 2),
                "unidad": meta["unidad"]
            }

        # 3. Predicción de Clasificación
        if self.modelo_clasificacion:
            modelo_clf = self.modelo_clasificacion["modelo"]
            meta_clf = self.modelo_clasificacion["metadata"]

            X_clf = df_procesado.reindex(columns=meta_clf["features_entrada"], fill_value=0)
            riesgo_idx = modelo_clf.predict(X_clf)[0]

            mapa_riesgo = {0: "Bajo", 1: "Intermedio", 2: "Crítico"}
            resultados_finales["riesgo"] = mapa_riesgo.get(int(riesgo_idx), "Desconocido")
        else:
            resultados_finales["riesgo"] = "Modelo de clasificación no cargado"

        return resultados_finales


# =====================================================================
if __name__ == "__main__":
    # Definimos las rutas absolutas a las dos carpetas de modelos
    RUTA_REG = os.path.join(ruta_raiz, "regression", "joblib")
    RUTA_CLF = os.path.join(ruta_raiz, "classification", "joblib")

    motor = MotorInferenciaNodeQuant(RUTA_REG, RUTA_CLF)

    # Rutas de prueba
    img_test = os.path.join(ruta_raiz, "db", "casos_prueba", "case_0165", "image.nii.gz")
    mask_test = os.path.join(ruta_raiz, "db", "casos_prueba", "case_0165", "mask.nii.gz")

    try:
        print("\nAnalizando nuevo paciente...")
        reporte = motor.predecir_paciente(img_test, mask_test)

        print("\n--- REPORTE CLÍNICO NODEQUANT ---")
        print(f"Volumen Estimado:   {reporte['volumen']['valor']} {reporte['volumen']['unidad']}")
        print(f"Eje Corto Estimado: {reporte['eje_corto']['valor']} {reporte['eje_corto']['unidad']}")
        print(f"Eje Largo Estimado: {reporte['eje_largo']['valor']} {reporte['eje_largo']['unidad']}")
        print(f"Nivel de Riesgo:    {reporte['riesgo']}")
        print("---------------------------------")

    except Exception as e:
        print(f"\n[ERROR]: {e}")