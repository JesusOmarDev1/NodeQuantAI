import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import subprocess
import tempfile
import SimpleITK as sitk
from typing import List, Tuple

ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ruta_raiz not in sys.path:
    sys.path.insert(0, ruta_raiz)

from regression.scripts.modelo_regresion import crear_features_derivadas


class MotorInferenciaNodeQuant:
    def __init__(self, ruta_reg, ruta_clf):
        # carga los modelos de ambas carpetas
        self.ruta_reg = ruta_reg
        self.ruta_clf = ruta_clf
        self.modelos_regresion = {}
        self.modelo_clasificacion = None
        self.python_radiomics_exe = self._resolver_python_radiomics()

        print("Cargando modelos de NodeQuant AI...")
        print(f"  [OK] Intérprete radiomics seleccionado: {self.python_radiomics_exe}")

        # cargar el modelo de regresión
        targets = ["volumen"]
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
                print(f"  [OK] Modelo regresión '{t}' cargado.")
            else:
                print(f"  [ALERTA] Faltan archivos para '{t}' en {self.ruta_reg}")

        # cargar modelo de clasificación
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
        #verificar que el interprete pueda importar radiomics
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
        #resuelve el intérprete
        candidatos: List[str] = []

        # override explícito por variable de entorno
        exe_env = os.environ.get("NODEQUANT_PYTHON_RADIOMICS", "").strip()
        if exe_env:
            candidatos.append(exe_env)

        # buscando al interprete
        candidatos.append(os.path.join(ruta_raiz, "sic_9", "Scripts", "python.exe"))

        conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
        if conda_prefix:
            candidatos.append(os.path.join(conda_prefix, "python.exe"))

        user_home = os.path.expanduser("~")
        bases_conda = ["anaconda3", "miniconda3"]
        envs_conda = ["sic_9", "py39", "venv9", "nodequant", "radiomics", "base"]
        for base in bases_conda:
            candidatos.append(os.path.join(user_home, base, "python.exe"))
            for env_name in envs_conda:
                candidatos.append(os.path.join(user_home, base, "envs", env_name, "python.exe"))

        # interprete actual como último recurso
        candidatos.append(sys.executable)

        # mantener orden y eliminar duplicados
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
            "No se encontró un intérprete válido para PyRadiomics"
            + "\n".join(errores)
        )

    def generar_visualizacion(self, ruta_imagen, ruta_mascara):
        # genera imágenes CT (original + overlay) con un subproceso con Python 3.9
        # retorna dict con imágenes base64 y metadata, o None si falla
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
        # llama al extractor de PyRadiomics con un subproceso
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

    def _resamplear_fantasma(self, ruta_in, ruta_out, es_mascara=False):
        #re-muestreo a 1x1x1 mm para la matemática interna del modelo
        imagen = sitk.ReadImage(ruta_in)
        espaciado_original = imagen.GetSpacing()
        tamano_original = imagen.GetSize()
        espaciado_objetivo = (1.0, 1.0, 1.0)

        nuevo_tamano = [
            int(np.round(tamano_original[0] * (espaciado_original[0] / espaciado_objetivo[0]))),
            int(np.round(tamano_original[1] * (espaciado_original[1] / espaciado_objetivo[1]))),
            int(np.round(tamano_original[2] * (espaciado_original[2] / espaciado_objetivo[2])))
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(nuevo_tamano)
        resampler.SetOutputSpacing(espaciado_objetivo)
        resampler.SetOutputOrigin(imagen.GetOrigin())
        resampler.SetOutputDirection(imagen.GetDirection())

        if es_mascara:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(-1000)  # HU del aire

        imagen_resampleada = resampler.Execute(imagen)
        sitk.WriteImage(imagen_resampleada, ruta_out)

    def predecir_paciente(self, ruta_imagen, ruta_mascara):
        #ejecuta todo el pipeline usando copias temporales resampleadas a 1x1x1mm

        # crear entorno temporal
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta_img_res = os.path.join(tmpdir, "img_1x1x1.nii.gz")
            ruta_mask_res = os.path.join(tmpdir, "mask_1x1x1.nii.gz")

            # resampling fantasma
            self._resamplear_fantasma(ruta_imagen, ruta_img_res, es_mascara=False)
            self._resamplear_fantasma(ruta_mascara, ruta_mask_res, es_mascara=True)

            # extracción de características
            df_crudo = self.extraer_radiomica(ruta_img_res, ruta_mask_res)

        # derivadas
        df_procesado = crear_features_derivadas(df_crudo)

        resultados_finales = {}

        # predicciones de volumen
        for target, info in self.modelos_regresion.items():
            modelo = info["modelo"]
            meta = info["metadata"]

            X_input = df_procesado.reindex(columns=meta["features_entrada"], fill_value=0)

            pred_raw = modelo.predict(X_input)[0]
            pred_final = np.maximum(0, pred_raw)

            resultados_finales[target] = {
                "valor": round(pred_final, 2),
                "unidad": meta["unidad"]
            }

        # predicción de riesgo
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
    RUTA_REG = os.path.join(ruta_raiz, "regression", "joblib")
    RUTA_CLF = os.path.join(ruta_raiz, "classification", "joblib")

    motor = MotorInferenciaNodeQuant(RUTA_REG, RUTA_CLF)

    # rutas de prueba
    img_test = os.path.join(ruta_raiz, "db", "casos_prueba", "case_0165", "image.nii.gz")
    mask_test = os.path.join(ruta_raiz, "db", "casos_prueba", "case_0165", "mask.nii.gz")

    try:
        print("\nAnalizando nuevo paciente...")
        reporte = motor.predecir_paciente(img_test, mask_test)

        print("\n--- REPORTE CLÍNICO NODEQUANT ---")
        print(f"Volumen Estimado:   {reporte['volumen']['valor']:,.1f} {reporte['volumen']['unidad']}")
        print(f"Nivel de Riesgo:    {reporte['riesgo']}")
        print("---------------------------------")

    except Exception as e:
        print(f"\n[ERROR]: {e}")