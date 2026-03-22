import os
import json
import joblib
import numpy as np
import pandas as pd
import subprocess
import sys

# --- AJUSTE DE RUTAS ---
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ruta_raiz not in sys.path:
    sys.path.insert(0, ruta_raiz)

# Importación absoluta desde la raíz del proyecto
from regression.scripts.entrenar_modelo import crear_features_derivadas


class MotorInferenciaNodeQuant:
    def __init__(self, ruta_reg, ruta_clf):
        """
        Carga los modelos de ambas carpetas de forma independiente.
        """
        self.ruta_reg = ruta_reg
        self.ruta_clf = ruta_clf
        self.modelos_regresion = {}
        self.modelo_clasificacion = None

        print("Cargando cerebro de NodeQuant AI...")

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

    def extraer_radiomica(self, ruta_imagen, ruta_mascara):
        """
        Llama al entorno de Python 3.9 mediante un subproceso.
        """
        python_39_exe = r"C:/Users/korev/anaconda3/envs/sic_9/python.exe"
        # Usamos ruta relativa al dashboard para el script extractor
        script_extractor = os.path.join(os.path.dirname(__file__), "extractor_py39.py")

        proceso = subprocess.run(
            [python_39_exe, script_extractor, ruta_imagen, ruta_mascara],
            capture_output=True,
            text=True
        )

        if proceso.returncode != 0:
            raise Exception(f"Fallo en el puente 3.9.\n[STDERR]: {proceso.stderr}\n[STDOUT]: {proceso.stdout}")

        try:
            resultado_json = json.loads(proceso.stdout)
        except json.JSONDecodeError:
            raise Exception(f"Salida no válida de PyRadiomics: {proceso.stdout}")

        if "error" in resultado_json:
            raise Exception(f"Error interno en PyRadiomics (3.9): {resultado_json['error']}")

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

            mapa_riesgo = {0: "Sin riesgo", 1: "Bajo riesgo", 2: "Notorio", 3: "Crítico"}
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
    img_test = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\NodeQuantAI\Local\Dataset_Preprocesado\case_0093\image.nii.gz"
    mask_test = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\NodeQuantAI\Local\Dataset_Preprocesado\case_0093\mask.nii.gz"

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