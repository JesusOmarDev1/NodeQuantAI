import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pycaret.regression import setup, compare_models, pull

# ===========================================================================
# 1. RUTAS Y CONFIGURACIÓN
# ===========================================================================
# Navega dos carpetas hacia atrás para encontrar la base de datos
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUTA_CSV = os.path.join(base_dir, "db", "ganglios_master.csv")
COLS_CLINICAS = ["Body Part Examined", "PatientSex", "PrimaryCondition"]


# ===========================================================================
# 2. FUNCIONES DE LIMPIEZA CLONADAS (Monolítico)
# ===========================================================================
def crear_features_derivadas(X):
    Xd = X.copy()
    _eps = 1e-9

    # Ratios
    if "firstorder_Energy" in X.columns and "firstorder_Entropy" in X.columns:
        Xd["ratio_Energy_Entropy"] = X["firstorder_Energy"] / (X["firstorder_Entropy"].abs() + _eps)
    if "firstorder_Mean" in X.columns and "firstorder_Variance" in X.columns:
        Xd["ratio_Mean_Variance"] = X["firstorder_Mean"] / (X["firstorder_Variance"].abs() + _eps)
    if "firstorder_90Percentile" in X.columns and "firstorder_10Percentile" in X.columns:
        Xd["ratio_90p_10p"] = X["firstorder_90Percentile"] / (X["firstorder_10Percentile"].abs() + _eps)
    if "firstorder_Median" in X.columns and "firstorder_Mean" in X.columns:
        Xd["ratio_Median_Mean"] = X["firstorder_Median"] / (X["firstorder_Mean"].abs() + _eps)
    if "glcm_Correlation" in X.columns and "glcm_Contrast" in X.columns:
        Xd["ratio_Corr_Contrast"] = X["glcm_Correlation"] / (X["glcm_Contrast"].abs() + _eps)
    if "glcm_Homogeneity1" in X.columns and "glcm_Contrast" in X.columns:
        Xd["ratio_Homog_Contrast"] = X["glcm_Homogeneity1"] / (X["glcm_Contrast"].abs() + _eps)
    if "glcm_JointEnergy" in X.columns and "glcm_JointEntropy" in X.columns:
        Xd["ratio_JE_JEntropy"] = X["glcm_JointEnergy"] / (X["glcm_JointEntropy"].abs() + _eps)
    if "glcm_SumAverage" in X.columns and "glcm_SumEntropy" in X.columns:
        Xd["ratio_SumAvg_SumEnt"] = X["glcm_SumAverage"] / (X["glcm_SumEntropy"].abs() + _eps)

    # Dispersión
    if "firstorder_Variance" in X.columns and "firstorder_Mean" in X.columns:
        Xd["cv_intensidad"] = np.sqrt(X["firstorder_Variance"].abs()) / (X["firstorder_Mean"].abs() + _eps)
    if "firstorder_90Percentile" in X.columns and "firstorder_10Percentile" in X.columns:
        Xd["rango_interpercentil"] = X["firstorder_90Percentile"] - X["firstorder_10Percentile"]

    # Transformaciones Log y Cuadradas
    if "firstorder_Energy" in X.columns:
        Xd["log_Energy"] = np.log1p(X["firstorder_Energy"].abs())
        Xd["sq_Energy"] = X["firstorder_Energy"] ** 2
    if "firstorder_TotalEnergy" in X.columns:
        Xd["log_TotalEnergy"] = np.log1p(X["firstorder_TotalEnergy"].abs())
        Xd["cbrt_TotalEnergy"] = np.cbrt(X["firstorder_TotalEnergy"])
    if "firstorder_Variance" in X.columns:
        Xd["log_Variance"] = np.log1p(X["firstorder_Variance"].abs())
    if "firstorder_Range" in X.columns:
        Xd["log_Range"] = np.log1p(X["firstorder_Range"].abs())
    if "firstorder_RootMeanSquared" in X.columns:
        Xd["sq_RMS"] = X["firstorder_RootMeanSquared"] ** 2
    if "glcm_Autocorrelation" in X.columns:
        Xd["sq_Autocorrelation"] = X["glcm_Autocorrelation"] ** 2
    if "glcm_SumAverage" in X.columns:
        Xd["sq_SumAverage"] = X["glcm_SumAverage"] ** 2
    if "firstorder_Energy" in X.columns:
        Xd["cbrt_Energy"] = np.cbrt(X["firstorder_Energy"])

    # Diferencias e Interacciones
    if "firstorder_Median" in X.columns and "firstorder_10Percentile" in X.columns:
        Xd["diff_Median_10p"] = X["firstorder_Median"] - X["firstorder_10Percentile"]
    if "firstorder_Maximum" in X.columns and "firstorder_Minimum" in X.columns:
        Xd["diff_Max_Min"] = X["firstorder_Maximum"] - X["firstorder_Minimum"]
    if "firstorder_Energy" in X.columns and "glcm_JointEnergy" in X.columns:
        Xd["inter_Energy_JointEnergy"] = X["firstorder_Energy"] * X["glcm_JointEnergy"]
    if "firstorder_TotalEnergy" in X.columns and "glcm_Autocorrelation" in X.columns:
        Xd["inter_TotalEnergy_Autocorr"] = X["firstorder_TotalEnergy"] * X["glcm_Autocorrelation"]
    if "firstorder_RootMeanSquared" in X.columns and "glcm_SumAverage" in X.columns:
        Xd["inter_RMS_SumAvg"] = X["firstorder_RootMeanSquared"] * X["glcm_SumAverage"]
    if "firstorder_Entropy" in X.columns and "glcm_JointEntropy" in X.columns:
        Xd["inter_Entropy_JEntropy"] = X["firstorder_Entropy"] * X["glcm_JointEntropy"]

    # Flag estático de necrosis
    if "firstorder_Minimum" in X.columns and "glszm_ZoneVariance" in X.columns:
        Xd["flag_conglomerado_necrotico"] = np.where(
            (X["firstorder_Minimum"] <= -30.0) & (X["glszm_ZoneVariance"] >= 15000.0), 1, 0
        )

    Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Xd


def preparar_datos_para_pycaret(df_raw):
    """Aplica el Escudo Anti-Fuga de Datos y limpieza."""
    df_raw = df_raw.drop_duplicates(subset="Paciente_ID").reset_index(drop=True)

    # Recorte del 98% (igual que en tu entrenamiento original)
    limite_superior = df_raw["shape_VoxelVolume"].quantile(0.98)
    df_raw = df_raw[df_raw["shape_VoxelVolume"] <= limite_superior].copy()

    shape_cols = [c for c in df_raw.columns if
                  c.startswith("shape_") and c not in ["shape_Sphericity", "shape_Elongation", "shape_Flatness",
                                                       "shape_VoxelVolume"]]

    y_temp_vol = df_raw["shape_VoxelVolume"].values
    fuga_cols = []
    cols_a_evaluar = [c for c in df_raw.columns if
                      c not in ["target_regresion", "target_eje_corto", "target_eje_largo", "Paciente_ID",
                                "target_riesgo", "shape_VoxelVolume"]
                      + COLS_CLINICAS + shape_cols]

    # Escudo Spearman > 0.75
    for col in cols_a_evaluar:
        coef, _ = spearmanr(df_raw[col], y_temp_vol)
        if abs(coef) > 0.75:
            fuga_cols.append(col)

    cols_drop = ["Paciente_ID", "target_riesgo", "target_regresion", "target_eje_corto",
                 "target_eje_largo"] + COLS_CLINICAS + shape_cols + fuga_cols

    df_limpio = df_raw.drop(columns=[c for c in cols_drop if c in df_raw.columns])
    return df_limpio


# ===========================================================================
# 3. EJECUCIÓN DEL TORNEO PYCARET
# ===========================================================================
if __name__ == "__main__":
    print("1. Cargando y limpiando datos (Modo Monolítico)...")
    df_crudo = pd.read_csv(RUTA_CSV)
    df_limpio = preparar_datos_para_pycaret(df_crudo)
    df_final = crear_features_derivadas(df_limpio)

    # Cambiamos la escala de mm³ a cm³ (Mililitros) para que no truene al transformar
    #df_final['shape_VoxelVolume'] = df_final['shape_VoxelVolume'] / 1000.0

    # Aplicando log1p nosotros mismos
    df_final['shape_VoxelVolume'] = np.log1p(df_final['shape_VoxelVolume'])

    print(f"Dataset listo: {df_final.shape[0]} pacientes, {df_final.shape[1]} columnas.")
    print("2. Iniciando entorno PyCaret (Escala en cm³)...")

    # Configuramos el entorno de PyCaret
    setup(
        data=df_final,
        target='shape_VoxelVolume',
        normalize=True,
        normalize_method='minmax',  # Experimentando con MinMaxScaler
        transform_target=False,  # Simula np.log1p para estabilizar los tumores gigantes Nota: falló por los outliers, apagamos para usarlos en crudo
        fold=5,
        session_id=42,
        verbose=False
    )

    print("\n3. Entrenando y comparando TODOS los algoritmos baselines...")
    mejores_modelos = compare_models(sort='MAE', n_select=5)

    print("\n" + "=" * 60)
    print("TABLA DE POSICIONES PYCARET (Baselines)")
    print("=" * 60)

    # Extraemos la tabla de resultados para mostrarla en consola
    tabla_resultados = pull()
    print(tabla_resultados[['MAE', 'RMSE', 'R2', 'MAPE']].head(10))

    print("\n[!] Experimento finalizado")