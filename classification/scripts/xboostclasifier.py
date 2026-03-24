# 0 - Importar librerías necesarias
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from joblib import dump
import warnings as w
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

w.filterwarnings("ignore")

# 1 - Configuración de la carpeta de datos y salida
csv_entrada = r"I:\GitHub\NodeQuantAI\db\ganglios_master.csv"
carpeta_modelos = r"I:\GitHub\NodeQuantAI\classification"


def entrenar_modelos_clasificacion():
    if not os.path.exists(csv_entrada):
        print(f"Error: No se encontró el CSV en {csv_entrada}")
        return None

    print("\nCargando dataset limpio...")
    df = pd.read_csv(csv_entrada)

    if df.empty:
        print("Error: El CSV está vacío.")
        return None

    objetivo_col = "target_riesgo"
    columna_id = "Paciente_ID"

    if objetivo_col not in df.columns:
        print(f"Error: La columna '{objetivo_col}' no existe en el dataset.")
        return None

    # Mapeo directo de tus clases (0, 1, 2, 3)
    nombres_clases = ["sin riesgo", "bajo riesgo", "notorio", "critico"]

    y = df[objetivo_col]
    x_raw = df.drop(columns=[objetivo_col, columna_id], errors='ignore')
    x = pd.get_dummies(x_raw)

    print(f"Características procesadas: {x.shape[1]} | Objetivo: {objetivo_col}")

    # Dividir en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)

    #Escalar los datos
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    f1_scores = {}
    predicciones = {}
    modelos_entrenados = {}

    # === 1. Regresión Logística ===
    print("\nEntrenando Regresión Logística...")
    modelo_rl = LogisticRegression(max_iter=1000, random_state=42)
    modelo_rl.fit(x_train_scaled, y_train)
    pred_rl = modelo_rl.predict(x_test_scaled)
    f1_scores["Regresión Logística"] = f1_score(y_test, pred_rl, average='weighted')
    predicciones["Regresión Logística"] = pred_rl
    modelos_entrenados["modelo_regresion_logistica"] = modelo_rl

    # === 2. Árbol de Decisión ===
    print("Entrenando Árbol de Decisión...")
    modelo_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    modelo_dt.fit(x_train_scaled, y_train)
    pred_dt = modelo_dt.predict(x_test_scaled)
    f1_scores["Árbol de Decisión"] = f1_score(y_test, pred_dt, average='weighted')
    predicciones["Árbol de Decisión"] = pred_dt
    modelos_entrenados["modelo_decision_tree"] = modelo_dt

    # === 3. Random Forest ===
    print("Entrenando Random Forest...")
    modelo_rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    modelo_rf.fit(x_train_scaled, y_train)
    pred_rf = modelo_rf.predict(x_test_scaled)
    f1_scores["Random Forest"] = f1_score(y_test, pred_rf, average='weighted')
    predicciones["Random Forest"] = pred_rf
    modelos_entrenados["modelo_random_forest"] = modelo_rf

    # === 4. XGBoost Classifier ===
    print("Entrenando XGBoost Classifier...")
    modelo_xgb = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'
    )
    modelo_xgb.fit(x_train_scaled, y_train)
    pred_xgb = modelo_xgb.predict(x_test_scaled)
    f1_scores["XGBoost"] = f1_score(y_test, pred_xgb, average='weighted')
    predicciones["XGBoost"] = pred_xgb
    modelos_entrenados["modelo_xgboost"] = modelo_xgb

    # === 5. Gradient Boosting Clásico (NUEVO) ===
    print("Entrenando Gradient Boosting Clásico...")
    modelo_gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
    )
    modelo_gb.fit(x_train_scaled, y_train)
    pred_gb = modelo_gb.predict(x_test_scaled)
    f1_scores["Gradient Boosting"] = f1_score(y_test, pred_gb, average='weighted')
    predicciones["Gradient Boosting"] = pred_gb
    modelos_entrenados["modelo_gradient_boosting"] = modelo_gb

    # === Comparación Automática de Modelos ===
    print("\n" + "=" * 50)
    print("COMPARACIÓN FINAL DE MODELOS (Por F1-Score):")
    modelos_ordenados = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)

    for nombre, puntaje in modelos_ordenados:
        print(f"  - {nombre:<25}: {puntaje:.4f}")

    mejor_modelo_nombre = modelos_ordenados[0][0]
    print(f"\n EL MEJOR MODELO ES: {mejor_modelo_nombre.upper()} ")
    print("=" * 50)

    mejor_prediccion = predicciones[mejor_modelo_nombre]

    # Reporte detallado
    print(f"\nReporte detallado del modelo ganador ({mejor_modelo_nombre}):")
    print(classification_report(y_test, mejor_prediccion, target_names=nombres_clases))

    # === Muestra de Predicciones Reales vs IA ===
    print("\n--- Muestra aleatoria de 10 predicciones vs. Realidad ---")
    resultados_df = pd.DataFrame({
        'Riesgo Real (Médico)': [nombres_clases[i] for i in y_test.values],
        'Riesgo Predicho (IA)': [nombres_clases[i] for i in mejor_prediccion]
    })
    print(resultados_df.sample(n=min(10, len(resultados_df)), random_state=42).to_string(index=False))

    # === Generar y Guardar Matriz de Confusión ===
    os.makedirs(carpeta_modelos, exist_ok=True)

    cm = confusion_matrix(y_test, mejor_prediccion)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nombres_clases)

    # Configuramos el gráfico usando una paleta de colores azules
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title(f"Matriz de Confusión - {mejor_modelo_nombre}", fontsize=14, pad=20)
    plt.tight_layout()

    ruta_grafico = os.path.join(carpeta_modelos, "matriz_confusion.png")
    plt.savefig(ruta_grafico)
    plt.close(fig)  # Cerramos la figura para liberar memoria
    print(f"\n - Gráfico de matriz de confusión guardado en:\n   {ruta_grafico}")

    # Guardar todos los modelos
    print(f"\nGuardando archivos de modelos en {carpeta_modelos}...")

    for nombre_archivo, modelo_obj in modelos_entrenados.items():
        dump(modelo_obj, os.path.join(carpeta_modelos, f"{nombre_archivo}.joblib"))

    dump(scaler, os.path.join(carpeta_modelos, "scaler_clf.joblib"))
    print(" - Modelos y StandardScaler guardados correctamente.\n")

    return True


if __name__ == "__main__":
    resultado = entrenar_modelos_clasificacion()

    if resultado:
        print("Proceso terminado exitosamente\n")
    else:
        print("Proceso terminado con errores.\n")