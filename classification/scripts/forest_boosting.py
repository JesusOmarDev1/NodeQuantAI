# 0 - Importar librerías necesarias
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
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

    objetivo_col = "target_riesgo"
    columna_id = "Paciente_ID"

    if objetivo_col not in df.columns:
        print(f"Error: La columna '{objetivo_col}' no existe en el dataset.")
        return None

    # === NUEVO: Fusión y Mapeo de Clases ===
    # Convertimos (0 y 1 -> 0: Bajo), (2 -> 1: Moderado), (3 -> 2: Medio-alto)
    mapeo_clases = {0: 0, 1: 0, 2: 1, 3: 2}
    y = df[objetivo_col].map(mapeo_clases)

    nombres_clases = ["Bajo", "Moderado", "Medio-alto"]

    x_raw = df.drop(columns=[objetivo_col, columna_id], errors='ignore')
    x = pd.get_dummies(x_raw)

    print(f"Características procesadas: {x.shape[1]} | Objetivo: {objetivo_col} (3 Clases)")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    f1_scores = {}
    predicciones = {}
    modelos_entrenados = {}

    # === Entrenar Modelos Base ===
    print("\nEntrenando modelos individuales basados en árboles...")

    modelo_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    modelo_rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    modelo_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, random_state=42, eval_metric='mlogloss')
    modelo_gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)

    modelos_individuales = {
        "Árbol de Decisión": modelo_dt,
        "Random Forest": modelo_rf,
        "XGBoost": modelo_xgb,
        "Gradient Boosting": modelo_gb
    }

    for nombre, modelo in modelos_individuales.items():
        modelo.fit(x_train_scaled, y_train)
        pred = modelo.predict(x_test_scaled)
        f1_scores[nombre] = f1_score(y_test, pred, average='weighted')
        predicciones[nombre] = pred
        modelos_entrenados[f"modelo_{nombre.replace(' ', '_').lower()}"] = modelo
        print(f" - {nombre} listo.")

    # === Ensamble por Votación (Voting Classifier) ===
    print("\nConstruyendo Súper Ensamble: Voting Classifier...")
    modelo_voting = VotingClassifier(
        estimators=[('rf', modelo_rf), ('xgb', modelo_xgb), ('gb', modelo_gb)],
        voting='soft'
    )
    modelo_voting.fit(x_train_scaled, y_train)
    pred_voting = modelo_voting.predict(x_test_scaled)
    f1_scores["Ensamble (Votación)"] = f1_score(y_test, pred_voting, average='weighted')
    predicciones["Ensamble (Votación)"] = pred_voting
    modelos_entrenados["modelo_ensamble_voting"] = modelo_voting

    # === Ensamble Apilado (Stacking Classifier) ===
    print("Construyendo Súper Ensamble: Stacking Classifier...")
    modelo_stacking = StackingClassifier(
        estimators=[('rf', modelo_rf), ('xgb', modelo_xgb), ('gb', modelo_gb)],
        final_estimator=DecisionTreeClassifier(max_depth=3, random_state=42)
    )
    modelo_stacking.fit(x_train_scaled, y_train)
    pred_stacking = modelo_stacking.predict(x_test_scaled)
    f1_scores["Ensamble (Apilado)"] = f1_score(y_test, pred_stacking, average='weighted')
    predicciones["Ensamble (Apilado)"] = pred_stacking
    modelos_entrenados["modelo_ensamble_stacking"] = modelo_stacking

    # === Comparación Automática de Modelos ===
    print("\n" + "=" * 50)
    print("COMPARACIÓN FINAL DE MODELOS (Por F1-Score):")
    modelos_ordenados = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)

    for nombre, puntaje in modelos_ordenados:
        if "Ensamble" in nombre:
            print(f"{nombre:<23}: {puntaje:.4f}")
        else:
            print(f"  - {nombre:<24}: {puntaje:.4f}")

    mejor_modelo_nombre = modelos_ordenados[0][0]
    print(f"\n EL MEJOR MODELO ES: {mejor_modelo_nombre.upper()} ")
    print("=" * 50)

    mejor_prediccion = predicciones[mejor_modelo_nombre]

    print(f"\nReporte detallado del modelo ganador ({mejor_modelo_nombre}):")
    print(classification_report(y_test, mejor_prediccion, target_names=nombres_clases))

    # Generar Matriz de Confusión
    os.makedirs(carpeta_modelos, exist_ok=True)
    cm = confusion_matrix(y_test, mejor_prediccion)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nombres_clases)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title(f"Matriz de Confusión - {mejor_modelo_nombre}", fontsize=14, pad=20)
    plt.tight_layout()
    ruta_grafico = os.path.join(carpeta_modelos, "matriz_confusion.png")
    plt.savefig(ruta_grafico)
    plt.close(fig)

    print(f"\nGuardando {len(modelos_entrenados)} modelos en {carpeta_modelos}...")
    for nombre_archivo, modelo_obj in modelos_entrenados.items():
        dump(modelo_obj, os.path.join(carpeta_modelos, f"{nombre_archivo}.joblib"))
    dump(scaler, os.path.join(carpeta_modelos, "scaler_clf.joblib"))
    print("Proceso terminado correctamente.\n")

    return True


if __name__ == "__main__":
    entrenar_modelos_clasificacion()