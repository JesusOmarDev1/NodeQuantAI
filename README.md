<!-- Badges -->

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![GitHub Repo Size](https://img.shields.io/github/repo-size/JesusOmarDev1/Lymph-Node)
![GitHub Stars](https://img.shields.io/github/stars/JesusOmarDev1/Lymph-Node)
![GitHub Issues](https://img.shields.io/github/issues/JesusOmarDev1/Lymph-Node)

# Lymph-Node

Sistema de segmentación volumétrica de ganglios linfáticos mediastínicos en tomografías computarizadas mediante una arquitectura **3D Attention U‑Net**, con análisis radiómico, clasificación automática y dashboard interactivo.

---

## 📌 Descripción

Este proyecto utiliza inteligencia artificial para segmentar automáticamente ganglios linfáticos mediastínicos en tomografías computarizadas (TC). Emplea una red 3D Attention U‑Net para procesar volúmenes de TC completos y generar máscaras de segmentación. A partir de estas segmentaciones se calcula el volumen de los ganglios, se extraen características radiómicas, se realiza clasificación (Normal / Sospechoso) y se visualiza todo en un dashboard interactivo.

---

## 🧠 Características principales

- Segmentación 3D automática de ganglios linfáticos mediastínicos.
- Arquitectura **Attention U‑Net 3D** para detección robusta en volúmenes heterogéneos.
- Cálculo automático de volumen ganglionar en mm³.
- Extracción de características radiómicas para análisis secundario.
- Clasificación binaria (Normal / Sospechoso) mediante modelos de ML.
- Dashboard interactivo para visualización de resultados y métricas.

---

## 📁 Estructura del repositorio

- `descargar_data_completa.py` — Organiza o descarga el dataset.
- `convertir_dcm_niigz.py` — Convierte DICOM a volumétrico.
- `alinear_mascaras.py` — Ajusta y prepara máscaras 3D.
- `pipeline/` — Flujo de procesamiento, entrenamiento e inferencia.
- Referencias de dataset de ganglios para entrenamiento y evaluación.

---

## 🚀 Requisitos e Instalación

### Requisitos Previos

- **Python 3.11** (requerido)

### Dependencias Principales

```bash
# === Core: Procesamiento de Imágenes Médicas ===
pip install tcia-utils       # Descarga de datos desde TCIA
pip install SimpleITK        # Procesamiento de imágenes médicas 3D

# === Deep Learning: Segmentación 3D ===
pip install tensorflow       # Framework principal de DL

# === Radiomics: Extracción de Características ===
pip install scikit-image     # Procesamiento adicional de imágenes

# === Machine Learning: Clasificación ===
pip install scikit-learn     # Modelos ML clásicos
pip install xgboost          # Gradient boosting para clasificación
pip install imbalanced-learn # Manejo de datasets desbalanceados

# === Data Science: Análisis y Manipulación ===
pip install pandas           # Manipulación de datos tabulares
pip install numpy            # Operaciones numéricas fundamentales
pip install scipy            # Funciones científicas avanzadas

# === Visualización: Gráficos y Dashboard ===
pip install matplotlib       # Visualización básica
pip install seaborn          # Visualización estadística
pip install plotly           # Gráficos interactivos 3D

# === Utilidades: Métricas y Evaluación ===
pip install opencv-python    # Procesamiento de imagen adicional

# === Utilidades: Radiomicas ===
pip install pyradiomics       # Extracción de características radiómicas

# === Utilidades: Progreso y Manejo de Archivos ===
pip install tqdm             # Barras de progreso para procesamiento
```

### Instalación Rápida

Para instalar todas las dependencias de una vez, puedes usar el siguiente comando:

```bash
pip install -r requirements.txt
```

### Entorno Virtual

Si utilizas un administrador de paquetes de python puedes usar el comando "py"

```bash
py -3.11 -m venv .venv # Configura un entorno virtual con Python 3.11
.\.venv\Scripts\Activate.ps1 # Activa el entorno virtual (PowerShell)
```

## 📚 Documentación Completa

### 📖 Recursos de Documentación

- **[DeepWiki - Documentación Técnica Completa](https://deepwiki.com/JesusOmarDev1/Lymph-Node)**  
  Documentación exhaustiva con arquitectura del sistema, guías de instalación, troubleshooting y temas avanzados.

- **[Google Docs - Documentación Interna](https://docs.google.com/document/d/1TyaG8ckQWVUdvuv7YyMhKxc6z-EF1spJzrGd1Tuq2Ms/edit?tab=t.0)**  
  _(Acceso restringido a colaboradores autorizados)_

### 📑 Temas Disponibles en DeepWiki

- **Overview**: Arquitectura del sistema y conceptos clave
- **Getting Started**: Prerequisitos, instalación y ejecución
- **Pipeline Stages**: Detalles de cada etapa del procesamiento
- **Understanding the Data**: Formatos DICOM/NIfTI y estructura de datos
- **Advanced Topics**: Manejo de errores, resampling espacial, metadatos DICOM
- **Troubleshooting**: Solución de problemas comunes
- **Reference**: Definición completa del pipeline y dependencias

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 📬 Contacto y Soporte

Para preguntas, problemas o sugerencias:

- Abre un [Issue](https://github.com/JesusOmarDev1/Lymph-Node/issues) en GitHub
- Consulta la [documentación en DeepWiki](https://deepwiki.com/JesusOmarDev1/Lymph-Node)
- Revisa la sección de [Troubleshooting](https://deepwiki.com/JesusOmarDev1/Lymph-Node#7) en la wiki

---
