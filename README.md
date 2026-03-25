<!-- Badges -->

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![Django 5.2](https://img.shields.io/badge/Django-5.2-green.svg)
![GitHub Repo Size](https://img.shields.io/github/repo-size/JesusOmarDev1/Lymph-Node)
![GitHub Stars](https://img.shields.io/github/stars/JesusOmarDev1/Lymph-Node)
![GitHub Issues](https://img.shields.io/github/issues/JesusOmarDev1/Lymph-Node)

# NodeQuant AI

Plataforma de soporte a decisiones clínicas en oncología para cuantificación de ganglios linfáticos mediastínicos.

Basado en el dataset **Mediastinal-Lymph-Node-SEG** del Cancer Imaging Archive:

> Idris, T., Somarouthu, S., Jacene, H., LaCasce, A., Ziegler, E., Pieper, S., Khajavi, R., Dorent, R., Pujol, S., Kikinis, R., & Harris, G. (2024). Mediastinal Lymph Node Quantification (LNQ): Segmentation of Heterogeneous CT Data (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/QVAZ-JA09

---

## Descripción

Plataforma analítica inteligente que cuantifica biomarcadores de tomografías computarizadas de ganglios linfáticos mediante radiómica avanzada y predice el riesgo de adenopatías mediastínicas a través de modelos de clasificación y regresión, integrados en un dashboard interactivo Django.

---

## Características principales

- **Dashboard interactivo** Django con interfaz web para carga de estudios NIfTI y generación de reportes clínicos automáticos.
- **Extracción radiómica** automática de 57+ features por ganglio usando PyRadiomics (firstorder, GLCM, GLRLM, GLSZM, GLDM).
- **Regresión volumétrica** con StackingRegressor (XGBoost + RandomForest + GradientBoosting) para predicción de volumen (mm³), eje corto y eje largo.
- **Clasificación de riesgo** en 4 niveles: Sin riesgo, Bajo riesgo, Notorio, Crítico.
- **Segmentación 3D** con Attention U-Net 3D para detección de ganglios en volúmenes heterogéneos (opcional).
- **Sistema de fallback multi-entorno** que soporta equipos con venv o Anaconda sin conflictos.

---

## Estructura del repositorio

```
NodeQuantAI/
├── preprocess/                 # Pipeline de preprocesamiento (DICOM → NIfTI → Radiomics)
│   ├── descargar_data_completa.py
│   ├── convertir_dcm_niigz.py
│   ├── resampling_isotropico.py
│   ├── verificar_resampling.py
│   ├── alinear_mascaras.py
│   ├── extraccion_radiomica.py
│   └── preparar_dataset.py
├── cnn/                        # Segmentación 3D (Attention U-Net)
│   ├── modelo_unet3d.py
│   ├── dataset_medico.py
│   └── training.py
├── classification/             # Clasificación de riesgo ganglionar
│   ├── scripts/
│   └── joblib/                 # Modelo serializado (.joblib)
├── regression/                 # Regresión volumétrica
│   ├── scripts/
│   ├── joblib/                 # Modelos serializados (.joblib)
│   └── metrics/                # Métricas y gráficas de evaluación
├── dashboard/                  # Dashboard interactivo Django
│   ├── run.py                  # Punto de entrada (auto-detecta intérpretes)
│   ├── backend/
│   │   ├── motor_inferencia.py # Orquestador del pipeline de inferencia
│   │   └── extractor_py39.py   # Extractor PyRadiomics (subprocess)
│   └── frontend/               # App Django (prediccion)
│       ├── manage.py
│       ├── core/               # Settings Django
│       └── prediccion/         # Vistas, templates, static
├── db/                         # CSVs con features y targets
│   ├── ganglios_radiomica.csv
│   ├── ganglios_master.csv
│   └── casos_prueba.csv
├── Dataset_NIFIT/              # Volúmenes NIfTI originales (no versionado)
├── Dataset_Preprocesado/       # Volúmenes preprocesados (no versionado)
└── Mediastinal_Data/           # Datos DICOM originales (no versionado)
```

---

## Arquitectura dual-venv

El proyecto usa dos entornos virtuales por restricciones de compatibilidad:

| Entorno    | Python | Propósito            | Paquetes clave                            |
| ---------- | ------ | -------------------- | ----------------------------------------- |
| **venv9**  | 3.9    | Extracción radiómica | PyRadiomics, SimpleITK                    |
| **venv11** | 3.11   | Dashboard y ML       | Django 5.2, scikit-learn, XGBoost, joblib |

El sistema de **fallback automático** resuelve qué intérprete usar en cada momento:

- `run.py` auto-detecta `venv11` para Django aunque se lance desde `venv9`.
- `motor_inferencia.py` auto-detecta `venv9` para PyRadiomics aunque Django corra en `venv11`.
- Cada candidato se valida con un import real antes de usarse.

---

## Requisitos e Instalación

### Requisitos previos

- **Python 3.9** (para PyRadiomics)
- **Python 3.11** (para Django y ML)
- **Git**

### 1. Clonar el repositorio

```bash
git clone https://github.com/JesusOmarDev1/Lymph-Node.git
cd Lymph-Node
```

### 2. Crear entornos virtuales

```powershell
# Entorno para radiomics (Python 3.9)
py -3.9 -m venv venv9

# Entorno para Django y ML (Python 3.11)
py -3.11 -m venv venv11
```

### 3. Instalar dependencias

```powershell
# --- venv9: Radiomics ---
& .\venv9\Scripts\Activate.ps1
pip install pyradiomics SimpleITK

# --- venv11: Django + ML ---
& .\venv11\Scripts\Activate.ps1
pip install django scikit-learn xgboost joblib pandas numpy scipy
pip install matplotlib seaborn plotly tqdm
```

### 4. Aplicar migraciones de Django

```powershell
& .\venv11\Scripts\python.exe dashboard\frontend\manage.py migrate
```

---

## Variables de entorno (opcional)

Para equipos con Anaconda o rutas personalizadas, se pueden configurar overrides:

| Variable                     | Descripción                          | Ejemplo                                         |
| ---------------------------- | ------------------------------------ | ----------------------------------------------- |
| `NODEQUANT_PYTHON_RADIOMICS` | Ruta al `python.exe` con PyRadiomics | `C:\Users\user\anaconda3\envs\py39\python.exe`  |
| `NODEQUANT_PYTHON_DJANGO`    | Ruta al `python.exe` con Django      | `C:\Users\user\anaconda3\envs\py311\python.exe` |

Si no se configuran, el sistema busca automáticamente en `venv9`/`venv11`, Conda activo y rutas comunes.

---

## Comandos útiles

### Activar entornos

```powershell
# Radiomics (Python 3.9)
& .\venv9\Scripts\Activate.ps1

# Django + ML (Python 3.11)
& .\venv11\Scripts\Activate.ps1
```

### Dashboard

```powershell
# Iniciar el dashboard (se puede lanzar desde cualquier entorno)
python dashboard\run.py

# El dashboard estará disponible en http://127.0.0.1:8000/
```

### Migraciones Django

```powershell
& .\venv11\Scripts\python.exe dashboard\frontend\manage.py migrate
```

### Pipeline de preprocesamiento (ejecutar en orden)

```powershell
& .\venv9\Scripts\Activate.ps1

python preprocess\convertir_dcm_niigz.py
python preprocess\resampling_isotropico.py
python preprocess\verificar_resampling.py
python preprocess\alinear_mascaras.py
python preprocess\extraccion_radiomica.py
python preprocess\preparar_dataset.py
```

### Entrenamiento de modelos

```powershell
& .\venv11\Scripts\Activate.ps1

# Regresión
python regression\scripts\entrenar_modelo.py
python regression\scripts\evaluar_modelos.py

# Clasificación
python classification\scripts\clasificar_modelo.py

# Segmentación 3D (opcional, requiere GPU)
python cnn\training.py
```

### Prueba directa del motor de inferencia

```powershell
& .\venv11\Scripts\Activate.ps1
python dashboard\backend\motor_inferencia.py
```

---

## Pipeline del proyecto

```
1. preprocess/descargar_data_completa.py     → Descarga DICOM desde TCIA
2. preprocess/convertir_dcm_niigz.py         → DICOM → NIfTI (.nii.gz)
3. preprocess/resampling_isotropico.py        → Resampling isotrópico a 1×1×1 mm
4. preprocess/verificar_resampling.py         → Validación del resampling
5. preprocess/alinear_mascaras.py             → Alineación espacial de máscaras
6. preprocess/extraccion_radiomica.py         → Extracción de 57 features → CSV
7. preprocess/preparar_dataset.py             → Preparación de datasets ML
8. regression/scripts/entrenar_modelo.py      → Entrenamiento de modelos de regresión
9. regression/scripts/evaluar_modelos.py      → Evaluación y comparación de modelos
10. cnn/training.py                           → Entrenamiento de segmentación 3D (opcional)
```

---

## Documentación

- **[DeepWiki - Documentación Técnica](https://deepwiki.com/JesusOmarDev1/Lymph-Node)** — Arquitectura, guías de instalación, troubleshooting y temas avanzados.
- **[Google Docs - Documentación Interna](https://docs.google.com/document/d/1TyaG8ckQWVUdvuv7YyMhKxc6z-EF1spJzrGd1Tuq2Ms/edit?tab=t.0)** — _(Acceso restringido a colaboradores autorizados)_

---

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## Contacto y Soporte

- Abre un [Issue](https://github.com/JesusOmarDev1/Lymph-Node/issues) en GitHub
- Consulta la [documentación en DeepWiki](https://deepwiki.com/JesusOmarDev1/Lymph-Node)

---
