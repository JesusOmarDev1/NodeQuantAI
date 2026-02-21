<!-- Badges -->

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![GitHub Repo Size](https://img.shields.io/github/repo-size/JesusOmarDev1/Lymph-Node)
![GitHub Stars](https://img.shields.io/github/stars/JesusOmarDev1/Lymph-Node)

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

Este proyecto requiere **Python 3.11** y un entorno virtual. Revisar archivo pipeline.
