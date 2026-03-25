import streamlit as st
import os
import sys
import tempfile
from matplotlib.colors import ListedColormap

# ==========================================
# 1. Configuración y Rutas del Proyecto
# ==========================================
st.set_page_config(
    page_title="NodeQuant AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajusta tu ruta raíz para que Python encuentre tus módulos
ruta_raiz = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\NodeQuantAI"
if ruta_raiz not in sys.path:
    sys.path.insert(0, ruta_raiz)

# Importamos tu motor de inferencia y la función de visualización
from dashboard.motor_inferencia import MotorInferenciaNodeQuant

# (Pegar aquí la función `generar_figura_fusion` que creamos en el paso anterior,
# o impórtala si la guardaste en otro archivo)
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def generar_figura_fusion(ruta_img, ruta_mask):
    try:
        ct_itk = sitk.ReadImage(ruta_img)
        mask_itk = sitk.ReadImage(ruta_mask)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_itk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform())
        mask_alineada = resampler.Execute(mask_itk)
        img_data = sitk.GetArrayFromImage(ct_itk)
        mask_data = sitk.GetArrayFromImage(mask_alineada)

        # Buscando el centro de la máscara
        # 1. Revisa todas las rebanadas del escáner y detecta cuáles tienen al menos un píxel de la máscara
        #slices_con_ganglio = np.any(mask_data > 0, axis=(1, 2))
        # 2. Guarda los números de esos cortes
        #indices_validos = np.where(slices_con_ganglio)[0]
        # 3. Va a la lista de cortes válidos y elige el que está exactamente a la mitad
        #slice_idx = indices_validos[len(indices_validos) // 2] if len(indices_validos) > 0 else img_data.shape[0] // 2

        # Buscando el área más grande la máscara
        pixeles_por_slice = np.sum(mask_data > 0, axis=(1, 2))
        if pixeles_por_slice.max() > 0:
            slice_idx = np.argmax(pixeles_por_slice)
        else:
            slice_idx = img_data.shape[0] // 2  # Por si la máscara viene vacía

        img_slice = np.rot90(img_data[slice_idx, :, :], 2)
        mask_slice = np.rot90(mask_data[slice_idx, :, :], 2)

        valor_fondo = img_slice.min()

        # Consideramos como "cuerpo del paciente" todo lo que supere ese vacío
        mascara_cuerpo = img_slice > (valor_fondo + 20)

        filas_validas = np.any(mascara_cuerpo, axis=1)
        cols_validas = np.any(mascara_cuerpo, axis=0)

        if np.any(filas_validas) and np.any(cols_validas):
            rmin, rmax = np.where(filas_validas)[0][[0, -1]]
            cmin, cmax = np.where(cols_validas)[0][[0, -1]]

            # Damos un margen para que no quede pegado a los bordes
            margen = 20
            rmin, rmax = max(0, rmin - margen), min(img_slice.shape[0], rmax + margen)
            cmin, cmax = max(0, cmin - margen), min(img_slice.shape[1], cmax + margen)

            # Aplicamos el recorte a tomografía y máscara al mismo tiempo
            img_slice = img_slice[rmin:rmax, cmin:cmax]
            mask_slice = mask_slice[rmin:rmax, cmin:cmax]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_alpha(0.0)
        vmin, vmax = np.percentile(img_slice, [2, 98])

        axes[0].imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Tomografía (Corte {slice_idx})', color='white')
        axes[0].axis('off')

        axes[1].imshow(mask_slice, cmap='bone', vmin=0, vmax=1)
        axes[1].set_title('Máscara del Ganglio', color='white')
        axes[1].axis('off')

        axes[2].imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax)
        cmap_rojo = ListedColormap(['#ff1111'])
        mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
        axes[2].imshow(mask_overlay, cmap=cmap_rojo, alpha=0.5, interpolation='none')
        axes[2].set_title('Superposición', color='white')
        axes[2].axis('off')

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error visual: {e}")
        return None


# ==========================================
# 2. Carga del Motor en Caché (¡Muy rápido!)
# ==========================================
@st.cache_resource
def cargar_motor():
    ruta_reg = os.path.join(ruta_raiz, "regression", "joblib")
    ruta_clf = os.path.join(ruta_raiz, "classification", "joblib")
    # Inicializa y devuelve la clase de tu motor
    return MotorInferenciaNodeQuant(ruta_reg, ruta_clf)


motor = cargar_motor()


def guardar_archivo_temporal(archivo_subido):
    if archivo_subido is not None:
        directorio_temp = tempfile.mkdtemp()
        ruta_completa = os.path.join(directorio_temp, archivo_subido.name)
        with open(ruta_completa, "wb") as f:
            f.write(archivo_subido.getbuffer())
        return ruta_completa
    return None


# ==========================================
# 3. Interfaz Visual
# ==========================================
st.title("🫁 NodeQuant AI")
st.markdown("Sube la tomografía (CT) y la máscara de segmentación para iniciar el análisis radiómico.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    archivo_img = st.file_uploader("Tomografía (image.nii.gz)", type=['nii', 'gz'])
with col2:
    archivo_mask = st.file_uploader("Máscara (mask.nii.gz)", type=['nii', 'gz'])

# 4. Acción de Procesamiento
if archivo_img and archivo_mask:
    if st.button("Procesar Paciente", type="primary", use_container_width=True):

        with st.spinner("Analizando biomarcadores..."):
            # 1. Guardar en temporales
            ruta_img = guardar_archivo_temporal(archivo_img)
            ruta_mask = guardar_archivo_temporal(archivo_mask)

            try:
                # 2. Llamada a TU MOTOR
                reporte = motor.predecir_paciente(ruta_img, ruta_mask)

                # 3. Llamada al Visor Espacial
                figura = generar_figura_fusion(ruta_img, ruta_mask)

                # 4. Despliegue de Resultados Clínicos (Métricas de Streamlit)
                st.success("¡Análisis completado exitosamente!")
                st.subheader("Reporte Clínico")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Volumen Tumoral", f"{reporte['volumen']['valor']} {reporte['volumen']['unidad']}")
                m2.metric("Eje Corto", f"{reporte['eje_corto']['valor']} {reporte['eje_corto']['unidad']}")
                m3.metric("Eje Largo", f"{reporte['eje_largo']['valor']} {reporte['eje_largo']['unidad']}")

                # Lógica de colores para el riesgo
                riesgo = reporte['riesgo']
                iconos_riesgo = {"Bajo": "🟢", "Moderado": "🟡", "Medio-Alto": "🟠", "Crítico": "🔴"}
                icono = iconos_riesgo.get(riesgo, "⚪")
                m4.metric("Riesgo de Malignidad", f"{icono} {riesgo}")

                # 5. Despliegue de la imagen de fusión
                st.divider()
                st.subheader("Visualización Radiológica")
                if figura:
                    st.pyplot(figura, use_container_width=True)

            except Exception as e:
                st.error(f"Error en el análisis: {e}")