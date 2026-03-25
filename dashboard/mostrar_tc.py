import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import streamlit as st


def generar_figura_fusion(ruta_img, ruta_mask):
    """
    Toma un NIfTI y su máscara, los alinea físicamente y devuelve
    una figura de Matplotlib lista para renderizarse en Streamlit.
    """
    try:
        # 1. Cargar imágenes
        ct_itk = sitk.ReadImage(ruta_img)
        mask_itk = sitk.ReadImage(ruta_mask)

        # 2. ALINEACIÓN ESPACIAL EN VIVO (Lógica de alinear_mascaras.py)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_itk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform())

        mask_alineada = resampler.Execute(mask_itk)

        # 3. Extraer arrays
        img_data = sitk.GetArrayFromImage(ct_itk)
        mask_data = sitk.GetArrayFromImage(mask_alineada)

        # 4. Encontrar el ganglio
        slices_con_ganglio = np.any(mask_data > 0, axis=(1, 2))
        indices_validos = np.where(slices_con_ganglio)[0]

        slice_idx = indices_validos[len(indices_validos) // 2] if len(indices_validos) > 0 else img_data.shape[0] // 2

        img_slice = img_data[slice_idx, :, :]
        mask_slice = mask_data[slice_idx, :, :]

        img_slice = np.rot90(img_slice, 2)
        mask_slice = np.rot90(mask_slice, 2)

        # 5. Dibujar la imagen para Streamlit
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Fondo transparente para que se integre nativamente al tema de Streamlit
        fig.patch.set_alpha(0.0)

        vmin, vmax = np.percentile(img_slice, [2, 98])

        # Panel A: Tomografía
        axes[0].imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Tomografía (Corte {slice_idx})', color='white', pad=10)
        axes[0].axis('off')

        # Panel B: Máscara
        axes[1].imshow(mask_slice, cmap='bone')
        axes[1].set_title('Máscara del Ganglio', color='white', pad=10)
        axes[1].axis('off')

        # Panel C: Fusión
        axes[2].imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax)
        mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)

        # El mapa de color 'autumn' mantiene los tonos cálidos (rojo/naranja) que vimos en tu prueba
        axes[2].imshow(mask_overlay, cmap='autumn', alpha=0.6)
        axes[2].set_title('Fusión Clínica Exacta', color='white', pad=10)
        axes[2].axis('off')

        plt.tight_layout()

        return fig  # En lugar de Base64, devolvemos el objeto Figure

    except Exception as e:
        st.error(f"Error al generar imagen de fusión: {e}")
        return None