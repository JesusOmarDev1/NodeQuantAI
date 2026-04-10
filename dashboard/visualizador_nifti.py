"""
Generador de visualizaciones CT con overlay de máscara.
Ejecutado como subproceso con Python 3.9 (SimpleITK + matplotlib).

Entrada (argv):  ruta_imagen  ruta_mascara
Salida (stdout): JSON con imágenes base64 y metadata.
"""

import sys
import json
import base64
import io
import numpy as np

try:
    import SimpleITK as sitk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError as e:
    print(json.dumps({"error": f"Dependencia faltante: {e}"}))
    sys.exit(1)


def cargar_volumen(ruta):
    """Lee un volumen NIfTI y devuelve array + spacing."""
    img = sitk.ReadImage(ruta)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return arr, spacing


def encontrar_corte_optimo(mask_arr):
    """Encuentra el corte axial con mayor área de máscara."""
    areas = mask_arr.sum(axis=(1, 2))
    if areas.max() == 0:
        return mask_arr.shape[0] // 2
    return int(np.argmax(areas))


def aplicar_ventana_hu(arr, centro=40, ancho=400):
    """Windowing de Hounsfield para tejido blando."""
    vmin = centro - ancho / 2
    vmax = centro + ancho / 2
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin)
    return arr.astype(np.float32)


def renderizar_ct_original(ct_slice):
    """Renderiza un corte CT como imagen PNG en base64."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    ax.imshow(ct_slice, cmap="bone", interpolation="bilinear")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def renderizar_ct_overlay(ct_slice, mask_slice):
    """Renderiza un corte CT con overlay de máscara segmentada."""
    # Colormap personalizado para la máscara: transparente → cyan
    colores = [(0, 0, 0, 0), (0, 0.85, 0.9, 0.55)]
    cmap_mask = LinearSegmentedColormap.from_list("nq_mask", colores, N=256)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    ax.imshow(ct_slice, cmap="bone", interpolation="bilinear")

    mask_float = mask_slice.astype(np.float32)
    if mask_float.max() > 0:
        ax.imshow(mask_float, cmap=cmap_mask, interpolation="nearest")
        # Contorno de la máscara
        ax.contour(mask_float, levels=[0.5], colors=["#00d9e6"], linewidths=1.2)

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Uso: visualizador_nifti.py <imagen> <mascara>"}))
        sys.exit(1)

    ruta_imagen = sys.argv[1]
    ruta_mascara = sys.argv[2]

    try:
        ct_arr, spacing = cargar_volumen(ruta_imagen)
        mask_arr, _ = cargar_volumen(ruta_mascara)

        corte = encontrar_corte_optimo(mask_arr)
        ct_slice = aplicar_ventana_hu(ct_arr[corte].astype(np.float64))
        mask_slice = (mask_arr[corte] > 0).astype(np.uint8)

        img_original = renderizar_ct_original(ct_slice)
        img_overlay = renderizar_ct_overlay(ct_slice, mask_slice)

        resultado = {
            "img_original": img_original,
            "img_overlay": img_overlay,
            "corte": int(corte),
            "total_cortes": int(ct_arr.shape[0]),
            "spacing": [round(s, 3) for s in spacing],
            "ventana_hu": "Tejido blando (C:40 W:400)",
        }
        print(json.dumps(resultado))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
