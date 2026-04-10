import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


def extraer_radiomica(carpeta_origen, archivo_salida):
    print("Iniciando motor PyRadiomics...")

    # configurar el extractor
    # usamos un ancho de bin (binWidth) de 25
    # calcular características de textura
    configuracion = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkNearestNeighbor}
    extractor = featureextractor.RadiomicsFeatureExtractor(**configuracion)

    # familias de carzteristicas
    extractor.disableAllFeatures()  # apagamos todo primero para tener control
    extractor.enableFeatureClassByName('shape') # forma 3D
    extractor.enableFeatureClassByName('firstorder') # distribución de intensidades de vóxeles
    extractor.enableFeatureClassByName('glcm') # co-ocurrencia de pares de niveles de gris
    extractor.enableFeatureClassByName('glrlm') # corridas de niveles de gris
    extractor.enableFeatureClassByName('glszm') # grupos contiguos de vóxeles con el mismo nivel de gris
    extractor.enableFeatureClassByName('gldm') # dependencia de niveles de gris

    lista_resultados = []
    pacientes = [p for p in os.listdir(carpeta_origen) if os.path.isdir(os.path.join(carpeta_origen, p))]

    print(f"Se encontraron {len(pacientes)} pacientes. Extrayendo biomarcadores...")

    for paciente in pacientes:
        ruta_ct = os.path.join(carpeta_origen, paciente, "image.nii.gz")
        ruta_seg = os.path.join(carpeta_origen, paciente, "mask.nii.gz")

        if not (os.path.exists(ruta_ct) and os.path.exists(ruta_seg)):
            continue

        # valisando que sí haya una máscara
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(ruta_seg))
        if mask_arr.sum() == 0:
            print(f"  [Aviso] Paciente {paciente} no tiene ganglios anotados. Saltando...")
            continue

        try:
            # extrayendo características
            print(f" -> Procesando paciente: {paciente}")
            resultado = extractor.execute(ruta_ct, ruta_seg, label=255)

            # limpiar diccionario, no es necesario conservarlo
            caracteristicas_limpias = {'Paciente_ID': paciente}
            for llave, valor in resultado.items():
                if llave.startswith('original_'):
                    # renombrar para que sea más legible en excel
                    nombre_corto = llave.replace('original_', '')
                    caracteristicas_limpias[nombre_corto] = float(valor)

            lista_resultados.append(caracteristicas_limpias)

        except Exception as e:
            print(f"  [Error] Fallo al procesar {paciente}: {e}")

    # convertir a dataframe y guardar en csv
    if lista_resultados:
        df = pd.DataFrame(lista_resultados)
        df.to_csv(archivo_salida, index=False)
        print(f"\n¡Éxito! Base de datos tabular guardada en: {archivo_salida}")
        print(f"Dimensiones de la tabla: {df.shape[0]} ganglios x {df.shape[1] - 1} características.")
    else:
        print("\nNo se pudo extraer información de ningún paciente.")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    carpeta_preprocesada = os.path.join(base_dir, "Dataset_Preprocesado")
    csv_salida = os.path.join(base_dir, "db", "ganglios_radiomica.csv")

    extraer_radiomica(carpeta_preprocesada, csv_salida)