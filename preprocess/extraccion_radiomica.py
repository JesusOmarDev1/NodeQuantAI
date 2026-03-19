import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


def extraer_radiomica(carpeta_origen, archivo_salida):
    print("Iniciando motor PyRadiomics...")

    # 1. Configurar el Extractor
    # Usamos un ancho de bin (binWidth) de 25, que es el estándar recomendado
    # para tomografías (CT) al calcular características de textura.
    configuracion = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkNearestNeighbor}
    extractor = featureextractor.RadiomicsFeatureExtractor(**configuracion)

    # Decidimos qué familias de características queremos extraer:
    extractor.disableAllFeatures()  # Apagamos todo primero para tener control
    extractor.enableFeatureClassByName('shape')  # Volumen, Diámetros, Esfericidad
    extractor.enableFeatureClassByName('firstorder')  # Media, Mediana, Asimetría (HU)
    extractor.enableFeatureClassByName('glcm')  # Textura (Homogeneidad, Contraste)

    # Habilitar las nuevas familias de texturas avanzadas
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')

    lista_resultados = []
    pacientes = [p for p in os.listdir(carpeta_origen) if os.path.isdir(os.path.join(carpeta_origen, p))]

    print(f"Se encontraron {len(pacientes)} pacientes. Extrayendo biomarcadores...")

    for paciente in pacientes:
        ruta_ct = os.path.join(carpeta_origen, paciente, "image.nii.gz")
        ruta_seg = os.path.join(carpeta_origen, paciente, "mask.nii.gz")

        if not (os.path.exists(ruta_ct) and os.path.exists(ruta_seg)):
            continue

        # 2. Validación crítica: ¿Hay un ganglio en esta máscara?
        # PyRadiomics falla si le pasamos una máscara vacía (puros 0s)
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(ruta_seg))
        if mask_arr.sum() == 0:
            print(f"  [Aviso] Paciente {paciente} no tiene ganglios anotados. Saltando...")
            continue

        try:
            print(f" -> Procesando paciente: {paciente}")
            # 3. La magia de PyRadiomics
            resultado = extractor.execute(ruta_ct, ruta_seg, label=255)

            # Limpiamos el diccionario (PyRadiomics devuelve mucha metadata que no necesitamos ahora)
            caracteristicas_limpias = {'Paciente_ID': paciente}
            for llave, valor in resultado.items():
                if llave.startswith('original_'):  # Solo nos interesan las métricas calculadas
                    # Renombramos para que sea más legible en Excel/Pandas
                    nombre_corto = llave.replace('original_', '')
                    caracteristicas_limpias[nombre_corto] = float(valor)

            lista_resultados.append(caracteristicas_limpias)

        except Exception as e:
            print(f"  [Error] Fallo al procesar {paciente}: {e}")

    # 4. Convertir a DataFrame y guardar en Excel/CSV
    if lista_resultados:
        df = pd.DataFrame(lista_resultados)
        df.to_csv(archivo_salida, index=False)
        print(f"\n¡Éxito! Base de datos tabular guardada en: {archivo_salida}")
        print(f"Dimensiones de la tabla: {df.shape[0]} ganglios x {df.shape[1] - 1} características.")
    else:
        print("\nNo se pudo extraer información de ningún paciente.")


# ==========================================
# EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    carpeta_preprocesada = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\Proyecto\Local\Dataset_Preprocesado"
    csv_salida = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\Proyecto\Local\nueva_ganglios_radiomica.csv"

    extraer_radiomica(carpeta_preprocesada, csv_salida)