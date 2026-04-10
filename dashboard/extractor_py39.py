<<<<<<< HEAD
import sys
import json
from radiomics import featureextractor


def main():
    # verificamos que se hayan pasado las dos rutas (imagen y máscara)
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Se requieren exactamente 2 argumentos: imagen y máscara"}))
        sys.exit(1)

    ruta_imagen = sys.argv[1]
    ruta_mascara = sys.argv[2]

    try:
        # configurar PyRadiomics
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('gldm')

        # extraer diciéndole que el tumor tiene el valor 255
        resultado = extractor.execute(ruta_imagen, ruta_mascara, label=255)

        # quitar metadatos
        features_limpias = {k.replace("original_", ""): float(v)
                            for k, v in resultado.items()
                            if not k.startswith("diagnostics_")}

        # retornar diccionario como un string json directo a la consola
        print(json.dumps(features_limpias))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
=======
import sys
import json
from radiomics import featureextractor


def main():
    # verificamos que se hayan pasado las dos rutas (imagen y máscara)
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Se requieren exactamente 2 argumentos: imagen y máscara"}))
        sys.exit(1)

    ruta_imagen = sys.argv[1]
    ruta_mascara = sys.argv[2]

    try:
        # configurar PyRadiomics
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('gldm')

        # extraer diciéndole que el tumor tiene el valor 255
        resultado = extractor.execute(ruta_imagen, ruta_mascara, label=255)

        # quitar metadatos
        features_limpias = {k.replace("original_", ""): float(v)
                            for k, v in resultado.items()
                            if not k.startswith("diagnostics_")}

        # retornar diccionario como un string json directo a la consola
        print(json.dumps(features_limpias))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
>>>>>>> 9015f7c609fa2b4ea4bfb8b397c19d6d54040751
    main()