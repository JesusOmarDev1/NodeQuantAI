import os
import sys
import tempfile
from django.shortcuts import render
from .forms import PredicionForm

# Apuntamos al backend
ruta_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
if ruta_backend not in sys.path:
    sys.path.insert(0, ruta_backend)

from motor_inferencia import MotorInferenciaNodeQuant

ruta_raiz = os.path.abspath(os.path.join(ruta_backend, '..', '..'))
RUTA_REG = os.path.join(ruta_raiz, 'regression', 'joblib')
RUTA_CLF = os.path.join(ruta_raiz, 'classification', 'joblib')
motor = MotorInferenciaNodeQuant(RUTA_REG, RUTA_CLF)

def index(request):
    form = PredicionForm()
    return render(request, 'prediccion/index.html', {'form': form})

def predecir(request):
    if request.method == 'POST':
        form = PredicionForm(request.POST, request.FILES)
        if form.is_valid():
            with tempfile.TemporaryDirectory() as tmpdir:
                img = request.FILES['imagen']
                mask = request.FILES['mascara']

                ruta_img = os.path.join(tmpdir, img.name)
                ruta_mask = os.path.join(tmpdir, mask.name)

                with open(ruta_img, 'wb') as f:
                    for chunk in img.chunks():
                        f.write(chunk)
                with open(ruta_mask, 'wb') as f:
                    for chunk in mask.chunks():
                        f.write(chunk)

                reporte = motor.predecir_paciente(ruta_img, ruta_mask)
                viz = motor.generar_visualizacion(ruta_img, ruta_mask)

            return render(request, 'prediccion/resultado.html', {
                'reporte': reporte,
                'viz': viz,
            })
    return render(request, 'prediccion/index.html', {'form': PredicionForm()})


def resultado(request):
    """Vista GET para reconstruir un reporte desde el historial del sidebar."""
    reporte = {
        'volumen': {
            'valor': request.GET.get('volumen_valor', '—'),
            'unidad': request.GET.get('volumen_unidad', 'mm³'),
        },
        'eje_corto': {
            'valor': request.GET.get('eje_corto_valor', '—'),
            'unidad': request.GET.get('eje_corto_unidad', 'mm'),
        },
        'eje_largo': {
            'valor': request.GET.get('eje_largo_valor', '—'),
            'unidad': request.GET.get('eje_largo_unidad', 'mm'),
        },
        'riesgo': request.GET.get('riesgo', 'Desconocido'),
    }
    return render(request, 'prediccion/resultado.html', {
        'reporte': reporte,
        'viz': None,
        'from_cache': True,
    })