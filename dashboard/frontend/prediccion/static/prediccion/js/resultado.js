(function () {
    'use strict';

    if (!window.NQHistory) return;

    var dataEl = document.getElementById('reporteData');
    if (!dataEl) return;

    var reporte;
    try {
        reporte = JSON.parse(dataEl.textContent);
    } catch (e) {
        return;
    }

    var params = new URLSearchParams(window.location.search);
    if (params.get('from_cache') === '1') return;

    window.NQHistory.add({
        volumen:   reporte.volumen,
        eje_corto: reporte.eje_corto,
        eje_largo: reporte.eje_largo,
        riesgo:    reporte.riesgo
    });

}());