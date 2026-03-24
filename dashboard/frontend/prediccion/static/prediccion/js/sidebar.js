(function () {
    'use strict';

    var historyContainer = document.getElementById('sidebarHistory');
    var emptyMsg         = document.getElementById('sidebarEmpty');
    var clearBtn         = document.getElementById('sidebarClearBtn');

    if (!historyContainer || !window.NQHistory) return;

    function riesgoClass(riesgo) {
        if (riesgo === 'Crítico')    return 'critico';
        if (riesgo === 'Notorio')    return 'notorio';
        if (riesgo === 'Bajo riesgo') return 'bajo';
        return 'sin';
    }

    function renderHistory() {
        var items = window.NQHistory.getAll();

        var existing = historyContainer.querySelectorAll('.sidebar-history-item');
        existing.forEach(function (el) { el.remove(); });

        if (items.length === 0) {
            if (emptyMsg) emptyMsg.style.display = 'block';
            if (clearBtn) clearBtn.style.display = 'none';
            return;
        }

        if (emptyMsg) emptyMsg.style.display = 'none';
        if (clearBtn) clearBtn.style.display = 'inline-flex';

        items.forEach(function (item) {
            var div = document.createElement('div');
            div.className = 'sidebar-history-item';
            div.setAttribute('role', 'button');
            div.setAttribute('tabindex', '0');

            var cls = riesgoClass(item.riesgo);

            div.innerHTML =
                '<div class="shi-top">' +
                    '<span class="shi-fecha">' + window.NQHistory.formatFecha(item.fecha) + '</span>' +
                    '<span class="shi-riesgo ' + cls + '">' + item.riesgo + '</span>' +
                '</div>' +
                '<div class="shi-metrics">' +
                    '<span>' + item.volumen.valor + ' ' + item.volumen.unidad + '</span>' +
                    '<span>' + item.eje_corto.valor + ' / ' + item.eje_largo.valor + ' mm</span>' +
                '</div>';

            div.addEventListener('click', function () {
                navigateToReport(item);
            });
            div.addEventListener('keydown', function (e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    navigateToReport(item);
                }
            });

            historyContainer.appendChild(div);
        });
    }

    function navigateToReport(item) {
        var params = new URLSearchParams({
            volumen_valor:   item.volumen.valor,
            volumen_unidad:  item.volumen.unidad,
            eje_corto_valor: item.eje_corto.valor,
            eje_corto_unidad:item.eje_corto.unidad,
            eje_largo_valor: item.eje_largo.valor,
            eje_largo_unidad:item.eje_largo.unidad,
            riesgo:          item.riesgo,
            from_cache:      '1'
        });
        window.location.href = '/resultado/?' + params.toString();
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            if (confirm('¿Limpiar todo el historial?')) {
                window.NQHistory.clear();
                renderHistory();
            }
        });
    }

    renderHistory();

}());