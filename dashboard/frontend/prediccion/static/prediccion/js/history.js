
(function () {
    'use strict';

    var STORAGE_KEY = 'nq_history';
    var MAX_ITEMS   = 10;

    function getAll() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
        } catch (e) {
            return [];
        }
    }

    function save(items) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
        } catch (e) {}
    }

    function add(entry) {
        var items = getAll();
        items.unshift({
            id:        Date.now(),
            fecha:     new Date().toISOString(),
            volumen:   entry.volumen,
            eje_corto: entry.eje_corto,
            eje_largo: entry.eje_largo,
            riesgo:    entry.riesgo
        });
        if (items.length > MAX_ITEMS) {
            items = items.slice(0, MAX_ITEMS);
        }
        save(items);
    }

    function clear() {
        save([]);
    }

    function formatFecha(iso) {
        var d = new Date(iso);
        return d.toLocaleDateString('es-MX', {
            day:   '2-digit',
            month: 'short',
            year:  'numeric',
            hour:  '2-digit',
            minute:'2-digit'
        });
    }

    window.NQHistory = {
        getAll:      getAll,
        add:         add,
        clear:       clear,
        formatFecha: formatFecha
    };

}());