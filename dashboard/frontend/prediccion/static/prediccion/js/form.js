(function () {
    'use strict';

    document.querySelectorAll('.file-input').forEach(function (input) {
        var wrap   = input.closest('.file-drop-area');
        var label  = document.getElementById('label-' + input.name);
        var status = wrap.querySelector('.file-status');

        input.addEventListener('change', function () {
            if (this.files[0]) {
                label.textContent = this.files[0].name;
                wrap.classList.add('has-file');
                wrap.classList.remove('spotlight');
                if (status) status.style.background = 'var(--color-success)';
            }
        });

        ['dragenter', 'dragover'].forEach(function (evt) {
            wrap.addEventListener(evt, function (e) {
                e.preventDefault();
                e.stopPropagation();
                wrap.style.borderColor = 'var(--color-accent)';
                wrap.style.background  = 'var(--color-accent-light)';
            });
        });

        ['dragleave', 'drop'].forEach(function (evt) {
            wrap.addEventListener(evt, function (e) {
                e.preventDefault();
                e.stopPropagation();
                if (!wrap.classList.contains('has-file')) {
                    wrap.style.borderColor = '';
                    wrap.style.background  = '';
                }
            });
        });

        wrap.addEventListener('drop', function (e) {
            var files = e.dataTransfer.files;
            if (files.length) {
                input.files = files;
                input.dispatchEvent(new Event('change'));
            }
        });

        wrap.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                input.click();
            }
        });
    });

    var formSubmitted  = false;
    var form           = document.getElementById('mainForm');
    var submitBtn      = document.getElementById('submitBtn');
    var loadingOverlay = document.getElementById('loadingOverlay');
    var bar            = loadingOverlay ? loadingOverlay.querySelector('.loading-progress-bar') : null;
    var loadingText    = loadingOverlay ? loadingOverlay.querySelector('.loading-text') : null;

    if (!form) return;

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        /* Validación básica */
        var inputs = form.querySelectorAll('input[type="file"][required]');
        var valid  = true;

        inputs.forEach(function (input) {
            if (!input.files || !input.files[0]) {
                valid = false;
                var dz = input.closest('.file-drop-area');
                dz.style.borderColor = 'var(--color-danger)';
                setTimeout(function () { dz.style.borderColor = ''; }, 2000);
            }
        });

        if (!valid) return;
        if (formSubmitted) return;

        formSubmitted = true;

        if (loadingOverlay) {
            loadingOverlay.classList.add('active');
            if (bar) bar.style.width = '0%';
        }

        submitBtn.disabled    = true;
        submitBtn.style.opacity = '0.7';
        submitBtn.innerHTML   = '<span class="btn-text">Procesando...</span>';

        var formData  = new FormData(form);
        var action    = submitBtn.getAttribute('formaction') || form.getAttribute('action') || '/predecir/';
        var csrfToken = form.querySelector('[name=csrfmiddlewaretoken]').value;

        var xhr = new XMLHttpRequest();
        var processingInterval = null;

        xhr.upload.addEventListener('progress', function (e) {
            if (e.lengthComputable && bar) {
                var pct = (e.loaded / e.total) * 60;
                bar.style.width = pct.toFixed(1) + '%';
            }
        });

        xhr.upload.addEventListener('load', function () {
            if (bar) bar.style.width = '65%';
            if (loadingText) loadingText.textContent = 'Analizando métricas radiómicas...';

            var current = 65;
            processingInterval = setInterval(function () {
                var step = (92 - current) * 0.04;
                current += step;
                if (bar) bar.style.width = current.toFixed(2) + '%';
                if (current >= 91.9) clearInterval(processingInterval);
            }, 400);
        });

        xhr.addEventListener('load', function () {
            clearInterval(processingInterval);
            if (bar) bar.style.width = '100%';

            if (xhr.status >= 200 && xhr.status < 400) {
                setTimeout(function () {
                    document.open();
                    document.write(xhr.responseText);
                    document.close();
                }, 300);
            } else {
                if (loadingOverlay) loadingOverlay.classList.remove('active');
                formSubmitted = false;
                submitBtn.disabled     = false;
                submitBtn.style.opacity = '';
                submitBtn.innerHTML    = '<span class="btn-text">Iniciar análisis</span><span class="btn-arrow" aria-hidden="true"></span>';
                alert('Error al procesar el estudio. Intenta nuevamente.');
            }
        });

        xhr.addEventListener('error', function () {
            clearInterval(processingInterval);
            if (loadingOverlay) loadingOverlay.classList.remove('active');
            formSubmitted = false;
            submitBtn.disabled     = false;
            submitBtn.style.opacity = '';
            submitBtn.innerHTML    = '<span class="btn-text">Iniciar análisis</span><span class="btn-arrow" aria-hidden="true"></span>';
            alert('Error de conexión. Verifica tu red e intenta de nuevo.');
        });

        xhr.open('POST', action);
        xhr.setRequestHeader('X-CSRFToken', csrfToken);
        xhr.send(formData);
    });

}());