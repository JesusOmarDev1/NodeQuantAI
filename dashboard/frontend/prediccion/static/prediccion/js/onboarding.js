(function () {
    'use strict';

    const STORAGE_KEY = 'nq_onboarding_seen';

    const overlay  = document.getElementById('onboardingOverlay');
    if (!overlay) return;

    const btnNext  = document.getElementById('obNext');
    const btnSkip  = document.getElementById('obSkip');
    const steps    = Array.from(overlay.querySelectorAll('.ob-step'));
    const dots     = Array.from(overlay.querySelectorAll('.ob-dot'));
    const TOTAL    = steps.length;

    let currentStep = 0;

    const spotlightMap = {
        1: 'wrap-imagen',
        2: 'wrap-mascara',
        3: 'submitBtn'
    };

    function removeSpotlights() {
        document.querySelectorAll('.spotlight').forEach(function (el) {
            el.classList.remove('spotlight');
        });
    }

    function goToStep(i) {
        steps[currentStep].classList.remove('active');
        dots[currentStep].classList.remove('active');
        dots[currentStep].classList.add('done');

        currentStep = i;

        steps[currentStep].classList.add('active');
        dots[currentStep].classList.add('active');

        removeSpotlights();
        var targetId = spotlightMap[currentStep];
        if (targetId) {
            var el = document.getElementById(targetId);
            if (el) el.classList.add('spotlight');
        }

        var label = btnNext.querySelector('.ob-btn-next-text');
        if (currentStep === TOTAL - 1) {
            label.textContent = 'Entendido';
            btnNext.classList.add('ob-btn-next--done');
        } else {
            label.textContent = 'Siguiente';
            btnNext.classList.remove('ob-btn-next--done');
        }
    }

    function closeOnboarding(saveToStorage) {
        removeSpotlights();
        overlay.classList.add('ob-closing');
        overlay.addEventListener('animationend', function handler() {
            overlay.style.display = 'none';
            overlay.classList.remove('ob-closing');
            overlay.removeEventListener('animationend', handler);
        });
        if (saveToStorage) {
            try { localStorage.setItem(STORAGE_KEY, '1'); } catch (e) {}
        }
    }

    function openOnboarding() {
        currentStep = 0;
        steps.forEach(function (s, i) { s.classList.toggle('active', i === 0); });
        dots.forEach(function (d, i) {
            d.classList.toggle('active', i === 0);
            d.classList.remove('done');
        });
        var label = btnNext.querySelector('.ob-btn-next-text');
        label.textContent = 'Siguiente';
        btnNext.classList.remove('ob-btn-next--done');
        removeSpotlights();
        overlay.style.display = 'flex';
        overlay.classList.remove('ob-closing');
    }

    try {
        if (localStorage.getItem(STORAGE_KEY) === '1') {
            overlay.style.display = 'none';
        }
    } catch (e) {
    }

    btnNext.addEventListener('click', function () {
        if (currentStep < TOTAL - 1) {
            goToStep(currentStep + 1);
        } else {
            closeOnboarding(false);
        }
    });

    btnSkip.addEventListener('click', function () {
        closeOnboarding(true);
    });

    var resetBtn = document.getElementById('tutorialReset');
    if (resetBtn) {
        resetBtn.addEventListener('click', function () {
            try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
            openOnboarding();
        });
    }
}());