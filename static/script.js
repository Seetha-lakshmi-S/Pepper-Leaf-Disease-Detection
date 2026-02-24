/* - FINAL COMPLETE CONSOLIDATED VERSION */
document.addEventListener('DOMContentLoaded', function() {
    console.log("script.js loaded and DOMContentLoaded fired");

    // ============================================================
    // 1. GLOBAL VARIABLES & SHARED STATE
    // ============================================================
    const mainContent = document.getElementById('main-content');
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('app-sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    const navLinks = document.querySelectorAll('.nav-link');

    // Animation State
    let featureSliderAnimId = null;

    // ============================================================
    // 2. MOBILE MENU & SIDEBAR LOGIC
    // ============================================================
    function closeSidebar() {
        if (sidebar) sidebar.classList.remove('active');
        if (overlay) overlay.classList.remove('active');
    }

    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            sidebar.classList.toggle('active');
            if (overlay) overlay.classList.toggle('active');
        });
    }

    if (overlay) {
        overlay.addEventListener('click', closeSidebar);
    }

    // ============================================================
    // 3. INFINITE SLIDER LOGIC (Smooth Motion)
    // ============================================================
    function initFeatureSlider() {
        const sliderContainer = document.getElementById('featuresSlider');
        const sliderInner = document.getElementById('featuresSliderInner');
        
        if (!sliderContainer || !sliderInner) return;

        // Clone content for seamless loop
        if (sliderInner.getAttribute('data-cloned') !== 'true') {
            sliderInner.innerHTML += sliderInner.innerHTML; 
            sliderInner.setAttribute('data-cloned', 'true');
        }

        sliderContainer.addEventListener('mouseenter', stopFeatureSlider);
        sliderContainer.addEventListener('mouseleave', startFeatureSlider);

        startFeatureSlider();
    }

    function startFeatureSlider() {
        const sliderContainer = document.getElementById('featuresSlider');
        if (!sliderContainer) return;
        
        stopFeatureSlider(); 

        function step() {
            if (!sliderContainer) return;
            sliderContainer.scrollLeft += 0.5;

            if (sliderContainer.scrollLeft >= (sliderContainer.scrollWidth / 2)) {
                sliderContainer.scrollLeft = 0;
            }
            featureSliderAnimId = requestAnimationFrame(step);
        }
        featureSliderAnimId = requestAnimationFrame(step);
    }

    function stopFeatureSlider() {
        if (featureSliderAnimId) {
            cancelAnimationFrame(featureSliderAnimId);
            featureSliderAnimId = null;
        }
    }

    // ============================================================
    // 4. AUTH PAGE VALIDATION (Signup)
    // ============================================================
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        const passwordInput = document.getElementById('password');
        const confirmInput = document.getElementById('confirm_password');
        const submitBtn = document.getElementById('submitBtn');
        const matchMsg = document.getElementById('match-message');

        const reqs = {
            len: document.getElementById('req-len'),
            low: document.getElementById('req-low'),
            up: document.getElementById('req-up'),
            num: document.getElementById('req-num'),
            sym: document.getElementById('req-sym')
        };

        const patterns = {
            low: /[a-z]/, up: /[A-Z]/, num: /[0-9]/, sym: /[!@#$%^&*(),.?":{}|<>_+\-=]/
        };

        function validateSignup() {
            if (!passwordInput || !confirmInput) return;
            const val = passwordInput.value;
            const confirmVal = confirmInput.value;
            let allValid = true;

            if (val.length >= 8) reqs.len.classList.add('valid'); else { reqs.len.classList.remove('valid'); allValid = false; }
            if (patterns.low.test(val)) reqs.low.classList.add('valid'); else { reqs.low.classList.remove('valid'); allValid = false; }
            if (patterns.up.test(val)) reqs.up.classList.add('valid'); else { reqs.up.classList.remove('valid'); allValid = false; }
            if (patterns.num.test(val)) reqs.num.classList.add('valid'); else { reqs.num.classList.remove('valid'); allValid = false; }
            if (patterns.sym.test(val)) reqs.sym.classList.add('valid'); else { reqs.sym.classList.remove('valid'); allValid = false; }

            if (confirmVal.length > 0) {
                if (val === confirmVal) {
                    matchMsg.textContent = "Passwords Match ✓";
                    matchMsg.className = "match-success";
                } else {
                    matchMsg.textContent = "Passwords Do Not Match";
                    matchMsg.className = "match-error";
                    allValid = false;
                }
            } else {
                matchMsg.textContent = "";
                allValid = false;
            }
            if (submitBtn) submitBtn.disabled = !allValid;
        }

        passwordInput.addEventListener('input', validateSignup);
        confirmInput.addEventListener('input', validateSignup);
    }

    // ============================================================
    // 5. NAVIGATION & AJAX LOADING
    // ============================================================
    function setActiveNav(panelName) {
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.dataset.panel === panelName) {
                link.classList.add('active');
            }
        });
    }

    window.loadPanel = function(panelName) {
        stopFeatureSlider();

        if (window.innerWidth <= 900) closeSidebar();
        if (mainContent) mainContent.innerHTML = '<div class="spinner-container"><div class="spinner" style="border-color:#064e3b; border-top-color:transparent;"></div></div>';

        let fetchUrl = `/get_panel/${panelName}`;
        if (panelName.startsWith('result-')) {
            const id = panelName.split('-')[1];
            fetchUrl = `/get_panel/result/${id}`;
        }

        fetch(fetchUrl)
            .then(response => {
                if (!response.ok) throw new Error("Panel not found");
                return response.text();
            })
            .then(html => {
                if (mainContent) {
                    mainContent.innerHTML = html;
                    const cleanName = panelName.startsWith('result-') ? 'result' : panelName;
                    document.dispatchEvent(new CustomEvent('panelLoaded', { detail: cleanName }));
                }
                setActiveNav(panelName);
            })
            .catch(err => {
                console.error(err);
                if (mainContent) mainContent.innerHTML = `<div class="alert alert-error">Error loading content. Please try again.</div>`;
            });
    };

    document.body.addEventListener('click', function(e) {
        const target = e.target.closest('.js-switch-panel, .nav-link');
        if (target && target.dataset.panel) {
            e.preventDefault();
            window.location.hash = target.dataset.panel;
        }

        if (e.target.closest('.js-scroll-link')) {
            e.preventDefault();
            const link = e.target.closest('.js-scroll-link');
            const targetId = link.dataset.target;
            const section = document.getElementById(targetId);
            
            if (section) {
                section.scrollIntoView({ behavior: 'smooth' });
            } else {
                window.pendingScrollTarget = targetId;
                window.location.hash = 'welcome';
            }
        }
    });

    window.addEventListener('hashchange', () => {
        const panel = window.location.hash.substring(1) || 'welcome';
        loadPanel(panel);
    });

    // ============================================================
    // 6. PANEL SPECIFIC LOGIC (Re-attached after AJAX)
    // ============================================================

    function attachDiagnoseAJAX() {
        const form = document.getElementById('diagnoseForm');
        const fileInput = document.getElementById('leaf_image');
        const nameDisp = document.getElementById('file-name-display');
        const btn = document.getElementById('diagnoseBtn');
        const uploadLabel = document.querySelector('.upload-label');

        if (fileInput && nameDisp) {
            fileInput.addEventListener('change', function() {
                if (this.files.length > 0) {
                    nameDisp.textContent = "Selected: " + this.files[0].name;
                    nameDisp.style.color = "#10b981";
                    if(uploadLabel) {
                        uploadLabel.style.borderColor = '#10b981';
                        uploadLabel.style.background = '#f0fdf4';
                    }
                }
            });
        }

        if (form) {
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                const spinner = btn.querySelector('.spinner');
                const btnText = btn.querySelector('.btn-text');
                if (spinner) spinner.style.display = 'inline-block';
                if (btnText) btnText.textContent = "Analyzing...";
                btn.disabled = true;

                const formData = new FormData(form);
                try {
                    const response = await fetch('/diagnose', { method: 'POST', body: formData });
                    const result = await response.json();
                    if (result.success && result.redirect) {
                        const id = result.redirect.split('/').pop();
                        window.location.hash = `result-${id}`;
                    } else {
                        alert("Error: " + (result.message || "Upload failed"));
                        resetBtn();
                    }
                } catch (err) {
                    alert("Network error. Please try again.");
                    resetBtn();
                }

                function resetBtn() {
                    if (spinner) spinner.style.display = 'none';
                    if (btnText) btnText.textContent = "Run Diagnosis 🚀";
                    btn.disabled = false;
                }
            });
        }
    }

    function setupHistorySearch() {
        const searchInput = document.getElementById('history-search-input');
        if (searchInput) {
            searchInput.addEventListener('keyup', function() {
                const query = this.value.toLowerCase();
                const cards = document.querySelectorAll('.plant-history-card');
                cards.forEach(card => {
                    const title = card.querySelector('h3').textContent.toLowerCase();
                    card.style.display = title.includes(query) ? "block" : "none";
                });
            });
        }
    }

    function initializeCharts() {
        if (typeof Chart === 'undefined') return;
        const ctx = document.getElementById('severityChart');
        if (ctx) {
            if (ctx.chartInstance) ctx.chartInstance.destroy();
            const counts = JSON.parse(ctx.dataset.counts || "[0,0,0,0]");
            ctx.chartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Healthy', 'Early', 'Mid', 'Advanced'],
                    datasets: [{
                        data: counts,
                        backgroundColor: ['#10b981', '#facc15', '#f97316', '#ef4444'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom' } }
                }
            });
        }
    }

    // ============================================================
    // 7. EVENT DELEGATION FOR STATIC ELEMENTS (PDF & Modals)
    // ============================================================

   document.body.addEventListener('click', function(e) {
        // Panel Switching
        const target = e.target.closest('.js-switch-panel, .nav-link');
        if (target && target.dataset.panel) {
            e.preventDefault();
            window.location.hash = target.dataset.panel;
        }

        // PDF Generation Logic
        const pdfBtn = e.target.closest('#download-pdf-btn');
        if (pdfBtn) {
            const originalText = pdfBtn.innerHTML;
            
            // GET PLANT ID: Extract the plant_id from the button's data attribute
            const plantId = pdfBtn.getAttribute('data-plant-id') || 'Unknown';
            
            pdfBtn.innerHTML = '⏳ Generating...';
            pdfBtn.disabled = true;

            const element = document.getElementById('printable-area'); 
            
            setTimeout(() => {
                if (typeof html2pdf !== 'undefined' && element) {
                    const opt = {
                        margin: 10,
                        // DYNAMIC FILENAME: Including the plant ID in the name
                        filename: `PepGuard_Report_${plantId}.pdf`, 
                        image: { type: 'jpeg', quality: 0.98 },
                        html2canvas: { 
                            scale: 2, 
                            useCORS: true, 
                            letterRendering: true 
                        },
                        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
                    };

                    html2pdf().set(opt).from(element).save()
                        .then(() => {
                            pdfBtn.innerHTML = originalText;
                            pdfBtn.disabled = false;
                        });
                }
            }, 800);
        }
        
        // Modal Open
        const openModalBtn = e.target.closest('.followup-btn');
        if (openModalBtn) {
            const modal = document.getElementById('followup-modal');
            const hiddenInput = document.getElementById('followup-plant-id');
            if (modal && hiddenInput) {
                hiddenInput.value = openModalBtn.dataset.plant;
                modal.classList.remove('modal-hidden');
                modal.style.display = 'flex';
            }
        }

        // Modal Close
        if (e.target.id === 'close-followup-modal' || e.target.classList.contains('modal-close') || e.target.id === 'followup-modal') {
            const modal = document.getElementById('followup-modal');
            if (modal) {
                modal.classList.add('modal-hidden');
                modal.style.display = 'none';
            }
        }
    });

    // Handle Follow-up Form (Delegated)
    if (mainContent) {
        mainContent.addEventListener('change', function(e) {
            if (e.target.id === 'followup_image') {
                const nameDisplay = document.getElementById('followup-file-name');
                if (e.target.files.length > 0 && nameDisplay) {
                    nameDisplay.textContent = "Selected: " + e.target.files[0].name;
                    nameDisplay.style.color = "#10b981";
                }
            }
        });

        mainContent.addEventListener('submit', async function(e) {
            if (e.target.id === 'followup-form') {
                e.preventDefault();
                const btn = e.target.querySelector('button[type="submit"]');
                const original = btn.innerText;
                btn.innerText = "Processing...";
                btn.disabled = true;

                try {
                    const response = await fetch('/diagnose', { method: 'POST', body: new FormData(e.target) });
                    const result = await response.json();
                    if (result.success) {
                        document.getElementById('followup-modal').style.display = 'none';
                        loadPanel('history');
                        alert("Follow-up added!");
                    } else {
                        alert(result.message || 'Error');
                    }
                } catch (err) {
                    alert('Error submitting.');
                } finally {
                    btn.innerText = original;
                    btn.disabled = false;
                }
            }
        });
    }

    // ============================================================
    // 8. MASTER LISTENER (Execute logic when panels load)
    // ============================================================
    document.addEventListener('panelLoaded', function(e) {
        const panel = e.detail;
        if (panel === 'welcome') {
            initFeatureSlider();
            if (window.pendingScrollTarget) {
                setTimeout(() => {
                    const section = document.getElementById(window.pendingScrollTarget);
                    if (section) section.scrollIntoView({ behavior: 'smooth' });
                    window.pendingScrollTarget = null;
                }, 300);
            }
        }
        if (panel === 'new_diagnosis') attachDiagnoseAJAX();
        if (panel === 'history') setupHistorySearch();
        if (panel === 'summary') initializeCharts();
    });

    // Logout Dropdown
    const userMenu = document.getElementById('user-dropdown');
    const logoutDrop = document.getElementById('logout-dropdown');
    if (userMenu && logoutDrop) {
        userMenu.addEventListener('click', (e) => {
            e.stopPropagation();
            logoutDrop.style.display = (logoutDrop.style.display === 'block') ? 'none' : 'block';
        });
        document.addEventListener('click', () => logoutDrop.style.display = 'none');
    }

    // ============================================================
    // 9. APP INITIALIZATION
    // ============================================================
    const initialPanel = window.location.hash.substring(1) || 'welcome';
    loadPanel(initialPanel);
});