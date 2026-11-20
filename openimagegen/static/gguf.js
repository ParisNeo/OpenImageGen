
'use strict';

// Curated GGUF preset catalog. Loaded after the main Studio script.
(() => {
    const modal = document.getElementById('ggufModal');
    const list = document.getElementById('ggufCatalogList');
    const openButton = document.getElementById('ggufCatalogBtn');
    const closeButton = document.getElementById('ggufClose');
    const keyInput = document.getElementById('apiKeyInput');
    let refreshTimer = null;

    function notify(message, type = 'info', duration = 4000) {
        if (typeof showToast === 'function') showToast(message, type, duration);
    }

    async function request(path, options = {}) {
        const response = await fetch(path, {
            ...options,
            headers: { ...authHeaders(), ...(options.headers || {}) },
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        return response.json();
    }

    function createPresetCard(preset) {
        const card = document.createElement('article');
        card.className = `gguf-preset${preset.recommended ? ' recommended' : ''}`;

        const title = document.createElement('div');
        title.className = 'gguf-preset-title';
        title.textContent = `${preset.title}${preset.recommended ? ' · Smallest test' : ''}`;

        const meta = document.createElement('div');
        meta.className = 'gguf-preset-meta';
        meta.textContent = `${preset.quantization} · ~${preset.transformer_size_gb} GB transformer · ${preset.default_steps} steps`;

        const description = document.createElement('div');
        description.className = 'gguf-preset-description';
        description.textContent = preset.description;

        const note = document.createElement('div');
        note.className = 'gguf-preset-note';
        note.textContent = `${preset.total_size_note} Base: ${preset.base_repository}. GGUF: ${preset.gguf_repository}.`;

        const actions = document.createElement('div');
        actions.className = 'gguf-preset-actions';
        const install = document.createElement('button');
        install.type = 'button';
        install.textContent = preset.downloaded
            ? 'Ready'
            : preset.installed
                ? 'Download missing files'
                : 'Install & download';
        install.disabled = preset.downloaded;
        install.addEventListener('click', () => installPreset(preset, install));

        const status = document.createElement('span');
        status.className = 'gguf-preset-status';
        status.textContent = preset.loaded
            ? 'Loaded in memory'
            : preset.downloaded
                ? 'All required files cached'
                : preset.installed
                    ? 'Installed; download incomplete'
                    : 'Not installed';

        actions.append(install, status);
        card.append(title, meta, description, note, actions);
        return card;
    }

    async function loadCatalog() {
        try {
            const data = await request('/gguf/presets');
            list.replaceChildren(...data.presets.map(createPresetCard));
        } catch (error) {
            list.textContent = `Could not load GGUF presets: ${error.message}`;
        }
    }

    async function installPreset(preset, button) {
        button.disabled = true;
        button.textContent = 'Starting…';
        try {
            const result = await request(`/gguf/presets/${preset.id}/install`, { method: 'POST' });
            notify(
                result.download_started
                    ? `${preset.title} download started. Monitor progress in the header.`
                    : `${preset.title} is installed. Download may already be running.`,
                'success',
                6000
            );
            window.dispatchEvent(new CustomEvent('oig:download-started', {
                detail: { job: result.download_job }
            }));
            if (typeof loadModels === 'function') await loadModels();
            await loadCatalog();
            scheduleRefresh();
        } catch (error) {
            notify(`GGUF installation failed: ${error.message}`, 'error', 7000);
            button.disabled = false;
            button.textContent = 'Retry installation';
        }
    }

    function scheduleRefresh() {
        if (refreshTimer) clearTimeout(refreshTimer);
        refreshTimer = setTimeout(async () => {
            if (modal.style.display === 'flex') {
                await loadCatalog();
                if (typeof checkHealth === 'function') await checkHealth();
                scheduleRefresh();
            }
        }, 5000);
    }

    function openCatalog() {
        modal.style.display = 'flex';
        loadCatalog();
        scheduleRefresh();
    }

    function closeCatalog() {
        modal.style.display = 'none';
        if (refreshTimer) clearTimeout(refreshTimer);
        refreshTimer = null;
    }

    try {
        keyInput.value = localStorage.getItem('oig_api_key_v1') || '';
    } catch {}
    keyInput.addEventListener('change', () => {
        try {
            if (keyInput.value.trim()) localStorage.setItem('oig_api_key_v1', keyInput.value.trim());
            else localStorage.removeItem('oig_api_key_v1');
        } catch {}
        if (typeof checkHealth === 'function') checkHealth();
    });

    openButton.addEventListener('click', openCatalog);
    closeButton.addEventListener('click', closeCatalog);
    modal.addEventListener('click', event => {
        if (event.target === modal) closeCatalog();
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && modal.style.display === 'flex') closeCatalog();
    });
})();
