
'use strict';

// Administration UI for hosted OpenAI-compatible image models.
// Only environment variable names are configured; secret values never enter this form.
(() => {
    const openButton = document.getElementById('apiModelsBtn');
    if (!openButton) return;

    const style = document.createElement('style');
    style.textContent = `
        .oig-api-overlay{position:fixed;inset:0;z-index:260;display:flex;align-items:center;justify-content:center;padding:24px;background:#000b;backdrop-filter:blur(6px)}
        .oig-api-overlay[hidden]{display:none}.oig-api-modal{width:min(760px,100%);max-height:90vh;overflow:auto;background:var(--bg-secondary);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow-lg)}
        .oig-api-header{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--border)}
        .oig-api-header h2{font-size:16px}.oig-api-close{border:0;background:none;color:var(--text-secondary);font-size:20px;cursor:pointer}
        .oig-api-body{padding:18px 20px}.oig-api-description{color:var(--text-secondary);font-size:12px;margin-bottom:16px}
        .oig-api-list{display:grid;gap:8px;margin-bottom:18px}.oig-api-item{display:flex;align-items:center;gap:10px;padding:11px;background:var(--bg-tertiary);border:1px solid var(--border);border-radius:8px}
        .oig-api-item-main{flex:1;min-width:0}.oig-api-item-title{font-weight:700;font-size:12px}.oig-api-item-detail{color:var(--text-tertiary);font-size:11px;overflow-wrap:anywhere}
        .oig-api-item button,.oig-api-submit{padding:7px 10px;border:1px solid var(--border);border-radius:6px;background:var(--bg-input);color:var(--text-secondary);cursor:pointer;font-size:11px}
        .oig-api-item .danger{color:var(--error)}.oig-api-form{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding-top:16px;border-top:1px solid var(--border)}
        .oig-api-form h3{grid-column:1/-1;font-size:13px;color:var(--accent)}.oig-api-field{display:grid;gap:5px}.oig-api-field.wide{grid-column:1/-1}
        .oig-api-field label{font-size:10px;color:var(--text-secondary);font-weight:700;text-transform:uppercase}.oig-api-field input{min-width:0}
        .oig-api-check{display:flex;align-items:center;gap:8px;color:var(--text-secondary);font-size:12px}.oig-api-check input{width:18px;height:18px}
        .oig-api-actions{grid-column:1/-1;display:flex;justify-content:flex-end;gap:8px}.oig-api-submit{background:var(--accent-soft);border-color:var(--accent);color:var(--accent);font-weight:700}
        .oig-api-empty{color:var(--text-tertiary);font-size:12px;padding:12px;border:1px dashed var(--border);border-radius:8px;text-align:center}
        @media(max-width:600px){.oig-api-form{grid-template-columns:1fr}.oig-api-field.wide,.oig-api-form h3,.oig-api-actions{grid-column:1}}
    `;
    document.head.appendChild(style);

    const overlay = document.createElement('div');
    overlay.className = 'oig-api-overlay';
    overlay.hidden = true;
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-labelledby', 'oigApiTitle');
    overlay.innerHTML = `
        <section class="oig-api-modal">
            <header class="oig-api-header"><h2 id="oigApiTitle">Hosted API models</h2><button class="oig-api-close" type="button" aria-label="Close">×</button></header>
            <div class="oig-api-body">
                <p class="oig-api-description">Configure OpenAI or compatible <code>/images/generations</code> providers. Put the secret in the server environment, then enter only its variable name here. Changes are saved to <code>config.toml</code>.</p>
                <div class="oig-api-list"></div>
                <form class="oig-api-form">
                    <h3>Add hosted model</h3>
                    <div class="oig-api-field"><label for="oigApiKey">Configuration key</label><input id="oigApiKey" required minlength="2" maxlength="64" pattern="[A-Za-z0-9_-]+" placeholder="my_gpt_image"></div>
                    <div class="oig-api-field"><label for="oigApiName">Provider model ID</label><input id="oigApiName" required maxlength="256" placeholder="gpt-image-1"></div>
                    <div class="oig-api-field wide"><label for="oigApiBaseUrl">Base URL</label><input id="oigApiBaseUrl" required type="url" value="https://api.openai.com/v1"></div>
                    <div class="oig-api-field"><label for="oigApiEnv">API key environment variable</label><input id="oigApiEnv" maxlength="128" value="OPENAI_API_KEY" placeholder="Empty for no authentication"></div>
                    <div class="oig-api-field"><label for="oigApiQuality">Quality (optional)</label><input id="oigApiQuality" maxlength="64" placeholder="high, hd…"></div>
                    <div class="oig-api-field"><label for="oigApiTimeout">Timeout (seconds)</label><input id="oigApiTimeout" required type="number" min="1" max="600" value="180"></div>
                    <label class="oig-api-check"><input id="oigApiVerify" type="checkbox" checked> Verify TLS certificates</label>
                    <div class="oig-api-actions"><button class="oig-api-cancel-edit" type="button" hidden>Cancel edit</button><button class="oig-api-submit" type="submit">Add model</button></div>
                </form>
            </div>
        </section>`;
    document.body.appendChild(overlay);

    const list = overlay.querySelector('.oig-api-list');
    const form = overlay.querySelector('form');
    const formTitle = form.querySelector('h3');
    const submit = form.querySelector('.oig-api-submit');
    const cancelEdit = form.querySelector('.oig-api-cancel-edit');
    const fields = {
        key: form.querySelector('#oigApiKey'),
        name: form.querySelector('#oigApiName'),
        base_url: form.querySelector('#oigApiBaseUrl'),
        api_key_env: form.querySelector('#oigApiEnv'),
        quality: form.querySelector('#oigApiQuality'),
        timeout_seconds: form.querySelector('#oigApiTimeout'),
        verify_ssl: form.querySelector('#oigApiVerify'),
    };
    let editingKey = null;

    function notify(message, type = 'info', duration = 4500) {
        if (typeof showToast === 'function') showToast(message, type, duration);
    }

    async function request(path, options = {}) {
        const response = await fetch(path, {
            ...options,
            headers: { ...authHeaders(), ...(options.headers || {}) },
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
            const detail = Array.isArray(error.detail)
                ? error.detail.map(item => item.msg).join('; ')
                : error.detail;
            throw new Error(detail || `HTTP ${response.status}`);
        }
        return response.json();
    }

    function resetForm() {
        editingKey = null;
        form.reset();
        fields.key.disabled = false;
        fields.base_url.value = 'https://api.openai.com/v1';
        fields.api_key_env.value = 'OPENAI_API_KEY';
        fields.timeout_seconds.value = '180';
        fields.verify_ssl.checked = true;
        formTitle.textContent = 'Add hosted model';
        submit.textContent = 'Add model';
        cancelEdit.hidden = true;
    }

    function editModel(model) {
        editingKey = model.key;
        fields.key.value = model.key;
        fields.key.disabled = true;
        fields.name.value = model.name || '';
        fields.base_url.value = model.base_url || 'https://api.openai.com/v1';
        fields.api_key_env.value = model.api_key_env ?? 'OPENAI_API_KEY';
        fields.quality.value = model.quality || '';
        fields.timeout_seconds.value = model.timeout_seconds || 180;
        fields.verify_ssl.checked = model.verify_ssl !== false;
        formTitle.textContent = `Edit ${model.key}`;
        submit.textContent = 'Save changes';
        cancelEdit.hidden = false;
        fields.name.focus();
    }

    function modelElement(model) {
        const item = document.createElement('article');
        item.className = 'oig-api-item';
        const main = document.createElement('div');
        main.className = 'oig-api-item-main';
        const title = document.createElement('div');
        title.className = 'oig-api-item-title';
        title.textContent = `${model.key} · ${model.name}`;
        const detail = document.createElement('div');
        detail.className = 'oig-api-item-detail';
        detail.textContent = `${model.base_url} · ${model.api_key_env ? `$${model.api_key_env}` : 'no authentication'}`;
        main.append(title, detail);
        const edit = document.createElement('button');
        edit.type = 'button';
        edit.textContent = 'Edit';
        edit.addEventListener('click', () => editModel(model));
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'danger';
        remove.textContent = 'Delete';
        remove.addEventListener('click', async () => {
            if (!confirm(`Delete hosted model '${model.key}'?`)) return;
            try {
                await request(`/v1/api-models/${encodeURIComponent(model.key)}`, { method: 'DELETE' });
                notify(`Deleted ${model.key}.`, 'success');
                if (editingKey === model.key) resetForm();
                await refresh();
                if (typeof loadModels === 'function') await loadModels();
            } catch (error) {
                notify(`Delete failed: ${error.message}`, 'error', 7000);
            }
        });
        item.append(main, edit, remove);
        return item;
    }

    async function refresh() {
        try {
            const result = await request('/v1/api-models');
            list.replaceChildren();
            if (!result.models.length) {
                const empty = document.createElement('div');
                empty.className = 'oig-api-empty';
                empty.textContent = 'No hosted models configured.';
                list.appendChild(empty);
            } else {
                list.append(...result.models.map(modelElement));
            }
        } catch (error) {
            list.textContent = `Could not load hosted models: ${error.message}`;
        }
    }

    form.addEventListener('submit', async event => {
        event.preventDefault();
        const key = editingKey || fields.key.value.trim();
        const payload = {
            name: fields.name.value.trim(),
            base_url: fields.base_url.value.trim(),
            api_key_env: fields.api_key_env.value.trim(),
            quality: fields.quality.value.trim() || null,
            timeout_seconds: Number(fields.timeout_seconds.value),
            verify_ssl: fields.verify_ssl.checked,
        };
        submit.disabled = true;
        try {
            await request(`/v1/api-models/${encodeURIComponent(key)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            notify(`Saved hosted model '${key}'.`, 'success');
            resetForm();
            await refresh();
            if (typeof loadModels === 'function') await loadModels();
        } catch (error) {
            notify(`Configuration failed: ${error.message}`, 'error', 8000);
        } finally {
            submit.disabled = false;
        }
    });

    function close() {
        overlay.hidden = true;
        resetForm();
    }

    openButton.addEventListener('click', () => {
        overlay.hidden = false;
        resetForm();
        refresh();
    });
    overlay.querySelector('.oig-api-close').addEventListener('click', close);
    cancelEdit.addEventListener('click', resetForm);
    overlay.addEventListener('click', event => {
        if (event.target === overlay) close();
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && !overlay.hidden) close();
    });
})();
