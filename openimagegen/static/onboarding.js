
'use strict';

// First-run Studio tour. Actions open the real product surfaces so the guide
// remains useful after its initial automatic display.
(() => {
    const trigger = document.getElementById('onboardingBtn');
    if (!trigger) return;

    const STORAGE_KEY = 'oig_onboarding_completed_v1';
    const style = document.createElement('style');
    style.textContent = `
        .oig-tour{position:fixed;inset:0;z-index:270;display:flex;align-items:center;justify-content:center;padding:24px;background:#000c;backdrop-filter:blur(7px)}
        .oig-tour[hidden]{display:none}.oig-tour-card{width:min(700px,100%);max-height:90vh;overflow:auto;background:var(--bg-secondary);border:1px solid var(--border-strong);border-radius:16px;box-shadow:var(--shadow-lg)}
        .oig-tour-header{display:flex;align-items:center;justify-content:space-between;padding:20px 24px;border-bottom:1px solid var(--border)}
        .oig-tour-eyebrow{color:var(--accent);font-size:10px;font-weight:750;letter-spacing:.1em;text-transform:uppercase}.oig-tour-header h2{margin-top:4px;font-size:21px}
        .oig-tour-close{border:0;background:none;color:var(--text-secondary);font-size:22px;cursor:pointer}.oig-tour-body{padding:24px}
        .oig-tour-step{display:none}.oig-tour-step.active{display:block;animation:oig-tour-in .2s ease}@keyframes oig-tour-in{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
        .oig-tour-step h3{font-size:18px;margin-bottom:10px}.oig-tour-step p{color:var(--text-secondary);font-size:13px;line-height:1.65;margin-bottom:12px}
        .oig-tour-callout{padding:13px 15px;margin:14px 0;background:var(--bg-input);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:7px;color:var(--text-secondary);font-size:12px;line-height:1.55}
        .oig-tour-callout code{color:var(--accent);overflow-wrap:anywhere}.oig-tour-actions-inline{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
        .oig-tour-link{display:inline-flex;align-items:center;justify-content:center;padding:8px 12px;border:1px solid var(--border);border-radius:7px;background:var(--bg-tertiary);color:var(--text-primary);font-size:12px;font-weight:650;text-decoration:none;cursor:pointer}
        .oig-tour-link.primary{border-color:var(--accent);background:var(--accent-soft);color:var(--accent)}
        .oig-tour-footer{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:16px 24px;border-top:1px solid var(--border)}
        .oig-tour-dots{display:flex;gap:7px}.oig-tour-dot{width:7px;height:7px;border-radius:50%;background:var(--border-strong)}.oig-tour-dot.active{background:var(--accent);box-shadow:0 0 8px var(--accent-glow)}
        .oig-tour-nav{display:flex;gap:8px}.oig-tour-nav button{min-width:82px}.oig-tour-nav button:disabled{opacity:.35;cursor:not-allowed}
        @media(max-width:520px){.oig-tour{padding:12px}.oig-tour-header,.oig-tour-body,.oig-tour-footer{padding:16px}.oig-tour-footer{align-items:flex-end}.oig-tour-nav button{min-width:66px}}
    `;
    document.head.appendChild(style);

    const overlay = document.createElement('div');
    overlay.className = 'oig-tour';
    overlay.hidden = true;
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-labelledby', 'oigTourTitle');
    overlay.innerHTML = `
        <section class="oig-tour-card">
            <header class="oig-tour-header">
                <div><div class="oig-tour-eyebrow">OpenImageGen Studio</div><h2 id="oigTourTitle">Welcome to your studio</h2></div>
                <button class="oig-tour-close" type="button" aria-label="Close guide">×</button>
            </header>
            <div class="oig-tour-body">
                <section class="oig-tour-step active" data-step="0">
                    <h3>From an idea to an image</h3>
                    <p>Describe the subject, setting, lighting, and composition in the Prompt field. Choose a model, then use <strong>Recommended</strong> for settings suited to that model.</p>
                    <div class="oig-tour-callout">Local models may download on their first run. Watch <strong>Downloads</strong> and the header progress. Hosted models skip local weights and run on the provider's infrastructure.</div>
                    <div class="oig-tour-actions-inline"><button class="oig-tour-link primary" type="button" data-action="prompt">Try the prompt editor</button><a class="oig-tour-link" href="/help/quickstart.html" target="_blank" rel="noopener">Quick-start documentation</a></div>
                </section>
                <section class="oig-tour-step" data-step="1">
                    <h3>Connect OpenAI or compatible image APIs</h3>
                    <p>Set the provider credential in the <strong>server process environment</strong>, then open <strong>API</strong> to add the provider model ID and base URL. The OpenAI defaults are <code>OPENAI_API_KEY</code> and <code>https://api.openai.com/v1</code>.</p>
                    <div class="oig-tour-callout">Only the environment variable's name is saved in <code>config.toml</code>. Never paste the provider secret into the model configuration form. The API-key field in the header is for OpenImageGen authentication, not your OpenAI key.</div>
                    <div class="oig-tour-actions-inline"><button class="oig-tour-link primary" type="button" data-action="api">Configure hosted models</button><a class="oig-tour-link" href="/help/onboarding.html#hosted-image-apis" target="_blank" rel="noopener">API setup instructions</a></div>
                </section>
                <section class="oig-tour-step" data-step="2">
                    <h3>Build reusable visual workflows</h3>
                    <p>The workflow editor connects prompts, models, samplers, remote text models, LoRAs, image transforms, and previews into a typed graph. Start with a prompt starter graph, then customize it.</p>
                    <div class="oig-tour-callout">Use <strong>Generate</strong> or <kbd>Ctrl</kbd>+<kbd>Enter</kbd> in Studio. Your recent generations, settings, and favorites stay in this browser; generated files still follow the server's retention policy.</div>
                    <div class="oig-tour-actions-inline"><a class="oig-tour-link primary" href="/workflow">Open workflow editor</a><a class="oig-tour-link" href="/help/index.html" target="_blank" rel="noopener">Browse all documentation</a></div>
                </section>
            </div>
            <footer class="oig-tour-footer">
                <div class="oig-tour-dots" aria-label="Guide progress"><span class="oig-tour-dot active"></span><span class="oig-tour-dot"></span><span class="oig-tour-dot"></span></div>
                <div class="oig-tour-nav"><button class="oig-tour-link" type="button" data-nav="back" disabled>Back</button><button class="oig-tour-link primary" type="button" data-nav="next">Next</button></div>
            </footer>
        </section>`;
    document.body.appendChild(overlay);

    const steps = [...overlay.querySelectorAll('.oig-tour-step')];
    const dots = [...overlay.querySelectorAll('.oig-tour-dot')];
    const back = overlay.querySelector('[data-nav="back"]');
    const next = overlay.querySelector('[data-nav="next"]');
    let currentStep = 0;
    let previouslyFocused = null;

    function markCompleted() {
        try { localStorage.setItem(STORAGE_KEY, 'true'); } catch {}
    }

    function renderStep() {
        steps.forEach((step, index) => step.classList.toggle('active', index === currentStep));
        dots.forEach((dot, index) => dot.classList.toggle('active', index === currentStep));
        back.disabled = currentStep === 0;
        next.textContent = currentStep === steps.length - 1 ? 'Start creating' : 'Next';
        overlay.querySelector('.oig-tour-card').scrollTop = 0;
    }

    function open() {
        previouslyFocused = document.activeElement;
        currentStep = 0;
        renderStep();
        overlay.hidden = false;
        overlay.querySelector('.oig-tour-close').focus();
    }

    function close(completed = true) {
        overlay.hidden = true;
        if (completed) markCompleted();
        previouslyFocused?.focus();
    }

    trigger.addEventListener('click', open);
    overlay.querySelector('.oig-tour-close').addEventListener('click', () => close());
    back.addEventListener('click', () => {
        currentStep = Math.max(0, currentStep - 1);
        renderStep();
    });
    next.addEventListener('click', () => {
        if (currentStep === steps.length - 1) {
            close();
            document.getElementById('prompt')?.focus();
            return;
        }
        currentStep += 1;
        renderStep();
    });
    overlay.addEventListener('click', event => {
        if (event.target === overlay) close();
        const action = event.target.closest('[data-action]')?.dataset.action;
        if (action === 'api') {
            close();
            document.getElementById('apiModelsBtn')?.click();
        } else if (action === 'prompt') {
            close();
            document.getElementById('prompt')?.focus();
        }
    });
    document.addEventListener('keydown', event => {
        if (overlay.hidden) return;
        if (event.key === 'Escape') close();
        if (event.key === 'ArrowRight' && currentStep < steps.length - 1) next.click();
        if (event.key === 'ArrowLeft' && currentStep > 0) back.click();
    });

    let completed = false;
    try { completed = localStorage.getItem(STORAGE_KEY) === 'true'; } catch {}
    if (!completed) setTimeout(open, 500);
})();
