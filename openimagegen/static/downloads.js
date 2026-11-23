
'use strict';

// Shared persistent download center for Studio and the workflow editor.
(() => {
    const style = document.createElement('style');
    style.textContent = `
        .oig-downloads-toggle{position:fixed;left:18px;bottom:18px;z-index:360;padding:9px 13px;border:1px solid var(--accent,#10b981);border-radius:8px;background:var(--bg-elevated,#222831);color:var(--text-primary,#fff);cursor:pointer;font:600 12px Inter,sans-serif;box-shadow:0 8px 24px #0008}
        .oig-downloads{position:fixed;left:18px;bottom:62px;z-index:360;width:min(430px,calc(100vw - 36px));max-height:55vh;overflow:auto;background:var(--bg-secondary,#12151b);border:1px solid var(--border-strong,#3a414d);border-radius:12px;box-shadow:0 16px 45px #000b;color:var(--text-primary,#fff);font:12px Inter,sans-serif}
        .oig-downloads[hidden]{display:none}.oig-downloads-header{display:flex;justify-content:space-between;align-items:center;padding:12px 14px;border-bottom:1px solid var(--border,#2a3038);font-weight:700}
        .oig-downloads-close{border:0;background:none;color:var(--text-secondary,#aaa);font-size:18px;cursor:pointer}.oig-download{padding:12px 14px;border-bottom:1px solid var(--border,#2a3038)}
        .oig-download:last-child{border-bottom:0}.oig-download-top{display:flex;justify-content:space-between;gap:10px}.oig-download-label{font-weight:650;overflow-wrap:anywhere}.oig-download-status{color:var(--accent,#10b981);text-transform:uppercase;font-size:10px;white-space:nowrap}
        .oig-download.failed .oig-download-status{color:var(--error,#ef4444)}.oig-download-message{margin:5px 0 8px;color:var(--text-secondary,#a0a6b0);line-height:1.4;overflow-wrap:anywhere}
        .oig-download-track{height:7px;border-radius:5px;background:var(--bg-tertiary,#1a1f27);overflow:hidden}.oig-download-fill{height:100%;width:0;background:linear-gradient(90deg,#10b981,#06b6d4);transition:width .3s}
        .oig-download.indeterminate .oig-download-fill{width:35%;animation:oig-download-slide 1.2s ease-in-out infinite}@keyframes oig-download-slide{0%{transform:translateX(-110%)}100%{transform:translateX(300%)}}
        .oig-download-empty{padding:18px;color:var(--text-secondary,#a0a6b0);text-align:center}
    `;
    document.head.appendChild(style);

    const panel = document.createElement('section');
    panel.className = 'oig-downloads';
    panel.hidden = true;
    panel.setAttribute('aria-label', 'Model downloads');
    panel.innerHTML = '<div class="oig-downloads-header"><span>Model downloads</span><button class="oig-downloads-close" aria-label="Close">×</button></div><div class="oig-download-list"></div>';

    const toggle = document.createElement('button');
    toggle.className = 'oig-downloads-toggle';
    toggle.type = 'button';
    toggle.textContent = 'Downloads';
    toggle.hidden = true;

    document.body.append(panel, toggle);
    const list = panel.querySelector('.oig-download-list');
    let timer = null;
    let latestJobs = [];
    let requestInFlight = false;
    let watchUntil = 0;

    function headers() {
        return typeof authHeaders === 'function' ? authHeaders() : {};
    }

    function createJobElement(job) {
        const article = document.createElement('article');
        const active = job.status === 'pending' || job.status === 'running';
        const indeterminate = active && job.progress === null;
        article.className = `oig-download ${job.status}${indeterminate ? ' indeterminate' : ''}`;

        const top = document.createElement('div');
        top.className = 'oig-download-top';
        const label = document.createElement('div');
        label.className = 'oig-download-label';
        label.textContent = job.label || job.job_id;
        const status = document.createElement('span');
        status.className = 'oig-download-status';
        status.textContent = job.status;
        top.append(label, status);

        const message = document.createElement('div');
        message.className = 'oig-download-message';
        message.textContent = job.message || '';

        const track = document.createElement('div');
        track.className = 'oig-download-track';
        const fill = document.createElement('div');
        fill.className = 'oig-download-fill';
        fill.style.width = `${Math.max(0, Math.min(100, job.progress || 0))}%`;
        track.appendChild(fill);
        article.append(top, message, track);
        return article;
    }

    function render() {
        list.replaceChildren();
        if (!latestJobs.length) {
            const empty = document.createElement('div');
            empty.className = 'oig-download-empty';
            empty.textContent = 'No model downloads yet.';
            list.appendChild(empty);
        } else {
            list.append(...latestJobs.slice(0, 12).map(createJobElement));
        }
        const active = latestJobs.filter(job => job.status === 'pending' || job.status === 'running').length;
        toggle.hidden = latestJobs.length === 0;
        toggle.textContent = active ? `Downloads · ${active} active` : 'Downloads';
        if (active && !panel.hidden) panel.scrollIntoView({ block: 'nearest' });
    }

    async function refresh() {
        if (requestInFlight) return;
        requestInFlight = true;
        try {
            const response = await fetch('/v1/downloads', { headers: headers() });
            if (!response.ok) return;
            const data = await response.json();
            latestJobs = data.jobs || [];
            render();
        } catch {
            // The normal health indicator remains responsible for offline status.
        } finally {
            requestInFlight = false;
        }
    }

    function hasActiveJobs() {
        return latestJobs.some(job => job.status === 'pending' || job.status === 'running');
    }

    function schedulePoll() {
        if (timer) clearTimeout(timer);
        let delay = 0;
        if (hasActiveJobs()) delay = 1000;
        else if (Date.now() < watchUntil) delay = 1000;
        else if (!panel.hidden) delay = 5000;
        if (delay > 0) {
            timer = setTimeout(runPollTick, delay);
        } else {
            timer = null;
        }
    }

    async function runPollTick() {
        await refresh();
        if (hasActiveJobs() && Date.now() < watchUntil) panel.hidden = false;
        schedulePoll();
    }

    async function kickPoll(options = {}) {
        if (options.watchSeconds) {
            watchUntil = Date.now() + options.watchSeconds * 1000;
        }
        await refresh();
        if (options.autoOpen && hasActiveJobs()) panel.hidden = false;
        schedulePoll();
    }

    async function open() {
        panel.hidden = false;
        await refresh();
        schedulePoll();
    }

    function close() {
        panel.hidden = true;
        schedulePoll();
    }

    panel.querySelector('.oig-downloads-close').addEventListener('click', close);
    toggle.addEventListener('click', () => {
        if (panel.hidden) open();
        else close();
    });
    document.getElementById('downloadsBtn')?.addEventListener('click', open);
    window.addEventListener('oig:download-started', () => { open(); });

    window.OIGDownloads = { open, refresh: kickPoll };
    render();
    kickPoll();
})();
