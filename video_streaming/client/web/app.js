/**
 * app.js — Secure Stream Decoder frontend logic.
 *
 * Controls connection to the Flask backend, updates the UI state,
 * and polls status for real-time info display.
 */

// ── State ───────────────────────────────────────────────────────────────────

let isConnected = false;
let isConnecting = false;
let statusInterval = null;

// ── DOM Elements ────────────────────────────────────────────────────────────

const elements = {
    serverHost:       () => document.getElementById('serverHost'),
    serverPort:       () => document.getElementById('serverPort'),
    password:         () => document.getElementById('password'),
    connectBtn:       () => document.getElementById('connectBtn'),
    btnText:          () => document.getElementById('btnText'),
    btnIcon:          () => document.getElementById('btnIcon'),
    statusBadge:      () => document.getElementById('statusBadge'),
    statusDot:        () => document.getElementById('statusDot'),
    statusText:       () => document.getElementById('statusText'),
    videoStream:      () => document.getElementById('videoStream'),
    videoPlaceholder: () => document.getElementById('videoPlaceholder'),
    videoOverlay:     () => document.getElementById('videoOverlay'),
    overlayText:      () => document.getElementById('overlayText'),
    videoContainer:   () => document.getElementById('videoContainer'),
    fpsValue:         () => document.getElementById('fpsValue'),
    frameCount:       () => document.getElementById('frameCount'),
    statusDetail:     () => document.getElementById('statusDetail'),
    togglePassword:   () => document.getElementById('togglePassword'),
};

// ── Initialize ──────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Password toggle
    elements.togglePassword().addEventListener('click', () => {
        const input = elements.password();
        input.type = input.type === 'password' ? 'text' : 'password';
    });

    // Enter key to connect
    document.querySelectorAll('input').forEach(input => {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') toggleConnection();
        });
    });

    // Start status polling
    statusInterval = setInterval(pollStatus, 600);

    // Animated background particles
    createParticles();
});

// ── Connection toggle ───────────────────────────────────────────────────────

async function toggleConnection() {
    if (isConnected || isConnecting) {
        await disconnect();
    } else {
        await connect();
    }
}

async function connect() {
    const host = elements.serverHost().value.trim();
    const port = parseInt(elements.serverPort().value);
    const password = elements.password().value;

    if (!host || !port || !password) {
        shakeElement(elements.connectBtn());
        return;
    }

    setConnecting();

    try {
        const res = await fetch('/api/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ host, port, password }),
        });

        if (!res.ok) {
            const data = await res.json();
            console.error('Connect error:', data.error);
            setDisconnected();
            return;
        }

        // Start showing the stream after a short delay
        setTimeout(() => {
            const stream = elements.videoStream();
            stream.src = '/stream?' + Date.now();
            stream.style.display = 'block';
            stream.onerror = () => {
                // Retry stream connection
                setTimeout(() => {
                    if (isConnected || isConnecting) {
                        stream.src = '/stream?' + Date.now();
                    }
                }, 2000);
            };
        }, 1500);

    } catch (err) {
        console.error('Connection failed:', err);
        setDisconnected();
    }
}

async function disconnect() {
    try {
        await fetch('/api/disconnect', { method: 'POST' });
    } catch (err) {
        console.error('Disconnect error:', err);
    }

    setDisconnected();
}

// ── Status polling ──────────────────────────────────────────────────────────

async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        if (data.connected && !isConnected) {
            setConnected();
        } else if (!data.connected && isConnected) {
            setDisconnected();
        }

        if (data.connected) {
            elements.fpsValue().textContent = data.fps ? data.fps.toFixed(1) : '—';
            elements.frameCount().textContent = data.frame_id || '—';
            elements.statusDetail().textContent = 'Decrypting';
        }
    } catch (err) {
        // Backend not responding
    }
}

// ── UI State management ─────────────────────────────────────────────────────

function setConnecting() {
    isConnecting = true;
    isConnected = false;

    const badge = elements.statusBadge();
    badge.className = 'status-badge connecting';
    elements.statusText().textContent = 'Connecting...';

    const btn = elements.connectBtn();
    btn.classList.add('disconnect');
    elements.btnText().textContent = 'Cancel';
    elements.btnIcon().textContent = '✕';

    elements.videoPlaceholder().style.display = 'none';
    elements.videoOverlay().classList.add('visible');
    elements.overlayText().textContent = 'Connecting to server...';
    elements.statusDetail().textContent = 'Connecting';

    // Disable inputs
    toggleInputs(true);
}

function setConnected() {
    isConnecting = false;
    isConnected = true;

    const badge = elements.statusBadge();
    badge.className = 'status-badge connected';
    elements.statusText().textContent = 'Connected';

    const btn = elements.connectBtn();
    btn.classList.add('disconnect');
    elements.btnText().textContent = 'Disconnect';
    elements.btnIcon().textContent = '⏹';

    elements.videoOverlay().classList.remove('visible');
    elements.videoContainer().classList.add('active');
    elements.statusDetail().textContent = 'Decrypting';
}

function setDisconnected() {
    isConnecting = false;
    isConnected = false;

    const badge = elements.statusBadge();
    badge.className = 'status-badge';
    elements.statusText().textContent = 'Disconnected';

    const btn = elements.connectBtn();
    btn.classList.remove('disconnect');
    elements.btnText().textContent = 'Connect';
    elements.btnIcon().textContent = '▶';

    const stream = elements.videoStream();
    stream.style.display = 'none';
    stream.src = '';

    elements.videoPlaceholder().style.display = 'flex';
    elements.videoOverlay().classList.remove('visible');
    elements.videoContainer().classList.remove('active');

    elements.fpsValue().textContent = '—';
    elements.frameCount().textContent = '—';
    elements.statusDetail().textContent = 'Idle';

    // Re-enable inputs
    toggleInputs(false);
}

function toggleInputs(disabled) {
    elements.serverHost().disabled = disabled;
    elements.serverPort().disabled = disabled;
    elements.password().disabled = disabled;
}

// ── Visual effects ──────────────────────────────────────────────────────────

function shakeElement(el) {
    el.style.animation = 'none';
    el.offsetHeight; // Trigger reflow
    el.style.animation = 'shake 0.4s ease';
    setTimeout(() => el.style.animation = '', 400);
}

// Add shake keyframes dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-6px); }
        50% { transform: translateX(6px); }
        75% { transform: translateX(-4px); }
    }
`;
document.head.appendChild(style);

// ── Background particles ────────────────────────────────────────────────────

function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;

    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        const size = Math.random() * 3 + 1;
        particle.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            background: rgba(6, 214, 160, ${Math.random() * 0.15 + 0.05});
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: particle-drift ${Math.random() * 20 + 15}s linear infinite;
            animation-delay: -${Math.random() * 20}s;
        `;
        container.appendChild(particle);
    }

    // Add particle animation
    const particleStyle = document.createElement('style');
    particleStyle.textContent = `
        @keyframes particle-drift {
            0% { transform: translate(0, 0) scale(1); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translate(${Math.random() > 0.5 ? '' : '-'}${Math.random() * 200 + 50}px, -${Math.random() * 300 + 100}px) scale(0.5); opacity: 0; }
        }
    `;
    document.head.appendChild(particleStyle);
}
