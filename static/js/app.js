/**
 * Smart Parking Assistant - Client Application JavaScript
 */

// Toast notification helper
function showToast(message, type = 'primary') {
    const toastEl = document.getElementById('appToast');
    if (!toastEl) return;

    const toastMessage = document.getElementById('toastMessage');
    toastEl.className = `toast align-items-center text-white border-0 bg-${type}`;

    let icon = 'fa-info-circle';
    if (type === 'success') icon = 'fa-circle-check';
    if (type === 'danger') icon = 'fa-circle-exclamation';
    if (type === 'warning') icon = 'fa-triangle-exclamation';

    toastMessage.innerHTML = `<i class="fa-solid ${icon} fs-5"></i> <div>${message}</div>`;
    const toast = new bootstrap.Toast(toastEl, { delay: 4000 });
    toast.show();
}

// Camera Stream Manager
class CameraManager {
    constructor(videoElementId) {
        this.videoEl = document.getElementById(videoElementId);
        this.stream = null;
        this.isActive = false;
    }

    async start() {
        if (!this.videoEl) return false;
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' }
            });
            this.videoEl.srcObject = this.stream;
            this.videoEl.play();
            this.isActive = true;
            return true;
        } catch (err) {
            console.warn("Could not access camera:", err);
            return false;
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.videoEl) {
            this.videoEl.srcObject = null;
        }
        this.isActive = false;
    }

    captureBlob() {
        if (!this.isActive || !this.videoEl) return null;
        const canvas = document.createElement('canvas');
        canvas.width = this.videoEl.videoWidth || 640;
        canvas.height = this.videoEl.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.videoEl, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.85);
    }
}

// Format duration minutes to human readable
function formatDuration(minutes) {
    if (!minutes || minutes < 1) return 'Just arrived';
    const hrs = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hrs === 0) return `${mins}m`;
    return `${hrs}h ${mins}m`;
}
