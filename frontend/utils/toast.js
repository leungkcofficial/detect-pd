const DEFAULT_DURATION = 6000;

export function showToast(message, type = 'info', duration = DEFAULT_DURATION) {
  const container = document.querySelector('.toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.role = 'status';
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    if (toast.isConnected) {
      toast.remove();
    }
  }, duration);
}
