import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

# --- Parameters ---
d = 12
N_t_1 = 1000
R = 10.0
num_samples = 200000

print(f"Generating {num_samples} clipped samples...")

# 1. Sample and clip
Z_samples = np.random.randn(num_samples, d)
Z_clipped = np.clip(Z_samples, -R, R)
z_norms = np.linalg.norm(Z_clipped, axis=1)
z_norms = np.maximum(z_norms, 1e-8)  # Prevent division by zero

# --- Evaluation Function ---
def compute_integral_1d(tau, gamma):
    A = np.sqrt(N_t_1) * gamma
    term1 = (tau * A) / z_norms
    term2 = z_norms / (2 * A)
    return np.mean(1 - norm.cdf(term1 - term2))

# --- Plot Setup ---
x_vals = np.linspace(0.01, 5.0, 100)
fixed_values = [0.2, 0.5, 1.0, 2.0, 8.0,20.0]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Extract shades from colormaps (using 0.4 to 1.0 to avoid invisible light colors)
blue_shades = cm.Blues(np.linspace(0.4, 1.0, len(fixed_values)))
red_shades = cm.Reds(np.linspace(0.4, 1.0, len(fixed_values)))

# --- Plot 1: Integral vs tau (varying gamma) ---
print("Computing Integral vs tau...")
for i, gamma_fixed in enumerate(fixed_values):
    y_vals = [compute_integral_1d(t, gamma_fixed) for t in x_vals]
    ax.plot(x_vals, y_vals, lw=2, color=blue_shades[i], label=rf'$\gamma = {gamma_fixed}$')

ax.set_title(r'$p_\tau$ vs $\tau$ (for fixed $\gamma$)', fontsize=20)
ax.set_xlabel(r'Threshold $\tau$', fontsize=20)
ax.set_ylabel(r'$p_\tau$', fontsize=20)
ax.legend(title=r'Fixed $\gamma$', fontsize=22)
ax.grid(True, linestyle='--', alpha=0.6)
