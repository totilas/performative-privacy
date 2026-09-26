import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar

# ============================================================
# Plot Aesthetics
# ============================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "image.cmap": "magma",
    "axes.grid": False
})

# ============================================================
# System Constants
# ============================================================
N0 = 1000                  # Population size
N = int(N0)
T = 50                     # Finite time horizon
d = 12                     # Data dimension
tr_sigma_inv = d           # Tr(Sigma^{-1}) for Sigma = I
tau = 0.02                 # Privacy threshold
R = 10.0                   # L2 clipping radius

# ============================================================
# Monte Carlo Parameters
# ============================================================
np.random.seed(42)

# Number of focal individuals / datasets sampled for the
# marginal leakage expectation.
NUM_SAMPLES = 50_000

# Generate the full datasets in chunks to avoid excessive memory use.
CHUNK_SIZE = 500

# ============================================================
# Helper: L2 Clip
# ============================================================
def clip_l2(X, radius):
    """
    Clip each row/vector of X to have L2 norm at most 'radius'.

    X shape:
        (..., d)
    """
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    scale = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return X * scale


# ============================================================
# Exact Dataset-Level Monte Carlo for the New Formulation
# ============================================================
# For each Monte Carlo draw:
#
#   z_bar = clipped focal record
#   mu_bar = empirical mean of the entire clipped dataset
#
# and the likelihood ratio depends on
#
#   ||z_bar - mu_bar||.
#
# This is the quantity appearing in the remove-and-recompute
# likelihood ratio.
# ============================================================

effective_norms = np.empty(NUM_SAMPLES, dtype=np.float64)

num_generated = 0

while num_generated < NUM_SAMPLES:
    batch_size = min(CHUNK_SIZE, NUM_SAMPLES - num_generated)

    # --------------------------------------------------------
    # Generate focal record z
    # --------------------------------------------------------
    z = np.random.randn(batch_size, d)

    # --------------------------------------------------------
    # Generate the other N-1 records
    # --------------------------------------------------------
    others = np.random.randn(batch_size, N - 1, d)

    # --------------------------------------------------------
    # Clip focal records and all other records individually
    # --------------------------------------------------------
    z_bar = clip_l2(z, R)
    others_bar = clip_l2(others, R)

    # --------------------------------------------------------
    # Empirical mean of the clipped full dataset
    #
    # mu_bar =
    # (z_bar + sum(other clipped records)) / N
    # --------------------------------------------------------
    mu_bar = (
        z_bar + np.sum(others_bar, axis=1)
    ) / N

    # --------------------------------------------------------
    # Effective distance governing leakage
    # --------------------------------------------------------
    delta = z_bar - mu_bar
    delta_norm = np.linalg.norm(delta, axis=1)

    effective_norms[
        num_generated:num_generated + batch_size
    ] = delta_norm

    num_generated += batch_size

# Remove numerically zero values
effective_norms = effective_norms[effective_norms > 1e-12]

print(
    f"Generated {len(effective_norms):,} Monte Carlo samples "
    f"for ||z_bar - mu_bar||."
)

# ============================================================
# Marginal Leakage Probability
# ============================================================
def p_tau_mc(gamma):
    """
    Computes the marginal leakage probability for the
    remove-and-recompute empirical mean mechanism.

    Conditional leakage probability:

        1 - Phi(
            tau (N-1) gamma / (sqrt(N) ||delta||)
            -
            sqrt(N) ||delta|| /
            (2 (N-1) gamma)
        )

    where delta = z_bar - mu_bar.
    """

    effective_norm = effective_norms

    term1 = (
        tau * (N - 1) * gamma
        / (np.sqrt(N) * effective_norm)
    )

    term2 = (
        np.sqrt(N) * effective_norm
        / (2.0 * (N - 1) * gamma)
    )

    leakage_probability = 1.0 - norm.cdf(term1 - term2)

    return np.mean(leakage_probability)


# ============================================================
# Objective Function
# ============================================================
def cumulative_error(gamma, q, r):
    """
    Finite-horizon closed-form objective.
    """

    # --------------------------------------------------------
    # Expected marginal leakage
    # --------------------------------------------------------
    p_t = p_tau_mc(gamma)

    # --------------------------------------------------------
    # Performative population growth factor
    # --------------------------------------------------------
    C_gamma = 1.0 + r - (q + r) * p_t

    # --------------------------------------------------------
    # Mechanism variance + intrinsic variance
    # --------------------------------------------------------
    num = (gamma**2 * tr_sigma_inv) + d

    # --------------------------------------------------------
    # Handle C(gamma) approximately equal to 1
    # --------------------------------------------------------
    if abs(C_gamma - 1.0) < 1e-8:
        return (num * T) / N

    # --------------------------------------------------------
    # Finite geometric-series objective
    # --------------------------------------------------------
    denom = N * (C_gamma - 1.0)

    geometric_factor = (
        1.0 - (1.0 / (C_gamma ** T))
    )

    return (num / denom) * geometric_factor


# ============================================================
# Grid Search
# ============================================================
GRID_RESOLUTION = 30

q_values = np.linspace(0.01, 0.99, GRID_RESOLUTION)
r_values = np.linspace(0.01, 0.99, GRID_RESOLUTION)

optimal_gammas = np.zeros(
    (GRID_RESOLUTION, GRID_RESOLUTION)
)

print(
    f"Running numerical optimization over "
    f"{GRID_RESOLUTION}x{GRID_RESOLUTION} grid..."
)

# Open a text file to write the results
output_file = open(f"optimal_gammas_theory_{tau}.txt", "w")

for i, q in enumerate(q_values):

    for j, r in enumerate(r_values):

        result = minimize_scalar(
            cumulative_error,
            bounds=(0.02, 20.0),
            args=(q, r),
            method="bounded"
        )

        optimal_gammas[i, j] = result.x

        # Create the formatted output string
        log_line = (
            f"Optimal gamma for "
            f"(q, r) = ({q:.2f}, {r:.2f}): "
            f"{result.x:.4f}"
        )
        
        # Print to console
        print(log_line)
        
        # Write to the text file
        output_file.write(log_line + "\n")

# Close the file after the grid search is done
output_file.close()

# ============================================================
# Plot Heatmap
# ============================================================
plt.figure(figsize=(9, 7))

heatmap = plt.pcolormesh(
    q_values,
    r_values,
    optimal_gammas.T,
    shading="nearest",
    cmap="viridis",
    vmin=0.2,
    vmax=12.5
)

cbar = plt.colorbar(heatmap)
cbar.set_label(
    r"Optimal $\gamma$",
    rotation=270,
    labelpad=20,
    fontsize=14
)
cbar.ax.tick_params(labelsize=11)

plt.title(
    rf"Optimal Noise Scale $\gamma$ ($\tau = {tau}$)",
    fontsize=16,
    pad=15
)

plt.xlabel("Leave probability q", fontsize=14)
plt.ylabel("Join probability r", fontsize=14)

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)

plt.tight_layout()

plt.savefig(
    f"optimal_gamma_heatmap_theory_{tau}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Optimization complete. "
    "Heatmap saved as 'optimal_gamma_heatmap.png'."
)
print("Text results saved as 'optimal_gammas.txt'.")
