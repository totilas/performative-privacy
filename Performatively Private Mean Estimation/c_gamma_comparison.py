import os
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ==========================================================
# Fixed parameters
# ==========================================================

d = 12
N_t_1 = 1000
R = 10.0
num_samples = 200000

# Main directory containing:
#   results_tau_0.02/
#   results_tau_0.05/
# Main directory containing the two folders
RESULT_DIR = "results"

# Folder to save figures
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================================
# Monte Carlo samples for theoretical curve
# ==========================================================

print("Generating Monte Carlo samples for theory...")

Z = np.random.randn(num_samples, d)
Z = np.clip(Z, -R, R)

z_norms = np.linalg.norm(Z, axis=1)
z_norms = np.maximum(z_norms, 1e-8)


# ==========================================================
# Theory
# ==========================================================

def compute_p_tau(tau, gamma):

    A = np.sqrt(N_t_1) * gamma

    term1 = (tau * A) / z_norms
    term2 = z_norms / (2 * A)

    return np.mean(
        1 - norm.cdf(term1 - term2)
    )


def compute_C_theory(gammas, tau, q, r):

    return np.array([
        1 + r - (q + r) * compute_p_tau(tau, g)
        for g in gammas
    ])


# ==========================================================
# Tau folders
# ==========================================================

TAU_FOLDERS = {
    0.02: "results_tau_0.02",
    0.05: "results_tau_0.05"
}


# ==========================================================
# Gamma values
# ==========================================================

gamma_values = np.geomspace(
    0.2,
    20,
    30
)


# ==========================================================
# Read data from both tau folders
# ==========================================================

data_by_tau = {}

for tau, folder_name in TAU_FOLDERS.items():

    input_dir = os.path.join(
        RESULT_DIR,
        folder_name
    )

    print("\n" + "=" * 70)
    print(f"Reading tau = {tau}")
    print(f"Directory: {input_dir}")
    print("=" * 70)

    if not os.path.isdir(input_dir):
        print(
            f"WARNING: Directory not found: {input_dir}"
        )
        continue

    files = sorted(
        glob.glob(
            os.path.join(
                input_dir,
                "*.pkl"
            )
        )
    )

    print(
        f"Found {len(files)} pickle files."
    )

    data_by_tau[tau] = []

    for file in files:

        with open(file, "rb") as f:
            results = pickle.load(f)

        q = results["q"]
        r = results["r"]

        results_by_tau = results["results_by_tau"]

        # Find the tau key closest to the folder tau
        available_taus = list(
            results_by_tau.keys()
        )

        matching_tau = min(
            available_taus,
            key=lambda x: abs(float(x) - tau)
        )

        if abs(
            float(matching_tau) - tau
        ) > 1e-6:

            print(
                f"WARNING: tau={tau} not found in {file}"
            )

            continue

        tau_results = results_by_tau[
            matching_tau
        ]

        empirical_C = np.asarray(
            tau_results[
                "empirical_C_gammas"
            ]
        )

        empirical_std = np.asarray(
            tau_results[
                "empirical_C_std_gammas"
            ]
        )

        theory_C = compute_C_theory(
            gamma_values,
            tau,
            q,
            r
        )

        data_by_tau[tau].append({
            "q": q,
            "r": r,
            "empirical_C": empirical_C,
            "empirical_std": empirical_std,
            "theory_C": theory_C
        })


# ==========================================================
# Identify all (q,r) pairs
# ==========================================================

qr_pairs = set()

for tau in data_by_tau:

    for data in data_by_tau[tau]:

        q = data["q"]
        r = data["r"]

        qr_pairs.add(
            (q, r)
        )


qr_pairs = sorted(qr_pairs)

print("\n" + "=" * 70)
print(f"Found {len(qr_pairs)} unique (q,r) pairs.")
print("=" * 70)


# ==========================================================
# Colors for the two tau values
# ==========================================================

tau_colors = {
    0.02: "dodgerblue",
    0.05: "darkorange"
}


# ==========================================================
# Create ONE plot for EACH (q,r) pair
# ==========================================================

for q, r in qr_pairs:

    print(
        f"\nCreating plot for q={q:.2f}, r={r:.2f}"
    )


    # ------------------------------------------------------
    # Create figure
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )


    # ======================================================
    # Plot both tau values
    # ======================================================

    for tau in [0.02, 0.05]:

        # Find matching data for this q,r pair
        matching_data = [
            data
            for data in data_by_tau.get(tau, [])
            if np.isclose(data["q"], q)
            and np.isclose(data["r"], r)
        ]

        if len(matching_data) == 0:

            print(
                f"WARNING: No data found for "
                f"q={q}, r={r}, tau={tau}"
            )

            continue


        # There should be exactly one match
        data = matching_data[0]

        empirical_C = data["empirical_C"]
        empirical_std = data["empirical_std"]
        theory_C = data["theory_C"]

        color = tau_colors[tau]


        # --------------------------------------------------
        # Theory
        # --------------------------------------------------

        ax.plot(
            gamma_values,
            theory_C,
            color=color,
            linewidth=3,
            linestyle="-",
            label=rf"Theory, $\tau={tau:.2f}$"
        )


        # --------------------------------------------------
        # Empirical
        # --------------------------------------------------

        ax.errorbar(
            gamma_values,
            empirical_C,
            yerr=empirical_std,
            fmt="o--",
            color=color,
            linewidth=2.2,
            markersize=5,
            capsize=4,
            label=rf"Empirical, $\tau={tau:.2f}$"
        )


        # --------------------------------------------------
        # Shaded ±1 sigma band
        # --------------------------------------------------

        ax.fill_between(
            gamma_values,
            empirical_C - empirical_std,
            empirical_C + empirical_std,
            color=color,
            alpha=0.12
        )


    # ======================================================
    # Bounds
    # ======================================================

    ax.axhline(
        1 - q,
        color="firebrick",
        linestyle="-",
        linewidth=2.5,
        label=rf"$1-q={1-q:.2f}$"
    )


    ax.axhline(
        1 + r,
        color="forestgreen",
        linestyle="-",
        linewidth=2.5,
        label=rf"$1+r={1+r:.2f}$"
    )


    # ======================================================
    # Formatting
    # ======================================================

    ax.set_xscale("log")


    ax.set_xlabel(
        r"Noise Scale $\gamma$",
        fontsize=35
    )


    ax.set_ylabel(
        r"$C_T(\gamma)$",
        fontsize=35
    )


    #ax.set_title(
      #  rf"$q={q:.2f}$, $r={r:.2f}$",
       # fontsize=22,
        #pad=15
    #)


    ax.tick_params(
        axis="both",
        which="major",
        labelsize=18
    )


    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )


    # ======================================================
    # Large readable legend
    # ======================================================

    ax.legend(
    fontsize=17,
    loc="best",
    frameon=True,
    framealpha=0.95,
    borderpad=1.4,
    labelspacing=1.2,
    handlelength=4,
    handletextpad=1.0,
    markerscale=1.5,
    ncol=2,  # number of columns
)


    # ======================================================
    # Layout
    # ======================================================

    fig.tight_layout()


    # ======================================================
    # Save
    # ======================================================

    outfile = os.path.join(
        SAVE_DIR,
        f"comparison_q_{q:.2f}_r_{r:.2f}.png"
    )


    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(fig)


    print(
        f"Saved: {outfile}"
    )


# ==========================================================
# Finished
# ==========================================================

print(
    f"\nAll figures saved to '{SAVE_DIR}'."
)
