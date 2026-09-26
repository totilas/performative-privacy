import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

def simulate_final_step_experiment(
    epsilon,
    time_horizon_T,
    initial_participants_N,
    join_prob,
    leave_prob,
    model_type="original",
    trials=10,
    nmax=20000,
    rng_seed=None
):
    rng = np.random.default_rng(rng_seed)
    p = np.exp(epsilon) / (1.0 + np.exp(epsilon))
    final_errors = []
    N_track = []

    for _ in range(trials):
        N = int(initial_participants_N)

        for t in range(time_horizon_T - 1):
            if N <= 0:
                N = 0
                break

            leaked_count = rng.binomial(N, p)

            if model_type == "original":
                unleaked_count = N - leaked_count
                joining = rng.binomial(unleaked_count, join_prob)
            elif model_type == "agnostic":
                joining = rng.binomial(N, join_prob)
            else:
                raise ValueError("model_type must be either 'original' or 'agnostic'")

            leaving = rng.binomial(leaked_count, leave_prob)

            N = N + joining - leaving
            N = min(max(0, int(N)), nmax)

        if N <= 0:
            final_errors.append(10)
            N_track.append(1)
        else:
            mu_T = rng.uniform(0.5, 1.0)
            X = rng.binomial(1, mu_T, N)

            keep_mask = rng.binomial(1, p, N)
            Y = np.where(keep_mask == 1, X, 1 - X)

            nu_T = np.mean(Y)
            mu_hat = (nu_T - (1.0 - p)) / (2.0 * p - 1.0)
            mu_hat = np.clip(mu_hat, 0.0, 1.0)

            final_errors.append((mu_T - mu_hat)**2)
            N_track.append(N)

    return np.mean(final_errors), N_track

def _compute_best_epsilon_for_cell(i, q, j, r, epsilons, T, N_init, model_type):
    errors = []
    for eps in epsilons:
        avg_error, _ = simulate_final_step_experiment(
            epsilon=eps,
            time_horizon_T=T,
            initial_participants_N=N_init,
            join_prob=r,
            leave_prob=q,
            model_type=model_type,
            trials=10,
            nmax=20000
        )
        errors.append(avg_error)
        
    best_eps = epsilons[np.argmin(errors)]
    return i, j, best_eps

def generate_epsilon_heatmaps():
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    T = 200
    N_init = 400
    grid_size = 30
    r_s = np.linspace(0.01, 0.99, grid_size)
    q_s = np.linspace(0.01, 0.99, grid_size)
    epsilons = np.geomspace(0.01, 20, 30)

    models = ["original", "agnostic"]
    max_workers = os.cpu_count() 

    for current_model in models:
        print(f"Running simulation for the {current_model.capitalize()} Model using {max_workers} workers...")
        best_eps_matrix = np.zeros((len(q_s), len(r_s)))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, q in enumerate(q_s):
                for j, r in enumerate(r_s):
                    futures.append(
                        executor.submit(
                            _compute_best_epsilon_for_cell,
                            i, q, j, r, epsilons, T, N_init, current_model
                        )
                    )
            
            for future in as_completed(futures):
                i, j, best_eps = future.result()
                best_eps_matrix[i, j] = best_eps

        # --- Plotting the Heatmap ---
        plt.figure(figsize=(10, 8))
        log_eps_matrix = np.log10(best_eps_matrix)

        # CHANGED: Swapped q_s and r_s, and transposed the matrix with .T
        mesh = plt.pcolormesh(
            q_s,
            r_s,
            log_eps_matrix.T,
            shading='auto',
            cmap='viridis',
            vmin=-2.0,  
            vmax=0.7,
            rasterized=True  
        )

        cbar = plt.colorbar(mesh)
        cbar.set_label(r"$\log_{10}(\epsilon^*)$", fontsize=22)
        cbar.ax.tick_params(labelsize=12)

        model_title = "Original Model" if current_model == "original" else "Agnostic Model"
        #plt.title(f"{model_title} Optimal Epsilon Landscape", fontsize=18, pad=15)
        
        # CHANGED: Swapped the axis labels
        plt.xlabel(r"Leaving Rate ($q$)", fontsize=22)
        plt.ylabel(r"Joining Rate ($r$)", fontsize=22)
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        file_path = os.path.join(output_dir, f"{current_model}_model_heatmap.pdf")
        plt.tight_layout()
        plt.savefig(file_path, format='pdf', dpi=300)
        plt.close()

        print(f" -> Saved: {file_path}")

if __name__ == "__main__":
    generate_epsilon_heatmaps()
