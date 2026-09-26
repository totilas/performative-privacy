import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

def compute_beta_from_eps(eps):
    return 0.5 * np.tanh(eps / 2.0)

def apply_randomized_response(states, beta, gen):
    return np.where(gen.random(len(states)) < (0.5 + beta), states, 1 - states)

def estimate_unbiased_prevalence(responses, beta):
    return (np.mean(responses) - (0.5 - beta)) / (2 * beta)

def causal_moving_average(data, window):
    cum_sum = np.cumsum(data)
    res = np.zeros_like(data, dtype=float)
    for t in range(len(data)):
        if t < window:
            res[t] = cum_sum[t] / (t + 1)
        else:
            res[t] = (cum_sum[t] - cum_sum[t - window]) / window
    return res

def run_simulation(leave_rate, join_rate, eps_array, max_time_steps=200, 
                   initial_pop=400, num_trials=50, seed_val=42, min_pop_size=1):
    
    random_gen = np.random.default_rng(seed_val)
    beta_array = compute_beta_from_eps(eps_array)

    # Track full history to allow for confidence band calculations
    pop_history = np.zeros((num_trials, eps_array.size, max_time_steps))
    err_history = np.zeros((num_trials, eps_array.size, max_time_steps))

    for trial in range(num_trials):
        for idx_eps, beta_val in enumerate(beta_array):
            current_pop = int(initial_pop)

            for step in range(max_time_steps):
                actual_prevalence = random_gen.random()
                
                actual_states = (random_gen.random(current_pop) < actual_prevalence).astype(int)
                noisy_responses = apply_randomized_response(actual_states, beta_val, random_gen)

                prevalence_est = estimate_unbiased_prevalence(noisy_responses, beta_val)
                prevalence_est = np.clip(prevalence_est, 0.0, 1.0)
                
                err_history[trial, idx_eps, step] = prevalence_est - actual_prevalence
                pop_history[trial, idx_eps, step] = current_pop

                exposed_mask = (actual_states == noisy_responses)
                num_exposed = int(np.sum(exposed_mask))

                departures = random_gen.binomial(num_exposed, leave_rate) if num_exposed > 0 else 0
                arrivals = random_gen.binomial(current_pop - num_exposed, join_rate) if current_pop > 0 else 0

                current_pop = current_pop - departures + arrivals
                current_pop = min(max(min_pop_size, int(current_pop)), 20000)

    # Return the raw trial data instead of aggregating immediately 
    return pop_history, err_history

def main():
    parser = argparse.ArgumentParser(description="Simulate and plot Population vs Variance with Confidence Bands.")
    parser.add_argument("-q", "--leave_rate", type=float, required=True, help="Leave rate (e.g., 0.6)")
    parser.add_argument("-r", "--join_rate", type=float, required=True, help="Join rate (e.g., 0.3)")
    args = parser.parse_args()

    q = args.leave_rate
    r = args.join_rate

    eps_array = np.array([0.01, 0.1, 1.0, 5.0])
    num_trials = 50

    print(f"Running dual-axis simulation for q={q}, r={r}...")
    pop_history, err_history = run_simulation(leave_rate=q, join_rate=r, eps_array=eps_array, num_trials=num_trials)

    # Calculate Population percentiles
    mean_pop = np.mean(pop_history, axis=0)
    pop_lower = np.percentile(pop_history, 10, axis=0)
    pop_upper = np.percentile(pop_history, 90, axis=0)

    # Calculate Variance percentiles (Smoothed per trial)
    sq_err_history = err_history ** 2
    smoothed_sq_err = np.zeros_like(sq_err_history)
    
    for trial in range(num_trials):
        for idx_eps in range(len(eps_array)):
            smoothed_sq_err[trial, idx_eps] = causal_moving_average(sq_err_history[trial, idx_eps], window=10)

    mean_var = np.mean(smoothed_sq_err, axis=0)
    var_lower = np.percentile(smoothed_sq_err, 10, axis=0)
    var_upper = np.percentile(smoothed_sq_err, 90, axis=0)

    # Plotting setup
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx() 

    # Setup Viridis Colormap & Logarithmic Norm
    cmap = plt.cm.viridis
    norm = LogNorm(vmin=float(np.min(eps_array)), vmax=float(np.max(eps_array)))
    time_steps = np.arange(mean_pop.shape[1])

    for i, eps in enumerate(eps_array):
        color = cmap(norm(eps))
        
        # Left Axis: Population Line + Confidence Band
        ax1.plot(time_steps, mean_pop[i], color=color, linestyle='-', linewidth=2, alpha=0.9)
        ax1.fill_between(time_steps, pop_lower[i], pop_upper[i], color=color, alpha=0.15)
        
        # Right Axis: Variance Line + Confidence Band
        ax2.plot(time_steps, mean_var[i], color=color, linestyle='--', linewidth=2, alpha=0.8)
        ax2.fill_between(time_steps, var_lower[i], var_upper[i], color=color, alpha=0.10)

    # Axis Labels and Formatting
    ax1.set_xlabel(r"Time step $t$", fontsize=20)
    ax1.set_ylabel(r"$N_t$ (Averaged across trials)", fontsize=20)
    ax2.set_ylabel(r"Variance (Moving Averaged)", fontsize=20)
    
    ax2.set_yscale('log')
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Hide top borders
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Line Style Legend (Inside Plot)
    custom_lines = [
        Line2D([0], [0], color='black', lw=2, linestyle='-'),
        Line2D([0], [0], color='black', lw=2, linestyle='--')
    ]
    ax1.legend(custom_lines, ['Population', 'Variance'], loc='lower right', frameon=False, fontsize=22)

    # Epsilon Colorbar (Outside right axis)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, pad=0.15)
    cbar.set_label(r"$\varepsilon$ (log scale)", fontsize=20)

    output_file = f"pop_var_dual_q{q}_r{r}.pdf"
    fig.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Successfully saved plot to {output_file}")

if __name__ == "__main__":
    main()
