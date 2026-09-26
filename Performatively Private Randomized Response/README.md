# Privacy Epsilon Simulation & Variance Analysis

This folder contains scripts for simulating the optimal privacy budget ($\epsilon$) under randomized response mechanisms and tracking the resulting population dynamics and variance over time.

## 📁 Folder Structure & Scripts

### 1. Epsilon Landscape Heatmaps

* **`heatmap_generator.py`**: Executes a 30x30 grid search to find the optimal privacy budget ($\epsilon$) that minimizes estimation error across various join ($r$) and leave ($q$) probabilities.


* It tests two distinct population dynamic frameworks: an "original" model and an "agnostic" model.


* The simulation runs over a time horizon of $T=200$ steps with an initial population of $N=400$.


* It leverages Python's `ProcessPoolExecutor` to run calculations in parallel using all available CPU cores.


* **Output**: Generates PDF heatmaps of the optimal epsilon landscape for both models, saving them to a `plots/` directory as `original_model_heatmap.pdf` and `agnostic_model_heatmap.pdf`.





### 2. Dual-Axis Population & Variance Tracking

* **`population_variance_plot_generator.py`**: Simulates and plots the interconnected relationship between population size and estimation variance over time.


* It evaluates four discrete epsilon values: 0.01, 0.1, 1.0, and 5.0.


* The script computes the 10th and 90th percentiles across 50 trials to generate shaded confidence bands for the data.


* To smooth the variance curve, it applies a causal moving average with a window of 10 steps.


* **Output**: Produces a dual-axis PDF plot (`pop_var_dual_q{q}_r{r}.pdf`) that visualizes population (solid lines) on the left axis and variance (dashed lines) on the right axis, mapped by a logarithmic colormap based on $\epsilon$.





---

## 🚀 Execution Workflow

**1. Generate the Optimal Epsilon Heatmaps**
Run this script to launch the parallelized grid search across all leave and join probabilities.

```bash
python heatmap_generator.py

```

This will create the `plots/` directory (if it doesn't exist) and populate it with the two PDF heatmaps.

**2. Generate the Population vs. Variance Plot**
Run the variance tracking simulation by passing specific leave (`-q`) and join (`-r`) rates via the command line.

```bash
python population_variance_plot_generator.py -q 0.6 -r 0.3

```

This will simulate the dynamics for the chosen $q$ and $r$ values and output the dual-axis chart to `pop_var_dual_q0.6_r0.3.pdf` in the current directory.

## 📦 Dependencies

Both scripts require standard scientific Python libraries:

* `numpy`

* `matplotlib`
