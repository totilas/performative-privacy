import os
import glob
import pickle
import re
import math
import numpy as np

def extract_empirical_gammas(input_folder, target_tau=0.02, output_txt=None):
    """
    Reads all empirical pickle files, extracts the optimal gamma for each (q, r),
    returns a dictionary structured as data[q][r] = gamma, and optionally saves to text.
    """
    empirical_data = {}
    pkl_files = glob.glob(os.path.join(input_folder, "*.pkl"))
    
    if not pkl_files:
        print(f"Warning: No .pkl files found in '{input_folder}'.")
        return empirical_data

    print(f"Processing {len(pkl_files)} empirical .pkl files...")
    flat_results = []

    for filepath in pkl_files:
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # 1. Extract q and r (Top-level keys)
            if 'q' in data and 'r' in data:
                q, r = data['q'], data['r']
            else:
                # Fallback: parse q and r directly from the filename
                match = re.search(r'q_([\d.]+)_r_([\d.]+)', os.path.basename(filepath))
                if match:
                    q, r = float(match.group(1)), float(match.group(2))
                else:
                    print(f"Skipping {filepath}: Could not extract q and r.")
                    continue
            
            # 2. Extract the gammas array (Nested inside 'config')
            gammas = data['config']['gammas']
            
            # 3. Match the tau key safely using math.isclose to avoid float precision mismatch
            results_by_tau = data.get('results_by_tau', {})
            matching_tau_key = None
            
            for key in results_by_tau.keys():
                if math.isclose(key, target_tau, rel_tol=1e-5, abs_tol=1e-8):
                    matching_tau_key = key
                    break
            
            if matching_tau_key is not None:
                gamma_errors = results_by_tau[matching_tau_key]['gamma_errors']
                
                # Identify optimal empirical gamma ignoring NaNs
                opt_gamma_idx = np.nanargmin(gamma_errors)
                opt_gamma = gammas[opt_gamma_idx]
                
                if q not in empirical_data:
                    empirical_data[q] = {}
                empirical_data[q][r] = opt_gamma
                
                flat_results.append((q, r, opt_gamma))
            else:
                print(f"Skipping {filepath}: tau={target_tau} missing. Available taus: {list(results_by_tau.keys())}")
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    if output_txt and flat_results:
        flat_results.sort(key=lambda x: (x[0], x[1]))
        with open(output_txt, 'w') as f:
            for q, r, opt_gamma in flat_results:
                f.write(f"Optimal gamma for (q, r) = ({q:.2f}, {r:.2f}): {opt_gamma:.4f}\n")
        print(f"Saved empirical summaries to '{output_txt}'.")
        
    return empirical_data

# ============================================================
# Execution Block
# ============================================================
if __name__ == "__main__":
    # Define inputs and target tau
    input_directory = "results"
    tau = 0.02
    
    # Format the output filename using an f-string to inject the tau value
    output_file = f"optimal_gammas_empirical_{tau}.txt"
    
    # Run the extraction
    extracted_data = extract_empirical_gammas(
        input_folder=input_directory, 
        target_tau=tau, 
        output_txt=output_file
    )
