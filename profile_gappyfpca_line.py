import cProfile
import pstats
import time

import numpy as np

# Assuming your package structure allows these imports from the root directory
from gappyfpca.fpca import gappyfpca, reconstruct_func
from gappyfpca.data_check import check_gappiness

def generate_test_data(M=1000, L=50, nan_fraction=0.3):
    """Generates synthetic gappy data similar to test_gappyfpca_integration."""
    np.random.seed(42)  # Ensure reproducibility
    x = np.linspace(0, 2 * np.pi, L)
    functions = np.array(
        [
            10
            + np.random.uniform(0.1, 5)
            * np.sin(
                x * np.random.uniform(1, 1.5)
                + np.random.uniform(0, np.pi / 2)
            )
            for _ in range(M)
        ]
    )

    data = np.copy(functions)
    # Artificially make it gappy
    for i in range(M):
        num_nans = int(L * nan_fraction * np.random.rand()) # Variable number of NaNs per row
        if num_nans > 0:
            nan_indices = np.random.choice(L, num_nans, replace=False)
            data[i, nan_indices] = np.nan

    # Check data validity before running gappyfpca
    try:
        check_gappiness(data)
        print("Generated data passed gappiness check.")
    except ValueError as e:
        print(f"Warning: Generated data failed gappiness check: {e}")
        # Depending on the error, you might want to regenerate or stop
        # For profiling, we might proceed cautiously or adjust generation

    return data, functions


def run_gappyfpca_profile(data, iparallel_mode=1):
    """Runs the gappyfpca function for profiling."""
    print(f"\nRunning gappyfpca with iparallel={iparallel_mode}...")
    start_time = time.time()
    # The functions decorated with @profile will be timed line-by-line by kernprof
    fpca_comps, fpca_coefs, evalue, run_stat = gappyfpca(
        data, var_rat=0.99, max_iter=15, num_iter=5, iparallel=iparallel_mode
    )
    end_time = time.time()
    print(f"gappyfpca execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    print("Generating synthetic data...")
    gappy_data, original_functions = generate_test_data(M=1000, L=50, nan_fraction=0.4)
    print(f"Data shape: {gappy_data.shape}")

    # Run the function(s) you want to profile
    # Run serial first to understand baseline
    print("\n--- Running Serial Execution (iparallel=0) ---")
    run_gappyfpca_profile(gappy_data, iparallel_mode=0)

    # Optionally run parallel (line_profiler works best on serial code)
    # print("\n--- Running Parallel Execution (iparallel=1) ---")
    # run_gappyfpca_profile(gappy_data, iparallel_mode=1)

    print("\nProfiling complete. Check kernprof output.")