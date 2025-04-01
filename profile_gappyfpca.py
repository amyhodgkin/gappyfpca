import cProfile
import pstats
import time

import numpy as np

# Assuming your package structure allows these imports from the root directory
from gappyfpca.fpca import gappyfpca, reconstruct_func


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

    return data, functions


def run_gappyfpca_profile(data, iparallel_mode=1):
    """Runs the gappyfpca function for profiling."""
    print(f"\nRunning gappyfpca with iparallel={iparallel_mode}...")
    start_time = time.time()
    fpca_comps, fpca_coefs, evalue, run_stat = gappyfpca(
        data, var_rat=0.99, max_iter=15, num_iter=5, iparallel=iparallel_mode
    )
    end_time = time.time()
    print(f"gappyfpca execution time: {end_time - start_time:.4f} seconds")
    # Optional: reconstruct to ensure it runs fully
    # function_recon = reconstruct_func(fpca_comps[0, :], fpca_comps[1:, :], fpca_coefs)


if __name__ == "__main__":
    print("Generating synthetic data...")
    gappy_data, original_functions = generate_test_data(M=1000, L=50, nan_fraction=0.4)
    print(f"Data shape: {gappy_data.shape}")

    # --- Profile Serial Execution ---
    print("\n--- Profiling Serial Execution (iparallel=0) ---")
    profiler_serial = cProfile.Profile()
    profiler_serial.enable()
    run_gappyfpca_profile(gappy_data, iparallel_mode=0)
    profiler_serial.disable()

    print("\nSerial Profiling Results (Top 15 by cumulative time):")
    stats_serial = pstats.Stats(profiler_serial).sort_stats("cumulative")
    stats_serial.strip_dirs()
    stats_serial.print_stats(15)

    # --- Profile Parallel Execution ---
    print("\n--- Profiling Parallel Execution (iparallel=1) ---")
    profiler_parallel = cProfile.Profile()
    profiler_parallel.enable()
    run_gappyfpca_profile(gappy_data, iparallel_mode=1)
    profiler_parallel.disable()

    print("\nParallel Profiling Results (Top 15 by cumulative time):")
    stats_parallel = pstats.Stats(profiler_parallel).sort_stats("cumulative")
    stats_parallel.strip_dirs()
    stats_parallel.print_stats(15)

    # Optional: Save stats to files for more detailed analysis
    # stats_serial.dump_stats("gappyfpca_serial.prof")
    # stats_parallel.dump_stats("gappyfpca_parallel.prof")
    # print("\nProfiling data saved to gappyfpca_serial.prof and gappyfpca_parallel.prof")