import os
import pandas as pd
import time
import concurrent.futures
import multiprocessing
import json
from SettlementModel import SettlementModel

# ========================================================================
# CONFIGURATION - Toggle this flag to control OCEL log saving
# ========================================================================
# Set to False to save OCEL logs (slower, but complete)
# Set to True to skip OCEL logs (faster runs for visualization only)
SKIP_OCEL_LOGS = True  # <-- TOGGLE THIS VALUE


# ========================================================================

def batch_runner_parallel():
    """
    Parallel implementation of the batch runner that uses multiple processes
    to run different simulation configurations simultaneously.
    """
    # Create new folders for parallel results to avoid overwriting original data
    log_folder = "parallel_logs"
    depth_folder = "parallel_depth_statistics"

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)

    num_institutions = 10  # Number of institutions in the simulation
    runs_per_config = 7  # Number of simulations per configuration
    # Use seeds for reproducibility and comparison
    base_seed = 42
    seed_list = [base_seed + i for i in range(runs_per_config)]

    # Determine optimal number of workers (leave one core for system processes)
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Running with {max_workers} parallel workers")
    print(f"OCEL logging is {'DISABLED' if SKIP_OCEL_LOGS else 'ENABLED'}")

    # Results collection
    all_results = []
    total_configs = num_institutions * runs_per_config
    completed = 0

    # Run all configurations in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create all simulation configurations
        futures = []

        for true_count in range(1, num_institutions + 1):
            # Create a tuple: the first 'true_count' positions are True, the rest are False
            partialsallowed = tuple([True] * true_count + [False] * (num_institutions - true_count))
            print(f"Preparing simulation configuration: {partialsallowed}")

            for run in range(1, runs_per_config + 1):
                seed = seed_list[run - 1]

                # Submit each configuration to the process pool
                future = executor.submit(
                    run_simulation,
                    partialsallowed=partialsallowed,
                    true_count=true_count,
                    run=run,
                    seed=seed,
                    log_folder=log_folder,
                    depth_folder=depth_folder
                )
                futures.append(future)

        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)

                # Update progress
                completed += 1
                print(
                    f"Progress: {completed}/{total_configs} simulations completed ({completed / total_configs * 100:.1f}%)")
            except Exception as e:
                print(f"ERROR in simulation: {str(e)}")
                # Continue with other simulations

    # Process and save results
    print("All simulations completed. Saving results...")

    # Create results folder if it doesn't exist
    results_folder = "parallel_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results_file = os.path.join(results_folder, "Parallel_measurement.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    return all_results


def run_simulation(partialsallowed, true_count, run, seed, log_folder, depth_folder):
    """
    Run a single simulation with the given configuration parameters.
    Returns the efficiency metrics and other results.
    """
    start_time = time.time()

    print(f"Starting simulation: Configuration with {true_count} True, run {run}, seed {seed}")

    try:
        # Create the simulation model
        model = SettlementModel(partialsallowed=partialsallowed, seed=seed)

        # Run the simulation until completion
        while model.simulated_time < model.simulation_end:
            model.step()

        # Set up filenames for logs
        ocel_filename = os.path.join(log_folder, f"simulation_config{true_count}_run{run}.jsonocel")
        depth_filename = os.path.join(depth_folder, f"depth_statistics_config{true_count}_run{run}.json")

        # Ensure depth statistics directory exists
        os.makedirs(os.path.dirname(depth_filename), exist_ok=True)

        # Only save OCEL logs if not skipped (controlled by the global flag)
        if not SKIP_OCEL_LOGS:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(ocel_filename), exist_ok=True)
            # Save the OCEL event log
            model.save_ocel_log(filename=ocel_filename)
            print(f"OCEL logs saved for configuration {true_count} run {run}")

        # Always save depth statistics (needed for visualizations)
        stats = model.generate_depth_statistics()
        with open(depth_filename, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Depth statistics saved for configuration {true_count} run {run}")

        # Calculate efficiency metrics (using optimized version if available)
        if hasattr(model, 'calculate_settlement_efficiency_optimized'):
            inst_eff, val_eff = model.calculate_settlement_efficiency_optimized()
        else:
            inst_eff, val_eff = model.calculate_settlement_efficiency()

        settled_count = model.count_settled_instructions()
        total_settled_amount = model.get_total_settled_amount()

        # Calculate runtime
        runtime = time.time() - start_time

        # Return the results
        return {
            'Partial': str(partialsallowed),
            'instruction_efficiency': inst_eff,
            'value_efficiency': val_eff,
            'settled_count': settled_count,
            'settled_amount': total_settled_amount,
            'seed': seed,
            'runtime_seconds': runtime,
            'config_num': true_count,
            'run_num': run
        }

    except RecursionError:
        print(
            f"RecursionError occurred in config{true_count}_run{run}: maximum recursion depth exceeded. Simulation terminated.")

        # Create error log folder
        error_folder = "parallel_error_logs"
        if not os.path.exists(error_folder):
            os.makedirs(error_folder)

        # Save error details to file
        error_file = os.path.join(error_folder, f"error_config{true_count}_run{run}.txt")
        with open(error_file, 'w') as f:
            f.write(f"RecursionError in configuration with {true_count} True institutions, run {run}, seed {seed}.\n")
            f.write("Maximum recursion depth exceeded.")

        # Record error but return partial results if possible
        return {
            'Partial': str(partialsallowed),
            'error': 'RecursionError',
            'seed': seed,
            'config_num': true_count,
            'run_num': run,
            'runtime_seconds': time.time() - start_time
        }
    except Exception as e:
        print(f"Error in config{true_count}_run{run}: {str(e)}. Simulation terminated.")

        # Create error log folder
        error_folder = "parallel_error_logs"
        if not os.path.exists(error_folder):
            os.makedirs(error_folder)

        # Save error details to file
        error_file = os.path.join(error_folder, f"error_config{true_count}_run{run}.txt")
        with open(error_file, 'w') as f:
            f.write(f"Error in configuration with {true_count} True institutions, run {run}, seed {seed}.\n")
            f.write(f"Error message: {str(e)}")

        # Record general errors
        return {
            'Partial': str(partialsallowed),
            'error': str(e),
            'seed': seed,
            'config_num': true_count,
            'run_num': run,
            'runtime_seconds': time.time() - start_time
        }


# Main entry point
if __name__ == "__main__":
    print("Starting parallel batch runner...")
    start_total = time.time()

    results = batch_runner_parallel()

    total_time = time.time() - start_total
    print(f"Parallel batch runner completed in {total_time:.2f} seconds")