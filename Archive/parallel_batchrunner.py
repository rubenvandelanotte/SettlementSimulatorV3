import os
import pandas as pd
import time
import concurrent.futures
import multiprocessing
import json
import gc
import sys
from SettlementModel import SettlementModel
from RuntimeTracker import RuntimeTracker


def deep_cleanup():
    """Perform basic garbage collection between simulation runs"""
    # Run garbage collection
    gc.collect()
    print("Garbage collection completed")


def run_simulation(config):
    """
    Run a single simulation with the given configuration parameters.
    Returns the efficiency metrics and other results.

    Args:
        config: Dictionary containing simulation parameters
    """
    # Extract configuration parameters
    partialsallowed = config["partialsallowed"]
    true_count = config["true_count"]
    run = config["run"]
    seed = config["seed"]
    log_folder = config["log_folder"]
    depth_folder = config["depth_folder"]

    print(f"Starting simulation: Configuration with {true_count} True, run {run}, seed {seed}")

    try:
        # Create the simulation model
        model = SettlementModel(partialsallowed=partialsallowed, seed=seed)

        # Simulate until the simulation is past the set end time
        while model.simulated_time < model.simulation_end:
            model.step()

        # Set filenames with configuration and run number
        ocel_filename = os.path.join(log_folder, f"simulation_config{true_count}_run{run}.jsonocel")
        depth_filename = os.path.join(depth_folder, f"depth_statistics_config{true_count}_run{run}.json")

        # Save depth statistics first (smaller data)
        try:
            stats = model.generate_depth_statistics()
            with open(depth_filename, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Depth statistics saved to {depth_filename}")
        except Exception as e:
            print(f"Error saving depth statistics for Config{true_count}_Run{run}: {str(e)}")

        # Calculate settlement efficiency with optimized method if available
        try:
            print(f"Calculating settlement efficiency for Config{true_count}_Run{run}")

            if hasattr(model, 'calculate_settlement_efficiency_optimized'):
                new_ins_eff, new_val_eff = model.calculate_settlement_efficiency_optimized()
            else:
                new_ins_eff, new_val_eff = model.calculate_settlement_efficiency()

            settled_count = model.count_settled_instructions()
            total_settled_amount = model.get_total_settled_amount()
        except Exception as e:
            print(f"Error calculating efficiency for Config{true_count}_Run{run}: {str(e)}")
            new_ins_eff, new_val_eff = 0, 0
            settled_count, total_settled_amount = 0, 0

        # Try to save OCEL logs
        try:
            print(f"Saving OCEL logs for Config{true_count}_Run{run}...")
            model.save_ocel_log(filename=ocel_filename)
            print(f"OCEL logs saved to {ocel_filename}")

            # Reset logger state
            if hasattr(model, 'logger') and hasattr(model.logger, 'reset'):
                print("Resetting logger state...")
                model.logger.reset()

        except Exception as e:
            print(f"Error saving full OCEL logs for Config{true_count}_Run{run}: {str(e)}")
            print(f"Attempting to save simplified logs...")

            try:
                # Try to save just the most important events to a backup file
                simplified_filename = os.path.join(log_folder, f"simplified_config{true_count}_run{run}.json")

                # Gather essential data
                essential_data = {
                    "configuration": {
                        "partialsallowed": str(partialsallowed),
                        "seed": seed,
                        "true_count": true_count,
                        "run": run
                    },
                    "results": {
                        "instruction_efficiency": new_ins_eff,
                        "value_efficiency": new_val_eff,
                        "settled_count": settled_count,
                        "settled_amount": total_settled_amount
                    },
                    "timestamp": time.time()
                }

                # Save this simplified data
                with open(simplified_filename, 'w') as f:
                    json.dump(essential_data, f, indent=2, default=str)

                print(f"Saved simplified logs to {simplified_filename}")
            except Exception as backup_error:
                print(f"Error saving simplified logs for Config{true_count}_Run{run}: {str(backup_error)}")
                print(f"Continuing without saving logs")

        # Explicit cleanup of model structures
        print(f"Explicitly cleaning model data structures for Config{true_count}_Run{run}")
        if hasattr(model, 'instructions'):
            model.instructions.clear()
        if hasattr(model, 'transactions'):
            model.transactions.clear()
        if hasattr(model, 'institutions'):
            model.institutions.clear()
        if hasattr(model, 'accounts'):
            model.accounts.clear()
        if hasattr(model, 'validated_delivery_instructions'):
            model.validated_delivery_instructions.clear()
        if hasattr(model, 'validated_receipt_instructions'):
            model.validated_receipt_instructions.clear()
        if hasattr(model, 'agents'):
            model.agents.clear()
        if hasattr(model, 'event_log'):
            model.event_log.clear()

        # Basic garbage collection
        gc.collect()

        # Return the results
        return {
            "Partial": str(partialsallowed),
            "instruction efficiency": new_ins_eff,
            "value efficiency": new_val_eff,
            "settled_count": settled_count,
            "settled_amount": total_settled_amount,
            "seed": seed,
            "runtime_seconds": time.time() - config["start_time"],
            "error": None
        }

    except RecursionError:
        print(
            f"RecursionError occurred in Config{true_count}_Run{run}: maximum recursion depth exceeded. Simulation terminated.")
        return {
            "Partial": str(partialsallowed),
            "instruction efficiency": 0,
            "value efficiency": 0,
            "settled_count": 0,
            "settled_amount": 0,
            "seed": seed,
            "runtime_seconds": time.time() - config["start_time"],
            "error": "RecursionError"
        }
    except MemoryError:
        print(f"MemoryError occurred in Config{true_count}_Run{run}: not enough memory. Simulation terminated.")
        return {
            "Partial": str(partialsallowed),
            "instruction efficiency": 0,
            "value efficiency": 0,
            "settled_count": 0,
            "settled_amount": 0,
            "seed": seed,
            "runtime_seconds": time.time() - config["start_time"],
            "error": "MemoryError"
        }
    except Exception as e:
        print(f"Error in Config{true_count}_Run{run}: {str(e)}. Simulation terminated.")
        return {
            "Partial": str(partialsallowed),
            "instruction efficiency": 0,
            "value efficiency": 0,
            "settled_count": 0,
            "settled_amount": 0,
            "seed": seed,
            "runtime_seconds": time.time() - config["start_time"],
            "error": str(e)
        }


def batch_runner_parallel():
    """
    Run a batch of simulations in parallel with different configurations.

    This function runs simulations for different combinations of parameters
    and tracks efficiency metrics and resource usage.

    Returns:
        list: List of dictionaries containing results for each simulation run
    """
    print("Starting parallel batch runner...")

    # Create a runtime tracker
    tracker = RuntimeTracker("parallel_runtime_results.json")

    # Create folders for logs if they don't exist
    log_folder = "parallel_logs"
    depth_folder = "parallel_depth_statistics"

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)

    num_institutions = 10  # Number of institutions in the simulation
    runs_per_config = 10  # Number of simulations per configuration
    # Use seeds to compare
    base_seed = 42
    seed_list = [base_seed + i for i in range(runs_per_config)]

    efficiencies = []

    # Determine optimal number of workers (leave one core for system processes)
    max_workers = get_optimal_workers()
    print(f"Running with {max_workers} parallel workers")

    # Create all simulation configurations
    configs = []
    for true_count in range(1, num_institutions + 1):
        # Create a tuple: the first 'true_count' positions are True, the rest are False
        partialsallowed = tuple([True] * true_count + [False] * (num_institutions - true_count))
        print(f"Preparing simulation configuration: {partialsallowed}")

        seed_index = 0
        for run in range(1, runs_per_config + 1):
            seed = seed_list[seed_index]
            seed_index += 1

            # Create configuration dictionary
            config = {
                "partialsallowed": partialsallowed,
                "true_count": true_count,
                "run": run,
                "seed": seed,
                "log_folder": log_folder,
                "depth_folder": depth_folder,
                "start_time": time.time()  # Track start time for each configuration
            }

            configs.append(config)

    # Run simulations in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all simulation configurations
        future_to_config = {executor.submit(run_simulation, config): config for config in configs}

        # Process results as they complete
        total_configs = len(configs)
        completed = 0

        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            true_count = config["true_count"]
            run = config["run"]

            try:
                # Get simulation results
                sim_results = future.result()
                efficiencies.append(sim_results)

                # Update progress
                completed += 1
                print(
                    f"Progress: {completed}/{total_configs} simulations completed ({completed / total_configs * 100:.1f}%)")

                # Basic cleanup
                deep_cleanup()

                # Save incremental results
                df_incremental = pd.DataFrame(efficiencies)
                df_incremental.to_csv("Parallel_Incremental_measurement.csv", index=False)

            except Exception as e:
                print(f"ERROR in Config{true_count}_Run{run}: {str(e)}")
                # Add error result to track it
                error_result = {
                    "Partial": str(config["partialsallowed"]),
                    "instruction efficiency": 0,
                    "value efficiency": 0,
                    "settled_count": 0,
                    "settled_amount": 0,
                    "seed": config["seed"],
                    "runtime_seconds": time.time() - config["start_time"],
                    "error": str(e)
                }
                efficiencies.append(error_result)

    # Save all runtime results
    print("All simulations completed. Saving final results...")
    df = pd.DataFrame(efficiencies)
    df.to_csv("Parallel_measurement.csv", index=False)

    return efficiencies


# Determine number of workers for parallel execution
def get_optimal_workers():
    try:
        # First check if an environment variable is set (for cloud environments)
        if "MAX_WORKERS" in os.environ:
            workers = int(os.environ.get("MAX_WORKERS"))
            print(f"Using environment-specified worker count: {workers}")
            return workers

        # Try to detect CPU count
        cpu_count = multiprocessing.cpu_count()
        # Use all CPUs minus one for system tasks
        workers = max(1, cpu_count - 1)
        print(f"Detected {cpu_count} CPUs, using {workers} workers")
        return workers
    except Exception as e:
        # Fallback to a conservative default if detection fails
        print(f"Error detecting CPU count: {e}. Using default worker count of 4")
        return 4

# Starting the parallel batchrunner
if __name__ == "__main__":
    new_measured_efficiency = []  # Initialize this outside the try block
    start_total = time.time()

    try:
        new_measured_efficiency = batch_runner_parallel()
        print(new_measured_efficiency)
        df = pd.DataFrame(new_measured_efficiency)
        df.to_csv("Parallel_measurement.csv", index=False)

        total_time = time.time() - start_total
        print(f"Parallel batch runner completed in {total_time:.2f} seconds")

    except Exception as e:
        print(f"Fatal error in parallel batch runner: {str(e)}")
        # Try to save any results we have
        if new_measured_efficiency:  # Check if we have any results
            df = pd.DataFrame(new_measured_efficiency)
            df.to_csv("Emergency_parallel_results.csv", index=False)
            print(f"Saved partial results to Emergency_parallel_results.csv")