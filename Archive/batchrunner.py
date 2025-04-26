import os
import pandas as pd
import time
import psutil  # For memory monitoring
import json
from SettlementModel import SettlementModel
from RuntimeTracker import RuntimeTracker
import gc
import sys


def get_memory_usage():
    """
    Get current memory usage of the process in MB.

    Returns:
        float: Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    return mem_usage


def log_memory_usage(label):
    """
    Log current memory usage with a label.

    Args:
        label (str): Label to identify this memory measurement
    """
    mem_usage = get_memory_usage()
    print(f"Memory usage [{label}]: {mem_usage:.2f} MB")
    return mem_usage


def deep_cleanup():
    """Perform aggressive memory cleanup between simulation runs"""
    # Run multiple garbage collection cycles with the highest generation
    for i in range(3):
        collected = gc.collect(2)
        print(f"GC run {i + 1}: collected {collected} objects")

    # On Windows, try to release memory back to the OS
    if sys.platform.startswith('win'):
        try:
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
            print("Released memory back to OS")
        except Exception as e:
            print(f"Error releasing memory: {e}")


# Call this function after each simulation run completes


def batch_runner():
    """
    Run a batch of simulations with different configurations.

    This function runs simulations for different combinations of parameters
    and tracks efficiency metrics and resource usage.

    Returns:
        list: List of dictionaries containing results for each simulation run
    """
    # Create a runtime tracker
    tracker = RuntimeTracker("runtime_results.json")

    # Create folders for logs if they don't exist
    log_folder = "simulatie_logs"
    depth_folder = "depth_statistics"

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

    # For each configuration: add one more institution with allowPartial = True
    for true_count in range(1, num_institutions + 1):
        # Create a tuple: the first 'true_count' positions are True, the rest are False
        partialsallowed = tuple([True] * true_count + [False] * (num_institutions - true_count))
        print(f"Simulation configuration: {partialsallowed}")

        seed_index = 0

        for run in range(1, runs_per_config + 1):
            print(f"Starting simulation: Configuration with {true_count} True, run {run}")
            seed = seed_list[seed_index]
            seed_index += 1

            # Define a run_simulation function that will be timed
            def run_simulation(config):
                # Extract the configuration parameters
                partialsallowed = config["partialsallowed"]
                seed = config["seed"]

                # Monitor memory before starting
                mem_before = log_memory_usage("before simulation")

                # Create and run the model
                model = SettlementModel(partialsallowed=partialsallowed, seed=seed)

                try:
                    # Simulate until the simulation is past the set end time
                    step_count = 0
                    while model.simulated_time < model.simulation_end:
                        model.step()
                        step_count += 1

                        # Periodically trigger garbage collection and check memory
                        if step_count % 50 == 0:
                            gc.collect()
                            mem_current = log_memory_usage(f"step {step_count}")

                            # If memory usage is getting too high, take preventive action
                            if mem_current > 3000:  # 3 GB threshold, adjust as needed
                                print("WARNING: High memory usage detected. Performing cleanup...")
                                # Try to reduce memory usage by emptying the event log if possible
                                try:
                                    if hasattr(model.logger, 'events') and len(model.logger.events) > 0:
                                        print(f"Clearing {len(model.logger.events)} events from logger")
                                        # Optionally save events to a temporary file before clearing
                                        temp_file = f"temp_events_{true_count}_{run}_{step_count}.json"
                                        with open(temp_file, 'w') as f:
                                            import json
                                            json.dump(model.logger.events, f)
                                        print(f"Saved events to {temp_file}")
                                        model.logger.events = []
                                except Exception as cleanup_error:
                                    print(f"Error during cleanup: {cleanup_error}")

                                # Force garbage collection
                                gc.collect()

                except RecursionError:
                    print("RecursionError occurred: maximum recursion depth exceeded. Simulation terminated.")
                    return {
                        "instruction_efficiency": 0,
                        "value_efficiency": 0,
                        "settled_count": 0,
                        "settled_amount": 0,
                        "error": "RecursionError"
                    }
                except MemoryError:
                    print("MemoryError occurred: not enough memory. Simulation terminated.")
                    return {
                        "instruction_efficiency": 0,
                        "value_efficiency": 0,
                        "settled_count": 0,
                        "settled_amount": 0,
                        "error": "MemoryError"
                    }

                # Set filenames with configuration and run number
                ocel_filename = os.path.join(log_folder, f"simulation_config{true_count}_run{run}.jsonocel")
                depth_filename = os.path.join(depth_folder, f"depth_statistics_config{true_count}_run{run}.json")

                # Monitor memory before saving logs
                mem_after_sim = log_memory_usage("after simulation")

                # Save depth statistics first (smaller data)
                try:
                    stats = model.generate_depth_statistics()
                    import json
                    with open(depth_filename, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"Depth statistics saved to {depth_filename}")
                except Exception as e:
                    print(f"Error saving depth statistics: {str(e)}")

                # Calculate settlement efficiency with optimized method if available
                try:
                    print(f"Calculating settlement efficiency")

                    if hasattr(model, 'calculate_settlement_efficiency_optimized'):
                        new_ins_eff, new_val_eff = model.calculate_settlement_efficiency_optimized()
                    else:
                        new_ins_eff, new_val_eff = model.calculate_settlement_efficiency()

                    settled_count = model.count_settled_instructions()
                    total_settled_amount = model.get_total_settled_amount()
                except Exception as e:
                    print(f"Error calculating efficiency: {str(e)}")
                    new_ins_eff, new_val_eff = 0, 0
                    settled_count, total_settled_amount = 0, 0

                # Try to save OCEL logs - if this fails due to memory, we'll try a simplified approach
                try:
                    print(f"Saving OCEL logs...")
                    model.save_ocel_log(filename=ocel_filename)
                    print(f"OCEL logs saved to {ocel_filename}")

                    # Reset logger state
                    if hasattr(model, 'logger') and hasattr(model.logger, 'reset'):
                        print("Resetting logger state...")
                        model.logger.reset()

                except Exception as e:
                    print(f"Error saving full OCEL logs: {str(e)}")
                    print(f"Attempting to save simplified logs...")

                    try:
                        # Try to save just the most important events to a backup file
                        simplified_filename = os.path.join(log_folder, f"simplified_config{true_count}_run{run}.json")

                        # Gather essential data without storing everything in memory
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

                        # Try to include a sample of events if possible
                        if hasattr(model.logger, 'events') and model.logger.events:
                            try:
                                sample_size = min(100, len(model.logger.events))
                                essential_data["sample_events"] = model.logger.events[:sample_size]
                            except Exception:
                                pass

                        # Save this simplified data
                        with open(simplified_filename, 'w') as f:
                            json.dump(essential_data, f, indent=2, default=str)

                        print(f"Saved simplified logs to {simplified_filename}")
                    except Exception as backup_error:
                        print(f"Error saving simplified logs: {str(backup_error)}")
                        print(f"Continuing without saving logs")

                # Explicit cleanup of model structures - Approach 1
                print("Explicitly cleaning model data structures")
                if hasattr(model, 'instructions'):
                    print(f"Clearing {len(model.instructions)} instructions")
                    model.instructions.clear()
                if hasattr(model, 'transactions'):
                    print(f"Clearing {len(model.transactions)} transactions")
                    model.transactions.clear()
                if hasattr(model, 'institutions'):
                    print(f"Clearing {len(model.institutions)} institutions")
                    model.institutions.clear()
                if hasattr(model, 'accounts'):
                    print(f"Clearing {len(model.accounts)} accounts")
                    model.accounts.clear()
                if hasattr(model, 'validated_delivery_instructions'):
                    print(f"Clearing validated delivery instructions")
                    model.validated_delivery_instructions.clear()
                if hasattr(model, 'validated_receipt_instructions'):
                    print(f"Clearing validated receipt instructions")
                    model.validated_receipt_instructions.clear()
                if hasattr(model, 'agents'):
                    print(f"Clearing agents")
                    model.agents.clear()
                if hasattr(model, 'event_log'):
                    print(f"Clearing event log")
                    model.event_log.clear()

                # Final memory check
                gc.collect()
                mem_final = log_memory_usage("final")

                # Return the results
                return {
                    "instruction_efficiency": new_ins_eff,
                    "value_efficiency": new_val_eff,
                    "settled_count": settled_count,
                    "settled_amount": total_settled_amount,
                    "memory_usage_mb": mem_final
                }

            # Track the runtime for this configuration
            config = {
                "partialsallowed": partialsallowed,
                "seed": seed
            }
            run_label = f"Config{true_count}_Run{run}"

            # Run the simulation with timing
            try:
                result = tracker.track_runtime(run_simulation, config, run_label)

                # Extract the simulation results
                sim_results = result["simulation_result"]
                runtime = result["execution_info"]["execution_time_seconds"]

                new_eff = {
                    'Partial': str(partialsallowed),
                    'instruction efficiency': sim_results.get("instruction_efficiency", 0),
                    'value efficiency': sim_results.get("value_efficiency", 0),
                    'settled_count': sim_results.get("settled_count", 0),
                    'settled_amount': sim_results.get("settled_amount", 0),
                    'seed': seed,  # log seed for traceability
                    'runtime_seconds': runtime,  # Add runtime to the results
                    'memory_usage_mb': sim_results.get("memory_usage_mb", 0),
                    'error': sim_results.get("error", None)  # Track any errors
                }
                efficiencies.append(new_eff)
            except Exception as e:
                print(f"ERROR in run {run_label}: {str(e)}")
                error_eff = {
                    'Partial': str(partialsallowed),
                    'instruction efficiency': 0,
                    'value efficiency': 0,
                    'settled_count': 0,
                    'settled_amount': 0,
                    'seed': seed,
                    'runtime_seconds': 0,
                    'error': str(e)
                }
                efficiencies.append(error_eff)

            # Save incremental results after each run
            df_incremental = pd.DataFrame(efficiencies)
            df_incremental.to_csv("Incremental_measurement.csv", index=False)

            # Force deep cleanup between runs
            print("Performing deep memory cleanup between runs...")
            deep_cleanup()

            # Short pause to let the system stabilize
            time.sleep(2)

            # Log memory after cleanup to check if it's being released
            log_memory_usage(f"After complete cleanup (Config{true_count}_Run{run})")

    # Save all runtime results
    tracker.save_results()

    return efficiencies


# Starting the batchrunner
if __name__ == "__main__":
    new_measured_efficiency = []  # Initialize this outside the try block
    try:
        new_measured_efficiency = batch_runner()
        print(new_measured_efficiency)
        df = pd.DataFrame(new_measured_efficiency)
        df.to_csv("New_measurement.csv", index=False)
    except Exception as e:
        print(f"Fatal error in batch runner: {str(e)}")
        # Try to save any results we have
        if new_measured_efficiency:  # Check if we have any results
            df = pd.DataFrame(new_measured_efficiency)
            df.to_csv("Emergency_results.csv", index=False)
            print(f"Saved partial results to Emergency_results.csv")