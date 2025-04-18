import os
import pandas as pd
import time
import psutil  # For memory monitoring
import json
import argparse
from SettlementModel import SettlementModel
from RuntimeTracker import RuntimeTracker
import gc
import sys
import scipy.stats as stats
import logging

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


def ensure_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def run_partial_allowance_analysis(runs_per_config=10, base_seed=42):
    """
    Run analysis for the impact of partial settlement allowance.

    Args:
        runs_per_config (int): Number of runs per configuration
        base_seed (int): Base seed for random number generation

    Returns:
        list: Results for all simulation runs
    """
    print("\n=== RUNNING PARTIAL ALLOWANCE ANALYSIS ===\n")

    # Create folders for logs
    log_folder = "partial_allowance_logs"
    depth_folder = "partial_allowance_depth"
    folder = "partial_allowance_files"
    ensure_directory(log_folder)
    ensure_directory(depth_folder)

    # Create a runtime tracker
    tracker = RuntimeTracker(os.path.join(folder, "runtime_partial_allowance.json"))

    num_institutions = 10  # Number of institutions in the simulation
    seed_list = [base_seed + i for i in range(runs_per_config)]

    efficiencies = []

    # For each configuration: add one more institution with allowPartial = True
    for true_count in range(1, num_institutions + 1):
        # Create a tuple: first 'true_count' positions are True, rest are False
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
                    with open(depth_filename, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"Depth statistics saved to {depth_filename}")
                except Exception as e:
                    print(f"Error saving depth statistics: {str(e)}")

                # Calculate settlement efficiency
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

                # Try to save OCEL logs
                try:
                    print(f"Saving OCEL logs...")
                    model.save_ocel_log(filename=ocel_filename)
                    print(f"OCEL logs saved to {ocel_filename}")

                    # Reset logger state if possible
                    if hasattr(model.logger, 'reset'):
                        model.logger.reset()

                except Exception as e:
                    print(f"Error saving full OCEL logs: {str(e)}")

                    # Try to save simplified logs
                    try:
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
                        print(f"Error saving simplified logs: {str(backup_error)}")

                # Explicit cleanup of model structures
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
                    'true_count': true_count,
                    'instruction_efficiency': sim_results.get("instruction_efficiency", 0),
                    'value_efficiency': sim_results.get("value_efficiency", 0),
                    'settled_count': sim_results.get("settled_count", 0),
                    'settled_amount': sim_results.get("settled_amount", 0),
                    'seed': seed,
                    'runtime_seconds': runtime,
                    'memory_usage_mb': sim_results.get("memory_usage_mb", 0),
                    'error': sim_results.get("error", None)
                }
                efficiencies.append(new_eff)
            except Exception as e:
                print(f"ERROR in run {run_label}: {str(e)}")
                error_eff = {
                    'Partial': str(partialsallowed),
                    'true_count': true_count,
                    'instruction_efficiency': 0,
                    'value_efficiency': 0,
                    'settled_count': 0,
                    'settled_amount': 0,
                    'seed': seed,
                    'runtime_seconds': 0,
                    'error': str(e)
                }
                efficiencies.append(error_eff)

            # Save incremental results after each run
            df_incremental = pd.DataFrame(efficiencies)
            df_incremental.to_csv(os.path.join(folder,"partial_allowance_results_incremental.csv"), index=False)

            # Force deep cleanup between runs
            print("Performing deep memory cleanup between runs...")
            deep_cleanup()

            # Short pause to let the system stabilize
            time.sleep(2)

            # Log memory after cleanup to check if it's being released
            log_memory_usage(f"After complete cleanup (Config{true_count}_Run{run})")

    # Save all runtime results
    tracker.save_results()

    # Save final results
    df = pd.DataFrame(efficiencies)
    df.to_csv(os.path.join(folder, "partial_allowance_final_results.csv"), index=False)

    print("\nAnalyzing results with confidence intervals...")
    analyze_results_with_confidence_intervals(df, "true_count", folder)

    return efficiencies


def run_max_child_depth_analysis(depths_to_test=[3, 8, 15], runs_per_config=20, base_seed=42):
    """
    Run analysis to test different maximum child depths.

    Args:
        depths_to_test (list): List of maximum child depths to test
        runs_per_config (int): Number of runs per configuration
        base_seed (int): Base seed for random number generation

    Returns:
        list: Results for all simulation runs
    """
    print("\n=== RUNNING MAXIMUM CHILD DEPTH ANALYSIS ===\n")

    # Create folders for logs
    log_folder = "max_depth_logs"
    depth_folder = "max_depth_stats"
    folder = "max_depth_files"
    ensure_directory(log_folder)
    ensure_directory(depth_folder)

    # Create a runtime tracker
    tracker = RuntimeTracker(os.path.join(folder, "runtime_max_depth.json"))

    seed_list = [base_seed + i for i in range(runs_per_config)]

    # Set up standard partial allowance configuration (8 True, 2 False)
    partialsallowed = tuple([True] * 8 + [False] * 2)

    efficiencies = []

    # Test each maximum child depth
    for max_depth in depths_to_test:
        print(f"Testing maximum child depth: {max_depth}")

        for run in range(1, runs_per_config + 1):
            print(f"Starting simulation: Max depth {max_depth}, run {run}")
            seed = seed_list[run - 1]

            # Define a run_simulation function that will be timed
            def run_simulation(config):
                # Extract the configuration parameters
                partialsallowed = config["partialsallowed"]
                max_child_depth = config["max_child_depth"]
                seed = config["seed"]

                # Monitor memory before starting
                mem_before = log_memory_usage("before simulation")

                # Create and run the model
                model = SettlementModel(partialsallowed=partialsallowed, seed=seed)

                # Override the maximum child depth
                model.max_child_depth = max_child_depth

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
                ocel_filename = os.path.join(log_folder, f"simulation_depth{max_child_depth}_run{run}.jsonocel")
                depth_filename = os.path.join(depth_folder, f"depth_statistics_depth{max_child_depth}_run{run}.json")

                # Monitor memory before saving logs
                mem_after_sim = log_memory_usage("after simulation")

                # Save depth statistics first (smaller data)
                try:
                    stats = model.generate_depth_statistics()
                    with open(depth_filename, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"Depth statistics saved to {depth_filename}")
                except Exception as e:
                    print(f"Error saving depth statistics: {str(e)}")

                # Calculate settlement efficiency
                try:
                    print(f"Calculating settlement efficiency")

                    if hasattr(model, 'calculate_settlement_efficiency_optimized'):
                        new_ins_eff, new_val_eff = model.calculate_settlement_efficiency_optimized()
                    else:
                        new_ins_eff, new_val_eff = model.calculate_settlement_efficiency()

                    settled_count = model.count_settled_instructions()
                    total_settled_amount = model.get_total_settled_amount()
                    avg_tree_depth = model.get_average_tree_depth() if hasattr(model, 'get_average_tree_depth') else 0
                    partial_settlements = model.get_partial_settlement_count() if hasattr(model,
                                                                                          'get_partial_settlement_count') else 0

                except Exception as e:
                    print(f"Error calculating efficiency: {str(e)}")
                    new_ins_eff, new_val_eff = 0, 0
                    settled_count, total_settled_amount = 0, 0
                    avg_tree_depth, partial_settlements = 0, 0

                # Try to save OCEL logs
                try:
                    print(f"Saving OCEL logs...")
                    model.save_ocel_log(filename=ocel_filename)
                    print(f"OCEL logs saved to {ocel_filename}")

                    # Reset logger state if possible
                    if hasattr(model.logger, 'reset'):
                        model.logger.reset()

                except Exception as e:
                    print(f"Error saving full OCEL logs: {str(e)}")

                    # Try to save simplified logs
                    try:
                        simplified_filename = os.path.join(log_folder,
                                                           f"simplified_depth{max_child_depth}_run{run}.json")

                        # Gather essential data
                        essential_data = {
                            "configuration": {
                                "partialsallowed": str(partialsallowed),
                                "max_child_depth": max_child_depth,
                                "seed": seed,
                                "run": run
                            },
                            "results": {
                                "instruction_efficiency": new_ins_eff,
                                "value_efficiency": new_val_eff,
                                "settled_count": settled_count,
                                "settled_amount": total_settled_amount,
                                "avg_tree_depth": avg_tree_depth,
                                "partial_settlements": partial_settlements
                            },
                            "timestamp": time.time()
                        }

                        # Save this simplified data
                        with open(simplified_filename, 'w') as f:
                            json.dump(essential_data, f, indent=2, default=str)

                        print(f"Saved simplified logs to {simplified_filename}")
                    except Exception as backup_error:
                        print(f"Error saving simplified logs: {str(backup_error)}")

                # Explicit cleanup of model structures
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
                    "avg_tree_depth": avg_tree_depth,
                    "partial_settlements": partial_settlements,
                    "memory_usage_mb": mem_final
                }

            # Track the runtime for this configuration
            config = {
                "partialsallowed": partialsallowed,
                "max_child_depth": max_depth,
                "seed": seed
            }
            run_label = f"Depth{max_depth}_Run{run}"

            # Run the simulation with timing
            try:
                result = tracker.track_runtime(run_simulation, config, run_label)

                # Extract the simulation results
                sim_results = result["simulation_result"]
                runtime = result["execution_info"]["execution_time_seconds"]

                new_eff = {
                    'max_child_depth': max_depth,
                    'instruction_efficiency': sim_results.get("instruction_efficiency", 0),
                    'value_efficiency': sim_results.get("value_efficiency", 0),
                    'settled_count': sim_results.get("settled_count", 0),
                    'settled_amount': sim_results.get("settled_amount", 0),
                    'avg_tree_depth': sim_results.get("avg_tree_depth", 0),
                    'partial_settlements': sim_results.get("partial_settlements", 0),
                    'seed': seed,
                    'runtime_seconds': runtime,
                    'memory_usage_mb': sim_results.get("memory_usage_mb", 0),
                    'error': sim_results.get("error", None)
                }
                efficiencies.append(new_eff)
            except Exception as e:
                print(f"ERROR in run {run_label}: {str(e)}")
                error_eff = {
                    'max_child_depth': max_depth,
                    'instruction_efficiency': 0,
                    'value_efficiency': 0,
                    'settled_count': 0,
                    'settled_amount': 0,
                    'avg_tree_depth': 0,
                    'partial_settlements': 0,
                    'seed': seed,
                    'runtime_seconds': 0,
                    'error': str(e)
                }
                efficiencies.append(error_eff)

            # Save incremental results after each run
            df_incremental = pd.DataFrame(efficiencies)
            df_incremental.to_csv(os.path.join(folder,"max_child_depth_results_incremental.csv"), index=False)

            # Force deep cleanup between runs
            print("Performing deep memory cleanup between runs...")
            deep_cleanup()

            # Short pause to let the system stabilize
            time.sleep(2)

            # Log memory after cleanup to check if it's being released
            log_memory_usage(f"After complete cleanup (Depth{max_depth}_Run{run})")

    # Save all runtime results
    tracker.save_results()

    # Save final results
    df = pd.DataFrame(efficiencies)
    df.to_csv(os.path.join(folder, "max_child_depth_final_results.csv"), index=False)

    print("\nAnalyzing results with confidence intervals...")
    analyze_results_with_confidence_intervals(df, "max_child_depth", folder)

    return efficiencies


def run_min_settlement_amount_analysis(percentages_to_test=[0.025, 0.05, 0.1], runs_per_config=5, base_seed=42):
    """
    Run analysis to test different minimum settlement amounts.

    Args:
        percentages_to_test (list): List of minimum settlement percentages to test (as decimals)
        runs_per_config (int): Number of runs per configuration
        base_seed (int): Base seed for random number generation

    Returns:
        list: Results for all simulation runs
    """
    print("\n=== RUNNING MINIMUM SETTLEMENT AMOUNT ANALYSIS ===\n")

    # Create folders for logs
    log_folder = "min_amount_logs"
    depth_folder = "min_amount_stats"
    folder = "min_amount_files"
    ensure_directory(log_folder)
    ensure_directory(depth_folder)

    # Create a runtime tracker
    tracker = RuntimeTracker(os.path.join(folder, "runtime_min_amount.json"))

    seed_list = [base_seed + i for i in range(runs_per_config)]

    # Set up standard partial allowance configuration (8 True, 2 False)
    partialsallowed = tuple([True] * 8 + [False] * 2)

    efficiencies = []

    # Test each minimum settlement percentage
    for min_pct in percentages_to_test:
        print(f"Testing minimum settlement percentage: {min_pct * 100:.1f}%")

        for run in range(1, runs_per_config + 1):
            print(f"Starting simulation: Min percentage {min_pct * 100:.1f}%, run {run}")
            seed = seed_list[run - 1]

            # Define a run_simulation function that will be timed
            def run_simulation(config):
                # Extract the configuration parameters
                partialsallowed = config["partialsallowed"]
                min_settlement_percentage = config["min_settlement_percentage"]
                seed = config["seed"]

                # Monitor memory before starting
                mem_before = log_memory_usage("before simulation")

                # Create and run the model
                model = SettlementModel(partialsallowed=partialsallowed, seed=seed)

                # Override the minimum settlement percentage
                model.min_settlement_percentage = min_settlement_percentage

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
                pct_str = f"{int(min_settlement_percentage * 100)}"
                ocel_filename = os.path.join(log_folder, f"simulation_minpct{pct_str}_run{run}.jsonocel")
                depth_filename = os.path.join(depth_folder, f"depth_statistics_minpct{pct_str}_run{run}.json")

                # Monitor memory before saving logs
                mem_after_sim = log_memory_usage("after simulation")

                # Save depth statistics first (smaller data)
                try:
                    stats = model.generate_depth_statistics()
                    with open(depth_filename, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"Depth statistics saved to {depth_filename}")
                except Exception as e:
                    print(f"Error saving depth statistics: {str(e)}")

                    # Calculate settlement efficiency
                try:
                    print(f"Calculating settlement efficiency")

                    if hasattr(model, 'calculate_settlement_efficiency_optimized'):
                        new_ins_eff, new_val_eff = model.calculate_settlement_efficiency_optimized()
                    else:
                        new_ins_eff, new_val_eff = model.calculate_settlement_efficiency()

                    settled_count = model.count_settled_instructions()
                    total_settled_amount = model.get_total_settled_amount()
                    partial_settlements = model.get_partial_settlement_count() if hasattr(model,
                                                                                          'get_partial_settlement_count') else 0

                except Exception as e:
                    print(f"Error calculating efficiency: {str(e)}")
                    new_ins_eff, new_val_eff = 0, 0
                    settled_count, total_settled_amount = 0, 0
                    partial_settlements = 0

                # Try to save OCEL logs
                try:
                    print(f"Saving OCEL logs...")
                    model.save_ocel_log(filename=ocel_filename)
                    print(f"OCEL logs saved to {ocel_filename}")

                    # Reset logger state if possible
                    if hasattr(model.logger, 'reset'):
                        model.logger.reset()

                except Exception as e:
                    print(f"Error saving full OCEL logs: {str(e)}")

                    # Try to save simplified logs
                    try:
                        simplified_filename = os.path.join(log_folder, f"simplified_minpct{pct_str}_run{run}.json")

                        # Gather essential data
                        essential_data = {
                            "configuration": {
                                "partialsallowed": str(partialsallowed),
                                "min_settlement_percentage": min_settlement_percentage,
                                "seed": seed,
                                "run": run
                            },
                            "results": {
                                "instruction_efficiency": new_ins_eff,
                                "value_efficiency": new_val_eff,
                                "settled_count": settled_count,
                                "settled_amount": total_settled_amount,
                                "partial_settlements": partial_settlements
                            },
                            "timestamp": time.time()
                        }

                        # Save this simplified data
                        with open(simplified_filename, 'w') as f:
                            json.dump(essential_data, f, indent=2, default=str)

                        print(f"Saved simplified logs to {simplified_filename}")
                    except Exception as backup_error:
                        print(f"Error saving simplified logs: {str(backup_error)}")

                # Explicit cleanup of model structures
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
                    "partial_settlements": partial_settlements,
                    "memory_usage_mb": mem_final
                }

                # Track the runtime for this configuration
            config = {
                "partialsallowed": partialsallowed,
                "min_settlement_percentage": min_pct,
                "seed": seed
            }
            run_label = f"MinPct{int(min_pct * 100)}_Run{run}"

            # Run the simulation with timing
            try:
                result = tracker.track_runtime(run_simulation, config, run_label)

                # Extract the simulation results
                sim_results = result["simulation_result"]
                runtime = result["execution_info"]["execution_time_seconds"]

                new_eff = {
                    'min_settlement_percentage': min_pct,
                    'min_pct_str': f"{min_pct * 100:.1f}%",
                    'instruction_efficiency': sim_results.get("instruction_efficiency", 0),
                    'value_efficiency': sim_results.get("value_efficiency", 0),
                    'settled_count': sim_results.get("settled_count", 0),
                    'settled_amount': sim_results.get("settled_amount", 0),
                    'partial_settlements': sim_results.get("partial_settlements", 0),
                    'seed': seed,
                    'runtime_seconds': runtime,
                    'memory_usage_mb': sim_results.get("memory_usage_mb", 0),
                    'error': sim_results.get("error", None)
                }
                efficiencies.append(new_eff)
            except Exception as e:
                print(f"ERROR in run {run_label}: {str(e)}")
                error_eff = {
                    'min_settlement_percentage': min_pct,
                    'min_pct_str': f"{min_pct * 100:.1f}%",
                    'instruction_efficiency': 0,
                    'value_efficiency': 0,
                    'settled_count': 0,
                    'settled_amount': 0,
                    'partial_settlements': 0,
                    'seed': seed,
                    'runtime_seconds': 0,
                    'error': str(e)
                }
                efficiencies.append(error_eff)

            # Save incremental results after each run
            df_incremental = pd.DataFrame(efficiencies)
            df_incremental.to_csv(os.path.join(folder,"min_settlement_amount_results_incremental.csv"), index=False)

            # Force deep cleanup between runs
            print("Performing deep memory cleanup between runs...")
            deep_cleanup()

            # Short pause to let the system stabilize
            time.sleep(2)

            # Log memory after cleanup to check if it's being released
            log_memory_usage(f"After complete cleanup (MinPct{int(min_pct * 100)}_Run{run})")

            # Save all runtime results

    tracker.save_results()

    # Save final results
    df = pd.DataFrame(efficiencies)
    df.to_csv(os.path.join(folder,"min_settlement_amount_final_results.csv"), index=False)

    print("\nAnalyzing results with confidence intervals...")
    analyze_results_with_confidence_intervals(df, "min_settlement_percentage", folder)

    return efficiencies

def analyze_results_with_confidence_intervals(results_df, group_column, output_folder):
    """
    Analyze results with confidence intervals and log them.

    Args:
        results_df (pandas.DataFrame): DataFrame containing the results
        group_column (str): Column name to group by (e.g., 'true_count', 'Partial', 'max_child_depth')
        output_folder (str): Folder to save the confidence interval logs
    """
        # Set up logging
    log_file = os.path.join(output_folder, "confidence_intervals.log")
    ci_logger = logging.getLogger("confidence_intervals")

    # Check if handlers already exist to avoid duplicate handlers
    if not ci_logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="w")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        ci_logger.setLevel(logging.INFO)
        ci_logger.addHandler(file_handler)

    def compute_confidence_interval(data, confidence=0.95):
        """
        Calculate the mean and confidence interval for the given data.
        """
        n = len(data)
        if n < 2:
            return data.mean(), data.mean(), data.mean()  # Can't compute CI with single value

        mean = data.mean()
        std_err = stats.sem(data)
        h = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
        return mean, mean - h, mean + h

    # List of metrics to analyze
    metrics = [
        'instruction_efficiency',
        'value_efficiency',
        'settled_count',
        'settled_amount'
    ]

    # Add other metrics if they exist in the DataFrame
    if 'avg_tree_depth' in results_df.columns:
        metrics.append('avg_tree_depth')
    if 'partial_settlements' in results_df.columns:
        metrics.append('partial_settlements')
    if 'runtime_seconds' in results_df.columns:
        metrics.append('runtime_seconds')

    results = {}

    # Analyze each metric
    for metric in metrics:
        if metric not in results_df.columns:
            print(f"Metric {metric} not found in results")
            continue

        grouped = results_df.groupby(group_column)[metric]
        metric_results = {}

        ci_logger.info(f"Confidence intervals for {metric} by {group_column}:")
        print(f"\nConfidence intervals for {metric} by {group_column}:")

        for config, values in grouped:
            try:
                # Skip if all values are NaN or there's only one value
                if values.isnull().all() or len(values) < 2:
                    mean = values.mean() if not values.isnull().all() else 0
                    ci_logger.info(f"{metric.upper()},{group_column}={config},Mean={mean:.4f},CI=N/A")
                    print(f"{config}: Mean = {mean:.2f}, CI = N/A (insufficient data)")
                    continue

                mean, lower, upper = compute_confidence_interval(values)
                metric_results[config] = {"mean": mean, "CI lower": lower, "CI upper": upper}
                ci_logger.info(
                    f"{metric.upper()},{group_column}={config},Mean={mean:.4f},Lower={lower:.4f},Upper={upper:.4f}")
                print(f"{config}: Mean = {mean:.2f}, CI = [{lower:.2f}, {upper:.2f}]")
            except Exception as e:
                ci_logger.error(f"Error computing CI for {metric}, {group_column}={config}: {str(e)}")
                print(f"Error for {config}: {str(e)}")

        results[metric] = metric_results

    # Save the results to a structured CSV file
    try:
        ci_data = []
        for metric, configs in results.items():
            for config, stats_dict in configs.items():
                ci_data.append({
                    'metric': metric,
                    group_column: config,
                    'mean': stats_dict.get('mean', 0),
                    'ci_lower': stats_dict.get('CI lower', 0),
                    'ci_upper': stats_dict.get('CI upper', 0)
                })

        ci_df = pd.DataFrame(ci_data)
        ci_csv = os.path.join(output_folder, f"{group_column}_confidence_intervals.csv")
        ci_df.to_csv(ci_csv, index=False)
        print(f"Confidence intervals saved to {ci_csv}")
    except Exception as e:
        print(f"Error saving confidence intervals to CSV: {str(e)}")

    return results



# Main entry point
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run settlement simulation analyses')
    parser.add_argument('analysis', type=str, choices=['partial', 'depth', 'amount'],
                        help='Which analysis to run: partial (allowance), depth (max child depth), or amount (min settlement)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs per configuration (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for random number generation (default: 42)')

    args = parser.parse_args()

    print(f"Starting analysis: {args.analysis}")
    print(f"Runs per configuration: {args.runs}")
    print(f"Base seed: {args.seed}")

    try:
        # Run the selected analysis
        if args.analysis == 'partial':
            results = run_partial_allowance_analysis(runs_per_config=args.runs, base_seed=args.seed)
            print("Partial allowance analysis completed successfully!")
            df = pd.DataFrame(results)
            df.to_csv("partial_allowance_results_final.csv", index=False)

        elif args.analysis == 'depth':
            results = run_max_child_depth_analysis(runs_per_config=args.runs, base_seed=args.seed)
            print("Maximum child depth analysis completed successfully!")
            df = pd.DataFrame(results)
            df.to_csv("max_child_depth_results_final.csv", index=False)

        elif args.analysis == 'amount':
            results = run_min_settlement_amount_analysis(runs_per_config=args.runs, base_seed=args.seed)
            print("Minimum settlement amount analysis completed successfully!")
            df = pd.DataFrame(results)
            df.to_csv("min_settlement_amount_results_final.csv", index=False)

    except Exception as e:
        print(f"Fatal error in batch runner: {str(e)}")
        # Try to save any partial results
        print("Analysis terminated with errors")