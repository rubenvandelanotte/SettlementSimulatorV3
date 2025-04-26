# UNIVERSAL BATCHRUNNER FOR ALL ANALYSES
import os
import time
import json
import pandas as pd
from SettlementModel import SettlementModel
from RuntimeTracker import RuntimeTracker
import gc
import sys
import argparse

# Import visualizers dynamically
from PartialAnalysis import SettlementAnalyzer
from MaxDepthVisualizer import MaxDepthVisualizer
from MinSettlementAmountVisualizer import MinSettlementAmountVisualizer

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log_memory_usage(label):
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    print(f"[MEM] {label}: {mem:.2f} MB")
    return mem

def deep_cleanup():
    for _ in range(3):
        gc.collect()
    if sys.platform.startswith('win'):
        try:
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
        except Exception as e:
            print(f"[WARNING] Memory release failed: {e}")

def run_analysis(label: str, config_generator: callable, runs_per_config: int, output_dir: str, base_seed: int):
    print(f"\n=== RUNNING ANALYSIS: {label.upper()} ===\n")

    results_dir = os.path.join(output_dir, "results_all_analysis")
    log_dir = os.path.join(output_dir, "logs")
    ensure_directory(results_dir)
    ensure_directory(log_dir)

    tracker = RuntimeTracker(os.path.join(output_dir, f"runtime_{label}.json"))

    for config in config_generator(base_seed):
        true_count = config.get("true_count", 0)
        run_number = config.get("run_number", 0)

        print(f"[INFO] {label} | Config {true_count}, Run {run_number}")

        def run_simulation(config):
            model = SettlementModel(partialsallowed=config["partialsallowed"], seed=config["seed"])

            if "max_child_depth" in config:
                model.max_child_depth = config["max_child_depth"]
            if "min_settlement_percentage" in config:
                model.min_settlement_percentage = config["min_settlement_percentage"]

            try:
                while model.simulated_time < model.simulation_end:
                    model.step()
            except (RecursionError, MemoryError) as e:
                print(f"[ERROR] Simulation crashed: {e}")
                crash_filename = os.path.join(
                    results_dir,
                    f"results_{label}_config{config['true_count']}_run{config['run_number']}_CRASH.json"
                )
                with open(crash_filename, "w") as f:
                    json.dump({
                        "error": str(e),
                        "seed": config["seed"],
                        "true_count": config["true_count"],
                        "run_number": config["run_number"],
                        "label": label
                    }, f, indent=2)
                return

            mem = log_memory_usage("after simulation")

            metadata = {
                "seed": config["seed"],
                "config_id": config["true_count"],
                "run": config["run_number"],
                "label": label,
                "memory_usage_mb": mem
            }

            ocel_log_filename = os.path.join(
                log_dir,
                f"log_{label}_config{config['true_count']}_run{config['run_number']}.jsonocel"
            )
            try:
                model.save_ocel_log(filename=ocel_log_filename)
                print(f"[✓] OCEL log saved: {ocel_log_filename}")
            except Exception as e:
                print(f"[ERROR] Could not save OCEL log: {e}")

            results_filename = os.path.join(
                results_dir,
                f"results_{label}_config{config['true_count']}_run{config['run_number']}.json"
            )
            try:
                model.export_all_statistics(filename=results_filename, config_metadata=metadata)
                print(f"[✓] Statistics saved: {results_filename}")
            except Exception as e:
                print(f"[ERROR] Could not export statistics: {e}")

            deep_cleanup()
            time.sleep(1)

        tracker.track_runtime(run_simulation, config, f"{label}_Config{true_count}_Run{run_number}")

    tracker.save_results()
    print("\n[✓] Simulation runs complete.")

def generate_partial_configs(base_seed, runs_per_config):
    for true_count in range(1, 11):
        for run_number in range(1, runs_per_config + 1):
            partialsallowed = tuple([True] * true_count + [False] * (10 - true_count))
            seed = base_seed + (run_number - 1)
            yield {
                "partialsallowed": partialsallowed,
                "seed": seed,
                "true_count": true_count,
                "run_number": run_number
            }

def generate_depth_configs(base_seed, runs_per_config):
    partialsallowed = tuple([True] * 8 + [False] * 2)
    for depth in [3, 8, 15]:
        for run_number in range(1, runs_per_config + 1):
            seed = base_seed + (run_number - 1)
            yield {
                "partialsallowed": partialsallowed,
                "seed": seed,
                "max_child_depth": depth,
                "depth": depth,
                "run_number": run_number
            }

def generate_amount_configs(base_seed, runs_per_config):
    partialsallowed = tuple([True] * 8 + [False] * 2)
    for pct in [0.025, 0.05, 0.1]:
        for run_number in range(1, runs_per_config + 1):
            seed = base_seed + (run_number - 1)
            yield {
                "partialsallowed": partialsallowed,
                "seed": seed,
                "min_settlement_percentage": pct,
                "percentage": pct,
                "run_number": run_number
            }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run settlement model analysis.")
    parser.add_argument("--analysis", choices=["partial", "depth", "amount"], required=True,
                        help="Which analysis to run")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    if args.analysis == "partial":
        run_analysis("partial", lambda base_seed: generate_partial_configs(base_seed, args.runs), args.runs, "partial_allowance_files", args.seed)
        SettlementAnalyzer(results_dir="partial_allowance_files/results_all_analysis").analyze_all(
            output_base="partial_allowance_files/visualizations")

    elif args.analysis == "depth":
        run_analysis("depth", lambda base_seed: generate_depth_configs(base_seed, args.runs), args.runs, "max_depth_files", args.seed)
        MaxDepthVisualizer(results_csv="max_depth_files/max_child_depth_final_results.csv",
                           output_dir="max_depth_files/visualizations")

    elif args.analysis == "amount":
        run_analysis("amount", lambda base_seed: generate_amount_configs(base_seed, args.runs), args.runs, "min_amount_files", args.seed)
        MinSettlementAmountVisualizer(results_csv="min_amount_files/min_settlement_amount_final_results.csv",
                                       output_dir="min_amount_files/visualizations")
