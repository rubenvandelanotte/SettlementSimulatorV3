# UNIVERSAL BATCHRUNNER FOR ALL ANALYSES
import os
import time
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

    for config_id, config in enumerate(config_generator(base_seed)):
        for run in range(1, runs_per_config + 1):
            print(f"[INFO] {label} | Config {config_id + 1}, Run {run}")

            def run_simulation(config):
                model = SettlementModel(partialsallowed=config["partialsallowed"], seed=config["seed"])

                try:
                    while model.simulated_time < model.simulation_end:
                        model.step()
                except (RecursionError, MemoryError) as e:
                    print(f"[ERROR] Simulation crashed: {e}")
                    return

                filename = os.path.join(
                    results_dir,
                    f"results_{label}_config{config_id + 1}_run{run}.json"
                )

                mem = log_memory_usage("after simulation")

                metadata = {
                    "seed": config["seed"],
                    "config_id": config_id + 1,
                    "run": run,
                    "label": label,
                    "memory_usage_mb": mem
                }

                try:
                    model.export_all_statistics(filename=filename, config_metadata=metadata)
                    print(f"[✓] Exported: {filename}")
                except Exception as e:
                    print(f"[ERROR] Could not export: {e}")


                try:
                    log_filename = os.path.join(
                        log_dir,
                        f"simulation_config{config_id + 1}_run{run}.jsonocel"
                    )
                    model.save_ocel_log(filename=log_filename)
                    print(f"[✓] Saved event log: {log_filename}")
                except Exception as e:
                    print(f"[ERROR] Could not save event log: {e}")

                deep_cleanup()
                time.sleep(1)

            tracker.track_runtime(run_simulation, config, f"{label}_Config{config_id + 1}_Run{run}")

    tracker.save_results()
    print("\n[✓] Simulation runs complete.")

# === EXAMPLE CONFIG GENERATORS ===

def generate_partial_configs(base_seed):
    for true_count in range(1, 11):
        partialsallowed = tuple([True] * true_count + [False] * (10 - true_count))
        for seed_offset in range(10):
            yield {"partialsallowed": partialsallowed, "seed": base_seed + seed_offset}

def generate_depth_configs(base_seed):
    partialsallowed = tuple([True] * 8 + [False] * 2)
    for depth in [3, 8, 15]:
        for seed_offset in range(10):
            yield {"partialsallowed": partialsallowed, "seed": base_seed + seed_offset, "max_child_depth": depth}

def generate_amount_configs(base_seed):
    partialsallowed = tuple([True] * 8 + [False] * 2)
    for pct in [0.025, 0.05, 0.1]:
        for seed_offset in range(10):
            yield {"partialsallowed": partialsallowed, "seed": base_seed + seed_offset, "min_settlement_percentage": pct}

# === ENTRYPOINT WITH ARGUMENT PARSING ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run settlement model analysis.")
    parser.add_argument("--analysis", choices=["partial", "depth", "amount"], required=True,
                        help="Which analysis to run")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    if args.analysis == "partial":
        run_analysis("partial", generate_partial_configs, args.runs, "partial_allowance_files", args.seed)
        SettlementAnalyzer(results_dir="partial_allowance_files/results_all_analysis").analyze_all(
            output_base="partial_allowance_files/visualizations")

    elif args.analysis == "depth":
        run_analysis("depth", generate_depth_configs, args.runs, "max_depth_files", args.seed)
        MaxDepthVisualizer(results_csv="max_depth_files/max_child_depth_final_results.csv",
                           output_dir="max_depth_files/visualizations")

    elif args.analysis == "amount":
        run_analysis("amount", generate_amount_configs, args.runs, "min_amount_files", args.seed)
        MinSettlementAmountVisualizer(results_csv="min_amount_files/min_settlement_amount_final_results.csv",
                                       output_dir="min_amount_files/visualizations")
