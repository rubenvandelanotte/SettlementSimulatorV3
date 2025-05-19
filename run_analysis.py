import os
import gc
import sys
import time
import argparse
import psutil

from SettlementModel import SettlementModel
from RuntimeTracker import RuntimeTracker
from settlement_analysis.run_partial_analysis import SettlementAnalysisSuite
from DepthVisualizer import MaxDepthVisualizer
from SettlementAmountVisualizer import SettlementAmountVisualizer

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

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



def finalize_visualizations(output_dir, label):
    if label == "partial":
        suite = SettlementAnalysisSuite(
            input_dir=output_dir,
            output_dir=os.path.join(output_dir, "visualizations", "partial")
        )
        suite.analyze_all()
    elif label == "depth":
        visualizer = MaxDepthVisualizer(
            results_dir=os.path.join(output_dir, "results_all_analysis"),
            output_dir=os.path.join(output_dir, "visualizations", "depth")
        )
        visualizer.generate_all_visualizations()
    elif label == "amount":
        visualizer = SettlementAmountVisualizer(
            results_dir=os.path.join(output_dir, "results_all_analysis"),
            output_dir=os.path.join(output_dir, "visualizations", "amount")
        )
        visualizer.generate_all_visualizations()


def generate_partial_configs(base_seed, runs_per_config):
    for true_count in [1, 100]:
        for run_number in range(1, runs_per_config + 1):
            partialsallowed = tuple([True] * true_count + [False] * (100 - true_count))
            seed = base_seed + (run_number - 1)
            yield {
                "partialsallowed": partialsallowed,
                "seed": seed,
                "true_count": true_count,
                "run_number": run_number
            }

def generate_depth_configs(base_seed, runs_per_config):
    partialsallowed = tuple([True] * 80 + [False] * 20)
    for depth in [0,1,2,4,6,8]:
        for run_number in range(1, runs_per_config + 1):
            seed = base_seed + (run_number - 1)
            yield {
                "partialsallowed": partialsallowed,
                "seed": seed,
                "max_child_depth": depth,
                "run_number": run_number
            }

def generate_amount_configs(base_seed, runs_per_config):
    partialsallowed = tuple([True] * 8 + [False] * 2)
    for pct in [0.02, 0.035, 0.05, 0.075, 0.1]:
        for run_number in range(1, runs_per_config + 1):
            seed = base_seed + (run_number - 1)
            yield {
                "partialsallowed": partialsallowed,
                "seed": seed,
                "min_settlement_percentage": pct,
                "percentage": pct,
                "run_number": run_number
            }

def run_analysis(label, config_generator, runs_per_config, output_dir, base_seed, visualize_only=False, no_visualization=False):
    print(f"=== RUNNING ANALYSIS: {label.upper()} ===")

    results_dir = os.path.join(output_dir, "results_all_analysis")
    log_dir = os.path.join(output_dir, "logs")
    ensure_directory(results_dir)
    ensure_directory(log_dir)

    if visualize_only:
        print("[INFO] Skipping simulation. Only running visualizations.")
        finalize_visualizations(output_dir, label)
        return

    tracker = RuntimeTracker(os.path.join(output_dir, f"runtime_{label}.json"))

    for config in config_generator(base_seed, runs_per_config):
        # ✨ New naming system for result files
        if "max_child_depth" in config:
            config_name = f"maxdepth{config['max_child_depth']}"
        elif "min_settlement_percentage" in config:
            pct = int(config['min_settlement_percentage'] * 1000)  # e.g., 0.05 -> 50
            config_name = f"minpct{pct}"
        elif "true_count" in config:
            config_name = f"truecount{config['true_count']}"
        else:
            config_name = "unknown"

        run_number = config.get("run_number", 0)

        print(f"[INFO] {label} | Config {config_name}, Run {run_number}")

        def run_simulation(config):

            process = psutil.Process(os.getpid())

            # Start meten
            start_time = time.time()
            start_mem = process.memory_info().rss / (1024 * 1024)

            model = SettlementModel(
                partialsallowed=config["partialsallowed"],
                seed=config["seed"],
                run_number=config.get("run_number", 1)  # Pass the run number explicitly
            )

            if "max_child_depth" in config:
                model.max_child_depth = config["max_child_depth"]
            if "min_settlement_percentage" in config:
                model.min_settlement_percentage = config["min_settlement_percentage"]

            max_retries = 3
            retries = 0

            while retries < max_retries:
                try:
                    while model.simulated_time < model.simulation_end:
                        model.step()
                    break
                except (RecursionError, MemoryError) as e:
                    retries += 1
                    print(f"[ERROR] Simulation crashed (attempt {retries}/{max_retries}): {e}")
                    if retries >= max_retries:
                        print("[FATAL] Simulation failed after maximum retries.")
                        return


            end_time = time.time()
            end_mem = process.memory_info().rss / (1024 * 1024)  # in MB

            execution_time_seconds = end_time - start_time
            memory_usage_mb = max(end_mem - start_mem, 0)

            config_metadata = config.copy()
            config_metadata["execution_time_seconds"] = execution_time_seconds
            config_metadata["memory_usage_mb"] = memory_usage_mb

            # 🚀 Save files with better names
            ocel_log_filename = os.path.join(log_dir, f"log_{label}_{config_name}_run{run_number}.jsonocel")
            model.save_ocel_log(filename=ocel_log_filename)

            results_filename = os.path.join(results_dir, f"results_{label}_{config_name}_run{run_number}.json")
            model.export_all_statistics(filename=results_filename, config_metadata=config_metadata)
            deep_cleanup()
            time.sleep(1)

        tracker.track_runtime(run_simulation, config, f"{label}_{config_name}_run{run_number}")

    tracker.save_results()
    print("[✓] Simulation runs complete.")


    if no_visualization:
        print("[INFO] Skipping visualizations because --no_visualization was passed.")
        return

    finalize_visualizations(output_dir, label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run settlement model analysis.")
    parser.add_argument("--analysis", choices=["partial", "depth", "amount", "all"], required=True, help="Which analysis to run")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--visualize_only", action="store_true", help="Only generate visualizations, skip simulation")
    parser.add_argument("--no_visualization", action="store_true", help="Run simulations but skip visualizations")
    args = parser.parse_args()

    if args.analysis == "partial":
        run_analysis("partial", generate_partial_configs, args.runs, "partial_allowance_files_2", args.seed, visualize_only=args.visualize_only, no_visualization=args.no_visualization)
    elif args.analysis == "depth":
        run_analysis("depth", generate_depth_configs, args.runs, "max_depth_files", args.seed, visualize_only=args.visualize_only, no_visualization=args.no_visualization)
    elif args.analysis == "amount":
        run_analysis("amount", generate_amount_configs, args.runs, "min_amount_files", args.seed, visualize_only=args.visualize_only, no_visualization=args.no_visualization)
    elif args.analysis == "all":
        run_analysis("partial", generate_partial_configs, args.runs, "partial_allowance_files", args.seed, visualize_only=args.visualize_only, no_visualization=args.no_visualization)
        run_analysis("depth", generate_depth_configs, args.runs, "max_depth_files", args.seed, visualize_only=args.visualize_only, no_visualization=args.no_visualization)
        run_analysis("amount", generate_amount_configs, args.runs, "min_amount_files", args.seed, visualize_only=args.visualize_only, no_visualization=args.no_visualization)
