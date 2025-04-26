import os
import gc
import sys
import time
import argparse

from SettlementModel import SettlementModel
from RuntimeTracker import RuntimeTracker
from settlement_analysis.run_partial_analysis import SettlementAnalysisSuite
from MaxDepthVisualizer import MaxDepthVisualizer
from MinSettlementAmountVisualizer import MinSettlementAmountVisualizer

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
            results_csv=os.path.join(output_dir, "max_child_depth_final_results.csv"),
            output_dir=os.path.join(output_dir, "visualizations", "depth")
        )
        visualizer.generate_all_visualizations()
    elif label == "amount":
        visualizer = MinSettlementAmountVisualizer(
            results_csv=os.path.join(output_dir, "min_settlement_amount_final_results.csv"),
            output_dir=os.path.join(output_dir, "visualizations", "amount")
        )
        visualizer.generate_all_visualizations()

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
    for depth in [3, 5, 7, 10, 15]:
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

def run_analysis(label, config_generator, runs_per_config, output_dir, base_seed, visualize_only=False):
    print(f"\n=== RUNNING ANALYSIS: {label.upper()} ===\n")

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
        true_count = config.get("true_count", config.get("depth", config.get("percentage", "-")))
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
                return

            ocel_log_filename = os.path.join(log_dir, f"log_{label}_config{true_count}_run{run_number}.jsonocel")
            model.save_ocel_log(filename=ocel_log_filename)

            results_filename = os.path.join(results_dir, f"results_{label}_config{true_count}_run{run_number}.json")
            model.export_all_statistics(filename=results_filename, config_metadata=config)

            deep_cleanup()
            time.sleep(1)

        tracker.track_runtime(run_simulation, config, f"{label}_Config{true_count}_Run{run_number}")

    tracker.save_results()
    print("\n[✓] Simulation runs complete.")

    finalize_visualizations(output_dir, label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run settlement model analysis.")
    parser.add_argument("--analysis", choices=["partial", "depth", "amount"], required=True, help="Which analysis to run")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--visualize_only", action="store_true", help="Skip simulation and only run visualization")
    args = parser.parse_args()

    if args.analysis == "partial":
        run_analysis("partial", lambda base_seed, runs: generate_partial_configs(base_seed, runs), args.runs, "partial_allowance_files", args.seed, visualize_only=args.visualize_only)
    elif args.analysis == "depth":
        run_analysis("depth", lambda base_seed, runs: generate_depth_configs(base_seed, runs), args.runs, "max_depth_files", args.seed, visualize_only=args.visualize_only)
    elif args.analysis == "amount":
        run_analysis("amount", lambda base_seed, runs: generate_amount_configs(base_seed, runs), args.runs, "min_amount_files", args.seed, visualize_only=args.visualize_only)