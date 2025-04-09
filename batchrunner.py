import os
import multiprocessing
from SettlementModel import SettlementModel
import pandas as pd

# Settings
num_institutions = 10
runs_per_config = 5
base_seed = 42
log_folder = "simulatie_logs"

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

def run_config(true_count):
    """Run meerdere simulaties voor een bepaalde configuratie (aantal True waarden)."""
    partialsallowed = tuple([True] * true_count + [False] * (num_institutions - true_count))
    print(f"[CONFIG] Start configuratie met {true_count}x True → {partialsallowed}")

    efficiencies = []
    for run in range(1, runs_per_config + 1):
        seed = base_seed + run
        print(f"  [RUN {run}] Seed: {seed}")

        model = SettlementModel(partialsallowed=partialsallowed, seed=seed)

        try:
            while model.simulated_time < model.simulation_end:
                model.step()
        except RecursionError:
            print("  [WAARSCHUWING] RecursionError bij config {true_count}, run {run}")

        # Save logs
        log_filename = os.path.join(log_folder, f"log_config{true_count}_run{run}.csv")
        ocel_filename = os.path.join(log_folder, f"simulation_config{true_count}_run{run}.jsonocel")
        model.save_log(filename=log_filename)
        model.save_ocel_log(filename=ocel_filename)

        # Metrics
        new_ins_eff, new_val_eff = model.calculate_settlement_efficiency()
        settled_count = model.count_settled_instructions()
        total_settled_amount = model.get_total_settled_amount()

        efficiencies.append({
            'Partial': str(partialsallowed),
            'instruction efficiency': new_ins_eff,
            'value efficiency': new_val_eff,
            'settled_count': settled_count,
            'settled_amount': total_settled_amount,
            'seed': seed
        })

    return efficiencies


if __name__ == "__main__":
    print("[INFO] Start parallelle batchrun per configuratie")

    # Detecteer het aantal CPU-kernen en beperk het aantal processen
    max_processes = min(multiprocessing.cpu_count(), num_institutions)
    print(f"[INFO] Aantal CPU-kernen gedetecteerd: {multiprocessing.cpu_count()}")
    print(f"[INFO] Maximaal {max_processes} parallelle processen toegelaten")

    with multiprocessing.Pool(processes=max_processes) as pool:
        all_results = pool.map(run_config, list(range(1, num_institutions + 1)))

    # Flatten results
    flat_results = [item for sublist in all_results for item in sublist]

    # Opslaan als CSV
    df = pd.DataFrame(flat_results)
    df.to_csv("New_measurement.csv", index=False)
    print("[INFO] Simulaties afgerond. Resultaten opgeslagen in 'New_measurement.csv'")
