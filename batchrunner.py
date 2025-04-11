import os
from SettlementModel import SettlementModel
import pandas as pd
from RuntimeTracker import RuntimeTracker


def batch_runner():
    # Create a runtime tracker
    tracker = RuntimeTracker("runtime_results.json")

    # Maak een folder voor de logs als deze nog niet bestaat
    log_folder = "simulatie_logs"
    depth_folder = "depth_statistics"

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)

    num_institutions = 10  # Aantal instituten in de simulatie
    runs_per_config = 10  # Aantal simulaties per configuratie
    # use seeds to compare
    base_seed = 42
    seed_list = [base_seed + i for i in range(runs_per_config)]

    efficiencies = []

    # Voor elke configuratie: telkens een extra instituut op allowPartial = True
    # Hierbij wordt ervan uitgegaan dat de instellingen in de simulatie worden meegegeven als een tuple
    # van lengte num_institutions, waarbij True betekent dat partial settlement is toegestaan.
    for true_count in range(1, num_institutions + 1):  # bijvoorbeeld van 1 tot 10
        # Maak een tuple: de eerste 'true_count' posities op True, de rest op False
        partialsallowed = tuple([True] * true_count + [False] * (num_institutions - true_count))
        print(f"Simulatie configuratie: {partialsallowed}")

        seed_index = 0

        for run in range(1, runs_per_config + 1):
            print(f"Start simulatie: Configuratie met {true_count} True, run {run}")
            seed = seed_list[seed_index]
            seed_index += 1

            # Define a run_simulation function that will be timed
            def run_simulation(config):
                # Extract the configuration parameters
                partialsallowed = config["partialsallowed"]
                seed = config["seed"]

                # Create and run the model
                model = SettlementModel(partialsallowed=partialsallowed, seed=seed)

                try:
                    # Simuleer totdat de simulatie voorbij de ingestelde eindtijd is
                    while model.simulated_time < model.simulation_end:
                        model.step()
                except RecursionError:
                    print("RecursionError opgetreden: maximum recursiediepte overschreden. Simulatie wordt beëindigd.")

                # Stel bestandsnamen in met de configuratie en run-nummer
                log_filename = os.path.join(log_folder, f"log_config{true_count}_run{run}.csv")
                ocel_filename = os.path.join(log_folder, f"simulation_config{true_count}_run{run}.jsonocel")
                depth_filename = os.path.join(depth_folder, f"depth_statistics_config{true_count}_run{run}.jsonocel")

                # Sla de logs op
                model.save_ocel_log(filename=ocel_filename)

                # Save depth statistics
                stats = model.generate_depth_statistics()
                import json
                with open(depth_filename, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"Depth statistics saved to {depth_filename}")

                print(f"Logs opgeslagen voor configuratie {true_count} run {run}")
                print(f"Bereken settlement efficiency")
                new_ins_eff, new_val_eff = model.calculate_settlement_efficiency()

                settled_count = model.count_settled_instructions()
                total_settled_amount = model.get_total_settled_amount()

                # Return the results we need
                return {
                    "instruction_efficiency": new_ins_eff,
                    "value_efficiency": new_val_eff,
                    "settled_count": settled_count,
                    "settled_amount": total_settled_amount
                }

            # Track the runtime for this configuration
            config = {
                "partialsallowed": partialsallowed,
                "seed": seed
            }
            run_label = f"Config{true_count}_Run{run}"

            # Run the simulation with timing
            result = tracker.track_runtime(run_simulation, config, run_label)

            # Extract the simulation results
            sim_results = result["simulation_result"]
            runtime = result["execution_info"]["execution_time_seconds"]

            new_eff = {
                'Partial': str(partialsallowed),
                'instruction efficiency': sim_results["instruction_efficiency"],
                'value efficiency': sim_results["value_efficiency"],
                'settled_count': sim_results["settled_count"],
                'settled_amount': sim_results["settled_amount"],
                'seed': seed,  # log seed for traceability
                'runtime_seconds': runtime  # Add runtime to the results
            }
            efficiencies.append(new_eff)

    # Save all runtime results
    tracker.save_results()

    return efficiencies


# Starting the batchrunner
if __name__ == "__main__":
    new_measured_efficiency = batch_runner()
    print(new_measured_efficiency)
    df = pd.DataFrame(new_measured_efficiency)
    df.to_csv("New_measurement.csv", index=False)