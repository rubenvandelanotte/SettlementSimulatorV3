import matplotlib.pyplot as plt
import re

# Path to your log file, make sure the path is correct
log_path = r"C:\Users\matth\Documents\GitHub\SettlementSimulatorV3\confidence_intervals.log"


def parse_log(log_path):
    instruction_data = {}
    value_data = {}

    # Regex pattern to match the log entries
    pattern = r"(INSTRUCTION_CI|VALUE_CI),Partial=\((.*?)\),Mean=([0-9.]+),Lower=([0-9.]+),Upper=([0-9.]+)"

    with open(log_path, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                entry_type, partial_str, mean, lower, upper = match.groups()
                # Count the number of True values in the Partial tuple to define the configuration
                config_num = partial_str.split(',').count("True")
                entry = {
                    "mean": float(mean),
                    "CI lower": float(lower),
                    "CI upper": float(upper)
                }
                if entry_type == "INSTRUCTION_CI":
                    instruction_data[config_num] = entry
                elif entry_type == "VALUE_CI":
                    value_data[config_num] = entry

    return instruction_data, value_data


def plot_ci(data_dict, title, color):
    configs = range(1, 11)  # We want 10 configurations
    means = []
    lower_errors = []
    upper_errors = []

    # Ensure that we have an entry for all configurations (1 to 10)
    for config in configs:
        entry = data_dict.get(config, {"mean": 0, "CI lower": 0, "CI upper": 0})
        means.append(entry["mean"])
        lower_errors.append(entry["mean"] - entry["CI lower"])
        upper_errors.append(entry["CI upper"] - entry["mean"])

    # Create the plot
    plt.errorbar(configs, means, yerr=[lower_errors, upper_errors], fmt='o', color=color,
                 capsize=5, label='Confidence Interval', markersize=8)

    # Customize the plot
    plt.title(title, fontsize=14)
    plt.xlabel("Configuratie (Partial)", fontsize=12)
    plt.ylabel("Efficiëntie (%)", fontsize=12)
    plt.xticks(configs)  # Show 1 to 10 on the x-axis
    plt.ylim(60, 100)  # Set the range for y-axis from 60 to 100
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# Run the parser and plotting functions
instruction_data, value_data = parse_log(log_path)

# Plot for instruction efficiency and value efficiency
plot_ci(instruction_data, "Betrouwbaarheidsintervallen voor instruction efficiency per configuratie", "skyblue")
plot_ci(value_data, "Betrouwbaarheidsintervallen voor value efficiency per configuratie", "salmon")

