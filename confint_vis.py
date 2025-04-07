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

                # Extract the boolean values from the partial string
                bool_values = [val.strip() == "True" for val in partial_str.split(',')]

                # Count True values to determine configuration number (1-10)
                config_num = sum(bool_values)

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


def plot_ci(data_dict, title, color, filename):
    # Sort configurations to ensure they're in order
    configs = sorted(data_dict.keys())
    means = [data_dict[config]["mean"] for config in configs]
    lower_errors = [data_dict[config]["mean"] - data_dict[config]["CI lower"] for config in configs]
    upper_errors = [data_dict[config]["CI upper"] - data_dict[config]["mean"] for config in configs]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(configs, means, yerr=[lower_errors, upper_errors], fmt='o', color=color,
                 capsize=5, label='Confidence Interval', markersize=8)

    # Customize the plot
    plt.title(title, fontsize=14)
    plt.xlabel("Configuratie (aantal True waarden in Partial)", fontsize=12)
    plt.ylabel("Efficiëntie (%)", fontsize=12)
    plt.xticks(configs)  # Show configurations on the x-axis
    plt.ylim(60, 100)  # Set the range for y-axis from 60 to 100
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)  # Save the plot with the specified filename
    plt.show()


def plot_combined_ci(instruction_data, value_data, filename):
    plt.figure(figsize=(12, 8))

    # Plot instruction efficiency
    configs_instr = sorted(instruction_data.keys())
    means_instr = [instruction_data[config]["mean"] for config in configs_instr]
    lower_errors_instr = [instruction_data[config]["mean"] - instruction_data[config]["CI lower"] for config in
                          configs_instr]
    upper_errors_instr = [instruction_data[config]["CI upper"] - instruction_data[config]["mean"] for config in
                          configs_instr]

    plt.errorbar(configs_instr, means_instr, yerr=[lower_errors_instr, upper_errors_instr],
                 fmt='o', color='skyblue', capsize=5, label='Instruction Efficiency', markersize=8)

    # Plot value efficiency
    configs_value = sorted(value_data.keys())
    means_value = [value_data[config]["mean"] for config in configs_value]
    lower_errors_value = [value_data[config]["mean"] - value_data[config]["CI lower"] for config in configs_value]
    upper_errors_value = [value_data[config]["CI upper"] - value_data[config]["mean"] for config in configs_value]

    plt.errorbar(configs_value, means_value, yerr=[lower_errors_value, upper_errors_value],
                 fmt='s', color='salmon', capsize=5, label='Value Efficiency', markersize=8)

    # Customize the plot
    plt.title("Betrouwbaarheidsintervallen voor Efficiency per configuratie", fontsize=14)
    plt.xlabel("Configuratie (aantal True waarden in Partial)", fontsize=12)
    plt.ylabel("Efficiëntie (%)", fontsize=12)
    plt.xticks(range(1, 11))  # Show 1 to 10 on the x-axis
    plt.ylim(60, 100)  # Set the range for y-axis from 60 to 100
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)  # Save the combined plot with the specified filename
    plt.show()


# Run the parser and plotting functions
instruction_data, value_data = parse_log(log_path)

# Plot for instruction efficiency with specified filename
plot_ci(instruction_data,
        "Betrouwbaarheidsintervallen voor instruction efficiency per configuratie",
        "skyblue",
        "instruction efficiency plot.png")

# Plot for value efficiency with specified filename
plot_ci(value_data,
        "Betrouwbaarheidsintervallen voor value efficiency per configuratie",
        "salmon",
        "value efficiency plot.png")

# Create a combined plot with specified filename
plot_combined_ci(instruction_data, value_data, "combined efficiency plot.png")