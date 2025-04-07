import matplotlib.pyplot as plt
import re
import numpy as np

# Path to your log file, make sure the path is correct
log_path = r"C:\Users\matth\Documents\GitHub\SettlementSimulatorV3\confidence_intervals.log"


def parse_log(log_path):
    instruction_data = {}
    value_data = {}
    settled_count_data = {}
    settled_amount_data = {}

    # Regex pattern to match the log entries, handling potential nan values
    pattern = r"(INSTRUCTION_CI|VALUE_CI|SETTLED_COUNT|SETTLED_AMOUNT),Partial=\((.*?)\),Mean=([0-9.]+|nan),Lower=([0-9.]+|nan),Upper=([0-9.]+|nan)"

    with open(log_path, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                entry_type, partial_str, mean, lower, upper = match.groups()

                # Extract the boolean values from the partial string
                bool_values = [val.strip() == "True" for val in partial_str.split(',')]

                # Count True values to determine configuration number (1-10)
                config_num = sum(bool_values)

                # Handle 'nan' values
                mean_val = float(mean) if mean != 'nan' else np.nan
                lower_val = float(lower) if lower != 'nan' else mean_val  # Use mean if lower is nan
                upper_val = float(upper) if upper != 'nan' else mean_val  # Use mean if upper is nan

                entry = {
                    "mean": mean_val,
                    "CI lower": lower_val,
                    "CI upper": upper_val
                }

                if entry_type == "INSTRUCTION_CI":
                    instruction_data[config_num] = entry
                elif entry_type == "VALUE_CI":
                    value_data[config_num] = entry
                elif entry_type == "SETTLED_COUNT":
                    settled_count_data[config_num] = entry
                elif entry_type == "SETTLED_AMOUNT":
                    settled_amount_data[config_num] = entry

    # For demonstration - if no data is found in log, create simulated data
    if not settled_count_data:
        print("No settled count data found in log. Using simulated data for demonstration.")
        for config in sorted(instruction_data.keys()):
            settled_count_data[config] = {
                "mean": 500 + config * 80,  # Base value increases with more partial settlements
                "CI lower": 500 + config * 80 - 50,
                "CI upper": 500 + config * 80 + 50
            }

    if not settled_amount_data:
        print("No settled amount data found in log. Using simulated data for demonstration.")
        for config in sorted(instruction_data.keys()):
            settled_amount_data[config] = {
                "mean": 5000000 + config * 1000000,  # Base value increases with more partial settlements
                "CI lower": 5000000 + config * 1000000 - 500000,
                "CI upper": 5000000 + config * 1000000 + 500000
            }

    return instruction_data, value_data, settled_count_data, settled_amount_data


def plot_instruction_efficiency(instruction_data, settled_counts, filename):
    """Plot instruction efficiency with settled instruction counts on secondary y-axis"""
    # Sort configurations to ensure they're in order
    configs = sorted(instruction_data.keys())
    means = [instruction_data[config]["mean"] for config in configs]

    # Calculate errors, handling nan values
    lower_errors = []
    upper_errors = []

    for config in configs:
        mean = instruction_data[config]["mean"]
        lower = instruction_data[config]["CI lower"]
        upper = instruction_data[config]["CI upper"]

        # For lower error, if mean or lower is nan, use 0
        if np.isnan(mean) or np.isnan(lower):
            lower_errors.append(0)
        else:
            lower_errors.append(mean - lower)

        # For upper error, if mean or upper is nan, use 0
        if np.isnan(mean) or np.isnan(upper):
            upper_errors.append(0)
        else:
            upper_errors.append(upper - mean)

    # Get settled counts for the same configurations
    settled_means = [settled_counts[config]["mean"] for config in configs]

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Primary axis: Efficiency percentage
    ax1.set_xlabel("Configuratie (aantal True waarden in Partial)", fontsize=12)
    ax1.set_ylabel("Instruction Efficiëntie (%)", fontsize=12, color='skyblue')
    ax1.set_ylim(60, 100)

    # Plot means with error bars
    ax1.errorbar(configs, means, yerr=[lower_errors, upper_errors], fmt='o', color='skyblue',
                 capsize=5, label='Instruction Efficiency (%)', markersize=8)

    # Add markers for just the means (to show even when CI is nan)
    ax1.plot(configs, means, 'o', color='skyblue', markersize=8)
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # Secondary axis: Number of settled instructions
    ax2 = ax1.twinx()
    ax2.set_ylabel('Aantal Settled Instructions', fontsize=12, color='darkgreen')
    ax2.plot(configs, settled_means, 's-', color='darkgreen', markersize=8, label='Aantal Settled')
    ax2.tick_params(axis='y', labelcolor='darkgreen')

    # Determine a reasonable y-axis range for settled counts
    min_val = min(settled_means) * 0.8 if settled_means else 0
    max_val = max(settled_means) * 1.2 if settled_means else 1000
    ax2.set_ylim(min_val, max_val)

    # Title and grid
    plt.title("Betrouwbaarheidsintervallen voor instruction efficiency per configuratie", fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Legend for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_value_efficiency(value_data, settled_amounts, filename):
    """Plot value efficiency with settled amount on secondary y-axis"""
    # Sort configurations to ensure they're in order
    configs = sorted(value_data.keys())
    means = [value_data[config]["mean"] for config in configs]

    # Calculate errors, handling nan values
    lower_errors = []
    upper_errors = []

    for config in configs:
        mean = value_data[config]["mean"]
        lower = value_data[config]["CI lower"]
        upper = value_data[config]["CI upper"]

        # For lower error, if mean or lower is nan, use 0
        if np.isnan(mean) or np.isnan(lower):
            lower_errors.append(0)
        else:
            lower_errors.append(mean - lower)

        # For upper error, if mean or upper is nan, use 0
        if np.isnan(mean) or np.isnan(upper):
            upper_errors.append(0)
        else:
            upper_errors.append(upper - mean)

    # Get settled amounts for the same configurations
    settled_means = [settled_amounts[config]["mean"] for config in configs]

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Primary axis: Efficiency percentage
    ax1.set_xlabel("Configuratie (aantal True waarden in Partial)", fontsize=12)
    ax1.set_ylabel("Value Efficiëntie (%)", fontsize=12, color='salmon')
    ax1.set_ylim(60, 100)

    # Plot means with error bars
    ax1.errorbar(configs, means, yerr=[lower_errors, upper_errors], fmt='o', color='salmon',
                 capsize=5, label='Value Efficiency (%)', markersize=8)

    # Add markers for just the means (to show even when CI is nan)
    ax1.plot(configs, means, 'o', color='salmon', markersize=8)
    ax1.tick_params(axis='y', labelcolor='salmon')

    # Secondary axis: Total settled amount
    ax2 = ax1.twinx()
    ax2.set_ylabel('Totaal Settled Bedrag (€)', fontsize=12, color='darkgreen')
    ax2.plot(configs, settled_means, 's-', color='darkgreen', markersize=8, label='Totaal Settled Bedrag')
    ax2.tick_params(axis='y', labelcolor='darkgreen')

    # Format y-axis labels for large numbers (millions/billions)
    import matplotlib.ticker as ticker
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f'{x / 1e6:.1f}M' if x < 1e9 else f'{x / 1e9:.1f}B'))

    # Determine a reasonable y-axis range for settled amounts
    min_val = min(settled_means) * 0.8 if settled_means else 0
    max_val = max(settled_means) * 1.2 if settled_means else 1000000
    ax2.set_ylim(min_val, max_val)

    # Title and grid
    plt.title("Betrouwbaarheidsintervallen voor value efficiency per configuratie", fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Legend for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_combined(instruction_data, value_data, settled_counts, settled_amounts, filename):
    """Plot both efficiency metrics with their respective secondary metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Sort configurations
    configs = sorted(instruction_data.keys())

    # Get means
    means_instr = [instruction_data[config]["mean"] for config in configs]
    means_value = [value_data[config]["mean"] for config in configs]

    # Calculate errors for instruction efficiency
    lower_errors_instr = []
    upper_errors_instr = []

    for config in configs:
        mean = instruction_data[config]["mean"]
        lower = instruction_data[config]["CI lower"]
        upper = instruction_data[config]["CI upper"]

        if np.isnan(mean) or np.isnan(lower):
            lower_errors_instr.append(0)
        else:
            lower_errors_instr.append(mean - lower)

        if np.isnan(mean) or np.isnan(upper):
            upper_errors_instr.append(0)
        else:
            upper_errors_instr.append(upper - mean)

    # Calculate errors for value efficiency
    lower_errors_value = []
    upper_errors_value = []

    for config in configs:
        mean = value_data[config]["mean"]
        lower = value_data[config]["CI lower"]
        upper = value_data[config]["CI upper"]

        if np.isnan(mean) or np.isnan(lower):
            lower_errors_value.append(0)
        else:
            lower_errors_value.append(mean - lower)

        if np.isnan(mean) or np.isnan(upper):
            upper_errors_value.append(0)
        else:
            upper_errors_value.append(upper - mean)

    # Get secondary values
    settled_count_means = [settled_counts[config]["mean"] for config in configs]
    settled_amount_means = [settled_amounts[config]["mean"] for config in configs]

    # First subplot: Instruction Efficiency with Number of Instructions
    ax1.set_ylabel('Instruction Efficiëntie (%)', color='skyblue', fontsize=12)
    ax1.errorbar(configs, means_instr, yerr=[lower_errors_instr, upper_errors_instr],
                 fmt='o', color='skyblue', capsize=5, label='Instruction Efficiency', markersize=8)
    ax1.plot(configs, means_instr, 'o', color='skyblue', markersize=8)
    ax1.set_ylim(60, 100)
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax1b = ax1.twinx()
    ax1b.set_ylabel('Aantal Settled Instructions', color='darkgreen', fontsize=12)
    ax1b.plot(configs, settled_count_means, 's-', color='darkgreen', markersize=8, label='Aantal Settled')
    ax1b.tick_params(axis='y', labelcolor='darkgreen')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='lower right')

    ax1.set_title('Betrouwbaarheidsintervallen voor instruction efficiency per configuratie', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Second subplot: Value Efficiency with Total Amount
    ax2.set_ylabel('Value Efficiëntie (%)', color='salmon', fontsize=12)
    ax2.errorbar(configs, means_value, yerr=[lower_errors_value, upper_errors_value],
                 fmt='o', color='salmon', capsize=5, label='Value Efficiency', markersize=8)
    ax2.plot(configs, means_value, 'o', color='salmon', markersize=8)
    ax2.set_ylim(60, 100)
    ax2.tick_params(axis='y', labelcolor='salmon')

    ax2b = ax2.twinx()
    ax2b.set_ylabel('Totaal Settled Bedrag (€)', color='darkgreen', fontsize=12)
    ax2b.plot(configs, settled_amount_means, 's-', color='darkgreen', markersize=8, label='Totaal Settled Bedrag')
    ax2b.tick_params(axis='y', labelcolor='darkgreen')

    # Format y-axis labels for large numbers
    import matplotlib.ticker as ticker
    ax2b.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f'{x / 1e6:.1f}M' if x < 1e9 else f'{x / 1e9:.1f}B'))

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2 + lines2b, labels2 + labels2b, loc='lower right')

    ax2.set_title('Betrouwbaarheidsintervallen voor value efficiency per configuratie', fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # Common x-axis label
    fig.text(0.5, 0.04, 'Configuratie (aantal True waarden in Partial)', ha='center', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, bottom=0.1)
    plt.savefig(filename)
    plt.show()


# Run the parser and plotting functions
instruction_data, value_data, settled_counts, settled_amounts = parse_log(log_path)

# Plot for instruction efficiency with instruction counts
plot_instruction_efficiency(
    instruction_data,
    settled_counts,
    "instruction efficiency plot.png"
)

# Plot for value efficiency with settled amounts
plot_value_efficiency(
    value_data,
    settled_amounts,
    "value efficiency plot.png"
)

# Create a combined plot with both visualizations
plot_combined(
    instruction_data,
    value_data,
    settled_counts,
    settled_amounts,
    "combined efficiency plot.png"
)