import pandas as pd
import scipy.stats as stats
import logging

logging.basicConfig(
    filename="confidence_intervals.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_confidence_interval(data, confidence=0.95):
    """
    Bereken het gemiddelde en het 95%-betrouwbaarheidsinterval voor de gegeven data.
    """
    n = len(data)
    mean = data.mean()
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - h, mean + h


# Lees de CSV met de gemeten efficiencies
df = pd.read_csv("New_measurement.csv")

# Bereken betrouwbaarheidsintervallen voor instruction efficiency per configuratie (Partial)
grouped_instruction = df.groupby("Partial")["instruction efficiency"]

results_instruction = {}

logging.info("Betrouwbaarheidsintervallen voor instruction efficiency per configuratie:")
for config, values in grouped_instruction:
    mean, lower, upper = compute_confidence_interval(values)
    results_instruction[config] = {"mean": mean, "CI lower": lower, "CI upper": upper}
    logging.info(f"INSTRUCTION_CI,Partial={config},Mean={mean:.4f},Lower={lower:.4f},Upper={upper:.4f}")

print("Betrouwbaarheidsintervallen voor instruction efficiency per configuratie:")
for config, stats_dict in results_instruction.items():
    print(
        f"{config}: Mean = {stats_dict['mean']:.2f}, CI = [{stats_dict['CI lower']:.2f}, {stats_dict['CI upper']:.2f}]")

# Eveneens voor value efficiency:
grouped_value = df.groupby("Partial")["value efficiency"]

results_value = {}

logging.info("Betrouwbaarheidsintervallen voor value efficiency per configuratie:")
for config, values in grouped_value:
    mean, lower, upper = compute_confidence_interval(values)
    results_value[config] = {"mean": mean, "CI lower": lower, "CI upper": upper}
    logging.info(f"VALUE_CI,Partial={config},Mean={mean:.4f},Lower={lower:.4f},Upper={upper:.4f}")

print("\nBetrouwbaarheidsintervallen voor value efficiency per configuratie:")
for config, stats_dict in results_value.items():
    print(
        f"{config}: Mean = {stats_dict['mean']:.2f}, CI = [{stats_dict['CI lower']:.2f}, {stats_dict['CI upper']:.2f}]")

# Add settled_count statistics
if 'settled_count' in df.columns:
    grouped_settled = df.groupby("Partial")["settled_count"]

    results_settled = {}

    logging.info("Betrouwbaarheidsintervallen voor aantal settled instructions per configuratie:")
    for config, values in grouped_settled:
        mean, lower, upper = compute_confidence_interval(values)
        results_settled[config] = {"mean": mean, "CI lower": lower, "CI upper": upper}
        logging.info(f"SETTLED_COUNT,Partial={config},Mean={mean:.4f},Lower={lower:.4f},Upper={upper:.4f}")

    print("\nBetrouwbaarheidsintervallen voor aantal settled instructions per configuratie:")
    for config, stats_dict in results_settled.items():
        print(
            f"{config}: Mean = {stats_dict['mean']:.2f}, CI = [{stats_dict['CI lower']:.2f}, {stats_dict['CI upper']:.2f}]")