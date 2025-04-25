# DEPTH VISUALIZATION MODULE - RECOMPILED
import os
import glob
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class SettlementAnalyzer:
    def __init__(self, results_dir="./"):
        self.results_dir = results_dir
        self.config_runs = defaultdict(dict)
        self.configs = {}

        result_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        if not result_files:
            print(f"No result JSON files found in {self.results_dir}")
            return

        for file_path in result_files:
            basename = os.path.basename(file_path)
            match = re.match(r"results_\w+_config(\d+)_run(\d+)\.json", basename)
            if match:
                config_name = f"config{match.group(1)}"
                run_name = f"run{match.group(2)}"
                data = self._load_results_from_json(file_path)
                if data:
                    self.config_runs[config_name][run_name] = data
            else:
                print(f"Skipping file with unexpected name format: {basename}")

        for config_name, runs in self.config_runs.items():
            if runs:
                self.configs[config_name] = self._aggregate_runs(runs)

    def _load_results_from_json(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if "depth_counts" in data and "depth_status_counts" in data:
                return {
                    "depth_counts": data.get("depth_counts", {}),
                    "depth_status_counts": data.get("depth_status_counts", {})
                }
            else:
                print(f"Warning: Missing depth data in {file_path}")
                return None
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _aggregate_runs(self, runs):
        if not runs:
            return {}

        aggregated = {
            "depth_counts": defaultdict(float),
            "depth_status_counts": defaultdict(lambda: defaultdict(float))
        }

        all_depths = set()
        all_statuses = set()

        for run_data in runs.values():
            if "depth_counts" in run_data:
                all_depths.update(run_data["depth_counts"].keys())
            if "depth_status_counts" in run_data:
                for depth, status_data in run_data["depth_status_counts"].items():
                    all_depths.add(depth)
                    all_statuses.update(status_data.keys())

        for depth in all_depths:
            depth_counts = []
            for run_data in runs.values():
                depth_counts.append(run_data.get("depth_counts", {}).get(depth, 0))
            aggregated["depth_counts"][depth] = sum(depth_counts) / len(runs)

        for depth in all_depths:
            for status in all_statuses:
                status_counts = []
                for run_data in runs.values():
                    depth_status = run_data.get("depth_status_counts", {}).get(depth, {})
                    status_counts.append(depth_status.get(status, 0))
                if any(status_counts):
                    aggregated["depth_status_counts"][depth][status] = sum(status_counts) / len(runs)

        return aggregated

    def analyze_single_config(self, config_name, output_dir="visualizations/"):
        if config_name not in self.configs:
            print(f"Configuration {config_name} not found")
            return

        data = self.configs[config_name]
        os.makedirs(output_dir, exist_ok=True)

        self._depth_distribution(data, config_name, os.path.join(output_dir, f"{config_name}_depth_distribution.png"))
        self._status_by_depth(data, config_name, os.path.join(output_dir, f"{config_name}_status_by_depth.png"))
        self._success_rate_by_depth(data, config_name, os.path.join(output_dir, f"{config_name}_success_rate.png"))

    def _depth_distribution(self, data, config_name, output_file):
        depth_counts = data["depth_counts"]
        depths = sorted([int(d) for d in depth_counts.keys()])
        counts = [depth_counts[str(d)] for d in depths]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(depths, counts, color='skyblue')

        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{count:.1f}", ha='center', va='bottom')

        plt.xlabel('Instruction Depth')
        plt.ylabel('Average Number of Instructions')
        plt.title(f'{config_name}: Avg. Distribution by Depth')
        plt.xticks(depths)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _status_by_depth(self, data, config_name, output_file):
        status_groups = {
            "Settled on time": ["Settled on time"],
            "Settled late": ["Settled late"],
            "Cancelled": ["Cancelled due to partial settlement", "Cancelled due to error"],
            "In process": ["Matched", "Validated", "Pending", "Exists"]
        }

        group_colors = {
            "Settled on time": "green",
            "Settled late": "orange",
            "Cancelled": "red",
            "In process": "lightgray"
        }

        depth_status_counts = data["depth_status_counts"]
        depths = sorted([int(d) for d in depth_status_counts.keys()])

        group_data = {group: [] for group in status_groups.keys()}

        for depth in depths:
            for group, statuses in status_groups.items():
                count = sum(depth_status_counts.get(str(depth), {}).get(status, 0) for status in statuses)
                group_data[group].append(count)

        plt.figure(figsize=(12, 8))
        bottom = [0] * len(depths)

        for group, counts in group_data.items():
            color = group_colors.get(group, 'gray')
            plt.bar(depths, counts, bottom=bottom, label=group, color=color)
            bottom = [b + c for b, c in zip(bottom, counts)]

        plt.xlabel('Instruction Depth')
        plt.ylabel('Average Number of Instructions')
        plt.title(f'{config_name}: Avg. Outcomes by Depth')
        plt.xticks(depths)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _success_rate_by_depth(self, data, config_name, output_file):
        depth_status_counts = data["depth_status_counts"]
        depths = sorted([int(d) for d in depth_status_counts.keys()])

        success_rates = []
        for depth in depths:
            statuses = depth_status_counts.get(str(depth), {})
            total = sum(statuses.values())
            successful = sum(statuses.get(status, 0) for status in ["Settled on time", "Settled late"])
            success_rate = (successful / total * 100) if total > 0 else 0
            success_rates.append(success_rate)

        plt.figure(figsize=(10, 6))
        plt.plot(depths, success_rates, 'o-', linewidth=2, markersize=8, color='blue')

        for i, rate in enumerate(success_rates):
            plt.annotate(f"{rate:.1f}%", (depths[i], success_rates[i]), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.xlabel('Depth Level')
        plt.ylabel('Success Rate (%)')
        plt.title(f'{config_name}: Settlement Success Rate by Depth')
        plt.xticks(depths)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, max(success_rates) * 1.2 if success_rates else 100)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_configs_depth_distribution(self, config_names, output_file="comparison_depth_distribution.png"):
        config_names = sorted(config_names, key=lambda x: int(re.search(r'\d+', x).group()))
        plt.figure(figsize=(12, 8))

        for config_name in config_names:
            if config_name not in self.configs:
                continue
            data = self.configs[config_name]
            depth_counts = data["depth_counts"]
            depths = sorted([int(d) for d in depth_counts.keys()])
            counts = [depth_counts[str(d)] for d in depths]
            plt.plot(depths, counts, marker='o', label=config_name)

        plt.xlabel('Instruction Depth')
        plt.ylabel('Average Number of Instructions')
        plt.title('Comparison of Depth Distributions')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_configs_success_rate(self, config_names, output_file="comparison_success_rate.png"):
        config_names = sorted(config_names, key=lambda x: int(re.search(r'\d+', x).group()))
        plt.figure(figsize=(12, 8))

        for config_name in config_names:
            if config_name not in self.configs:
                continue
            data = self.configs[config_name]
            depth_status_counts = data["depth_status_counts"]
            depths = sorted([int(d) for d in depth_status_counts.keys()])

            success_rates = []
            for depth in depths:
                statuses = depth_status_counts.get(str(depth), {})
                total = sum(statuses.values())
                successful = sum(statuses.get(status, 0) for status in ["Settled on time", "Settled late"])
                success_rate = (successful / total * 100) if total > 0 else 0
                success_rates.append(success_rate)

            plt.plot(depths, success_rates, marker='o', label=config_name)

        plt.xlabel('Instruction Depth')
        plt.ylabel('Success Rate (%)')
        plt.title('Comparison of Success Rates')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_depth_completion_rates(self, output_file="comparison_depth_completion_rates.png"):
        """
        Compare how settlement completion rates drop with increasing depth across configurations.
        """
        if not self.configs:
            print("No configuration data loaded.")
            return

        max_depth = 0
        for data in self.configs.values():
            depths = [int(d) for d in data["depth_counts"].keys()]
            if depths:
                max_depth = max(max_depth, max(depths))

        completion_rates = {}

        for config_name, data in self.configs.items():
            depth_status_counts = data["depth_status_counts"]
            rates = []

            for depth in range(max_depth + 1):
                str_depth = str(depth)
                if str_depth in depth_status_counts:
                    statuses = depth_status_counts[str_depth]
                    total = sum(statuses.values())
                    successful = sum(statuses.get(status, 0) for status in ["Settled on time", "Settled late"])
                    rate = (successful / total * 100) if total > 0 else 0
                    rates.append(rate)
                else:
                    rates.append(0)

            completion_rates[config_name] = rates

        plt.figure(figsize=(12, 8))

        for config_name, rates in completion_rates.items():
            if rates and rates[0] > 0:
                normalized_rates = [r / rates[0] * 100 for r in rates]
                plt.plot(range(len(normalized_rates)), normalized_rates, 'o-', linewidth=2, markersize=6,
                         label=config_name)

        plt.xlabel('Depth Level')
        plt.ylabel('Completion Rate (% of Depth 0)')
        plt.title('Normalized Settlement Completion Rate by Depth')
        plt.xticks(range(max_depth + 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_settlement_times_by_instruction_count(self,
                                                      output_file="comparison_settlement_times_instruction_count.png"):
        """
        Compare the proportion of on-time vs late settlements across configurations based on instruction count.
        """
        config_names = sorted(self.configs.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        on_time_percentages = []
        late_percentages = []

        for config_name in config_names:
            data = self.configs[config_name]
            depth_status_counts = data["depth_status_counts"]

            on_time_total = 0
            late_total = 0

            for statuses in depth_status_counts.values():
                on_time_total += statuses.get("Settled on time", 0)
                late_total += statuses.get("Settled late", 0)

            total_settled = on_time_total + late_total
            if total_settled > 0:
                on_time_percentages.append((on_time_total / total_settled) * 100)
                late_percentages.append((late_total / total_settled) * 100)
            else:
                on_time_percentages.append(0)
                late_percentages.append(0)

        # Plotting
        plt.figure(figsize=(12, 8))
        bar_width = 0.8
        indices = np.arange(len(config_names))

        plt.bar(indices, on_time_percentages, label="Settled on time", color="green")
        plt.bar(indices, late_percentages, bottom=on_time_percentages, label="Settled late", color="orange")

        plt.xlabel("Configuration")
        plt.ylabel("Percentage of Settlements (Instruction Count)")
        plt.title("On-Time vs Late Settlements (Instruction Count) Across Configurations")
        plt.xticks(indices, config_names, rotation=45, ha="right")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_settlement_times_by_value_from_json(self, output_file="comparison_settlement_times_value.png"):
        """
        Compare the proportion of on-time vs late settlements across configurations based only on JSON statistics.
        """
        if not self.config_runs:
            print("No configuration run data loaded.")
            return

        config_names = sorted(self.config_runs.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        on_time_percentages = []
        late_percentages = []

        for config_name in config_names:
            runs = self.config_runs[config_name]
            on_time_amounts = []
            late_amounts = []

            for run_name, run_data in runs.items():
                on_time = run_data.get('settled_on_time_amount', 0)
                late = run_data.get('settled_late_amount', 0)

                on_time_amounts.append(on_time)
                late_amounts.append(late)

            if on_time_amounts or late_amounts:
                avg_on_time = sum(on_time_amounts) / len(on_time_amounts)
                avg_late = sum(late_amounts) / len(late_amounts)
                total = avg_on_time + avg_late

                if total > 0:
                    on_time_percentages.append((avg_on_time / total) * 100)
                    late_percentages.append((avg_late / total) * 100)
                else:
                    on_time_percentages.append(0)
                    late_percentages.append(0)
            else:
                on_time_percentages.append(0)
                late_percentages.append(0)

        # Plotting
        plt.figure(figsize=(12, 8))
        bar_width = 0.8
        indices = np.arange(len(config_names))

        plt.bar(indices, on_time_percentages, label="Settled Value On Time", color="green")
        plt.bar(indices, late_percentages, bottom=on_time_percentages, label="Settled Value Late", color="orange")

        plt.xlabel("Configuration")
        plt.ylabel("Percentage of Settled Value")
        plt.title("On-Time vs Late Settlements (Value) Across Configurations (from JSON Statistics)")
        plt.xticks(indices, config_names, rotation=45, ha="right")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_status_distributions_grouped(self, output_file="comparison_status_distribution_grouped.png"):
        """
        Compare the overall grouped status distribution (on time, late, cancelled, in process) across configurations.
        """
        if not self.configs:
            print("No configuration data loaded.")
            return

        status_groups = {
            "Settled on time": ["Settled on time"],
            "Settled late": ["Settled late"],
            "Cancelled": ["Cancelled due to timeout", "Cancelled due to partial settlement", "Cancelled due to error"],
            "In process": ["Matched", "Validated", "Pending", "Exists"]
        }

        group_colors = {
            "Settled on time": "green",
            "Settled late": "orange",
            "Cancelled": "red",
            "In process": "lightgray"
        }

        config_names = sorted(self.configs.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        status_data = {group: [] for group in status_groups.keys()}

        for config_name in config_names:
            data = self.configs[config_name]
            depth_status_counts = data["depth_status_counts"]

            total_instructions = 0
            group_counts = {group: 0 for group in status_groups.keys()}

            for depth, statuses in depth_status_counts.items():
                for group, status_list in status_groups.items():
                    group_counts[group] += sum(statuses.get(status, 0) for status in status_list)
                total_instructions += sum(statuses.values())

            if total_instructions > 0:
                for group in status_groups.keys():
                    status_data[group].append((group_counts[group] / total_instructions) * 100)
            else:
                for group in status_groups.keys():
                    status_data[group].append(0)

        # Plot
        plt.figure(figsize=(12, 8))
        bar_width = 0.8
        bottom = np.zeros(len(config_names))

        for group, percentages in status_data.items():
            plt.bar(config_names, percentages, bar_width, bottom=bottom,
                    label=group, color=group_colors.get(group, 'gray'))
            bottom += np.array(percentages)

        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Instructions')
        plt.title('Grouped Status Distribution Across Configurations')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def generate_summary_table(self, config_names):
        summary_data = []
        config_names = sorted(config_names, key=lambda x: int(re.search(r'\d+', x).group()))

        for config_name in config_names:
            if config_name not in self.configs:
                continue
            data = self.configs[config_name]
            summary_data.append({
                "Config": config_name,
                "Avg Tree Depth": data.get("avg_tree_depth", 0),
                "Memory Usage (MB)": data.get("memory_usage_mb", 0),
                "Runtime (s)": data.get("runtime_seconds", 0)
            })

        df_summary = pd.DataFrame(summary_data)
        return df_summary

    def plot_avg_tree_depth_vs_max_depth(self, config_names, output_file="avg_tree_depth_vs_max_depth.png"):
        config_names = sorted(config_names, key=lambda x: int(re.search(r'\d+', x).group()))
        avg_depths = []

        for config_name in config_names:
            if config_name not in self.configs:
                continue
            avg_depth = self.configs[config_name].get("avg_tree_depth", 0)
            avg_depths.append(avg_depth)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(avg_depths) + 1), avg_depths, marker='o')
        plt.xlabel("Configuration")
        plt.ylabel("Average Tree Depth")
        plt.title("Average Tree Depth vs Max Configurations")
        plt.grid(True)
        plt.xticks(range(1, len(avg_depths) + 1), config_names, rotation=45)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def plot_memory_usage_vs_depth(self, config_names, output_file="memory_usage_vs_depth.png"):
        config_names = sorted(config_names, key=lambda x: int(re.search(r'\d+', x).group()))
        memory_usages = []

        for config_name in config_names:
            if config_name not in self.configs:
                continue
            memory = self.configs[config_name].get("memory_usage_mb", 0)
            memory_usages.append(memory)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(memory_usages) + 1), memory_usages, marker='o', color='orange')
        plt.xlabel("Configuration")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage vs Max Configurations")
        plt.grid(True)
        plt.xticks(range(1, len(memory_usages) + 1), config_names, rotation=45)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def plot_runtime_scaling(self, config_names, output_file="runtime_scaling.png"):
        config_names = sorted(config_names, key=lambda x: int(re.search(r'\d+', x).group()))
        runtimes = []

        for config_name in config_names:
            if config_name not in self.configs:
                continue
            runtime = self.configs[config_name].get("runtime_seconds", 0)
            runtimes.append(runtime)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(runtimes) + 1), runtimes, marker='o', color='red')
        plt.xlabel("Configuration")
        plt.ylabel("Runtime (seconds)")
        plt.title("Runtime Scaling Across Configurations")
        plt.grid(True)
        plt.xticks(range(1, len(runtimes) + 1), config_names, rotation=45)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def plot_elbow_analysis(self, config_names, output_file="elbow_analysis.png"):
        config_names = sorted(config_names, key=lambda x: int(re.search(r'\d+', x).group()))
        runtimes = []
        avg_depths = []

        for config_name in config_names:
            if config_name not in self.configs:
                continue
            runtimes.append(self.configs[config_name].get("runtime_seconds", 0))
            avg_depths.append(self.configs[config_name].get("avg_tree_depth", 0))

        plt.figure(figsize=(10, 6))
        plt.plot(avg_depths, runtimes, 'o-')
        plt.xlabel("Average Tree Depth")
        plt.ylabel("Runtime (seconds)")
        plt.title("Elbow Analysis: Runtime vs Tree Depth")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)

    def create_heatmap_comparison(self, output_file="comparison_success_rate_heatmap.png"):
        """
        Create a heatmap comparing settlement success rates at each depth across configurations.
        """
        if not self.configs:
            print("No configuration data loaded.")
            return

        max_depth = 0
        for data in self.configs.values():
            depths = [int(d) for d in data["depth_counts"].keys()]
            if depths:
                max_depth = max(max_depth, max(depths))

        config_names = sorted(self.configs.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        success_rates = np.zeros((len(config_names), max_depth + 1))

        for i, config_name in enumerate(config_names):
            data = self.configs[config_name]
            depth_status_counts = data["depth_status_counts"]

            for depth in range(max_depth + 1):
                str_depth = str(depth)
                if str_depth in depth_status_counts:
                    statuses = depth_status_counts[str_depth]
                    total = sum(statuses.values())
                    successful = sum(statuses.get(status, 0) for status in ["Settled on time", "Settled late"])

                    if total > 0:
                        rate = (successful / total) * 100
                        success_rates[i, depth] = rate
                    else:
                        success_rates[i, depth] = 0

        # Plot heatmap
        plt.figure(figsize=(max(10, max_depth * 0.8), len(config_names) * 0.6))
        ax = plt.gca()

        im = ax.imshow(success_rates, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        cbar = plt.colorbar(im)
        cbar.set_label('Success Rate (%)')

        ax.set_xticks(np.arange(max_depth + 1))
        ax.set_yticks(np.arange(len(config_names)))
        ax.set_xticklabels(range(max_depth + 1))
        ax.set_yticklabels(config_names)

        # Add labels inside the heatmap cells
        for i in range(len(config_names)):
            for j in range(max_depth + 1):
                rate = success_rates[i, j]
                text_color = "black" if 30 < rate < 70 else "white"
                ax.text(j, i, f"{rate:.1f}%", ha="center", va="center", color=text_color, fontsize=8)

        plt.xlabel('Depth Level')
        plt.title('Settlement Success Rate Heatmap by Configuration and Depth')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_total_instructions(self, output_file="comparison_total_instructions.png"):
        """
        Compare the total number of instructions processed in each configuration.
        """
        if not self.configs:
            print("No configuration data loaded.")
            return

        config_names = sorted(self.configs.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        total_counts = []

        for config_name in config_names:
            data = self.configs[config_name]
            depth_counts = data["depth_counts"]

            total_instructions = sum(float(count) for count in depth_counts.values())
            total_counts.append(total_instructions)

        plt.figure(figsize=(12, 8))
        bars = plt.bar(config_names, total_counts, color='steelblue')

        # Add labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                     f"{height:.1f}", ha='center', va='bottom')

        plt.xlabel('Configuration')
        plt.ylabel('Total Instructions')
        plt.title('Total Instructions Processed Across Configurations')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_total_settled_value_from_json(self, output_file="comparison_total_settled_value.png"):
        """
        Compare the total settled value (EUR) across configurations based only on the JSON statistics.
        """
        if not self.config_runs:
            print("No configuration run data loaded.")
            return

        config_names = sorted(self.config_runs.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        settled_values = []

        for config_name in config_names:
            runs = self.config_runs[config_name]
            settled_amounts = []

            for run_name, run_data in runs.items():
                on_time = run_data.get('settled_on_time_amount', 0)
                late = run_data.get('settled_late_amount', 0)
                total = on_time + late
                settled_amounts.append(total)

            if settled_amounts:
                avg_settled = sum(settled_amounts) / len(settled_amounts)
            else:
                avg_settled = 0

            settled_values.append(avg_settled)

        plt.figure(figsize=(12, 8))
        bars = plt.bar(config_names, settled_values, color='seagreen')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                     f"{height / 1e6:.1f}M", ha='center', va='bottom')

        plt.xlabel('Configuration')
        plt.ylabel('Average Settled Value (EUR)')
        plt.title('Total Settled Value Across Configurations (from JSON Statistics)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def analyze_rtp_vs_batch_from_logs(self, log_folder, output_file="comparison_rtp_vs_batch.png"):
        """
        Analyze settlement timing patterns (RTP vs Batch) across configurations by reading event logs.

        Args:
            log_folder (str): Path to the folder containing simulation .jsonocel logs
            output_file (str): Path to save the output plot
        """
        if not os.path.exists(log_folder):
            print(f"Log folder '{log_folder}' does not exist.")
            return

        log_files = [f for f in os.listdir(log_folder) if f.endswith(".jsonocel")]
        if not log_files:
            print(f"No log files found in '{log_folder}'.")
            return

        config_data = defaultdict(lambda: {"rtp": 0, "batch": 0})

        for log_file in log_files:
            parts = log_file.split("_")
            if len(parts) >= 3 and parts[0] == "simulation" and parts[1].startswith("config"):
                try:
                    config_num = int(parts[1].replace("config", ""))
                    config_name = f"config{config_num}"
                except:
                    continue
            else:
                continue

            with open(os.path.join(log_folder, log_file), 'r') as f:
                log_data = json.load(f)

            events = log_data.get("events", [])

            for event in events:
                event_type = event.get("type", "")
                timestamp_str = event.get("time", "")

                if event_type in ["Settled On Time", "Settled Late"] and timestamp_str:
                    try:
                        from datetime import datetime, time

                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        event_time = timestamp.time()

                        if time(1, 30) <= event_time <= time(19, 30):
                            config_data[config_name]["rtp"] += 1
                        elif event_time >= time(22, 0):
                            config_data[config_name]["batch"] += 1
                    except Exception as e:
                        print(f"Error parsing timestamp {timestamp_str}: {e}")

        if not config_data:
            print("No RTP/Batch settlement data extracted.")
            return

        config_names = sorted(config_data.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        rtp_counts = [config_data[c]["rtp"] for c in config_names]
        batch_counts = [config_data[c]["batch"] for c in config_names]

        total_counts = [rtp + batch for rtp, batch in zip(rtp_counts, batch_counts)]
        rtp_percentages = [(rtp / total) * 100 if total > 0 else 0 for rtp, total in zip(rtp_counts, total_counts)]
        batch_percentages = [(batch / total) * 100 if total > 0 else 0 for batch, total in
                             zip(batch_counts, total_counts)]

        plt.figure(figsize=(12, 8))

        plt.bar(config_names, rtp_percentages, label='Real-Time Processing', color='skyblue')
        plt.bar(config_names, batch_percentages, bottom=rtp_percentages, label='Batch Processing', color='salmon')

        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Settlements')
        plt.title('RTP vs Batch Settlements by Configuration')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def analyze_lateness_from_depth_stats(self, output_file="comparison_late_settlement_percentage.png"):
        """
        Analyze settlement lateness patterns using depth statistics data.
        """
        if not self.configs:
            print("No configuration data loaded.")
            return

        config_names = sorted(self.configs.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        late_percentages = []

        for config_name in config_names:
            data = self.configs[config_name]
            depth_status_counts = data["depth_status_counts"]

            on_time = 0
            late = 0

            for statuses in depth_status_counts.values():
                on_time += statuses.get("Settled on time", 0)
                late += statuses.get("Settled late", 0)

            total = on_time + late
            if total > 0:
                late_percentage = (late / total) * 100
            else:
                late_percentage = 0

            late_percentages.append(late_percentage)

        # Bar plot: late settlement percentage
        plt.figure(figsize=(12, 8))
        bars = plt.bar(config_names, late_percentages, color='orangered')

        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                     f"{height:.1f}%", ha='center', va='bottom')

        plt.xlabel('Configuration')
        plt.ylabel('Late Settlement Percentage (%)')
        plt.title('Late Settlement Percentage Across Configurations')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def analyze_runtime_from_json(self, output_file="comparison_runtime.png"):
        """
        Analyze average simulation runtime across configurations based on JSON statistics.
        """
        if not self.config_runs:
            print("No configuration run data loaded.")
            return

        config_names = sorted(self.config_runs.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        avg_runtimes = []

        for config_name in config_names:
            runs = self.config_runs[config_name]
            runtimes = []

            for run_name, run_data in runs.items():
                runtime = run_data.get('runtime_seconds', None)
                if runtime is not None:
                    runtimes.append(runtime)

            if runtimes:
                avg_runtime = sum(runtimes) / len(runtimes)
            else:
                avg_runtime = 0

            avg_runtimes.append(avg_runtime)

        plt.figure(figsize=(12, 8))
        bars = plt.bar(config_names, avg_runtimes, color='royalblue')

        # Add runtime labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                     f"{height:.1f}s", ha='center', va='bottom')

        plt.xlabel('Configuration')
        plt.ylabel('Average Runtime (seconds)')
        plt.title('Average Simulation Runtime Across Configurations')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def analyze_lateness_hours(self, log_folder, output_file="lateness_distribution_hours.png"):
        """
        Analyze lateness duration (in hours) based on settlement events from log files.

        Args:
            log_folder (str): Path to the folder containing simulation .jsonocel logs
            output_file (str): Path to save the lateness distribution plot
        """
        if not os.path.exists(log_folder):
            print(f"Log folder '{log_folder}' does not exist.")
            return

        log_files = [f for f in os.listdir(log_folder) if f.endswith(".jsonocel")]
        if not log_files:
            print(f"No log files found in '{log_folder}'.")
            return

        lateness_hours = []

        for log_file in log_files:
            with open(os.path.join(log_folder, log_file), 'r') as f:
                log_data = json.load(f)

            events = log_data.get("events", [])

            for event in events:
                event_type = event.get("type", "")
                if event_type == "Settled Late":
                    timestamp_str = event.get("time", "")
                    deadline_str = event.get("event_attributes", {}).get("deadline", "")

                    if timestamp_str and deadline_str:
                        try:
                            from datetime import datetime

                            settlement_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            deadline_time = datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))

                            lateness = (settlement_time - deadline_time).total_seconds() / 3600.0
                            if lateness > 0:
                                lateness_hours.append(lateness)

                        except Exception as e:
                            print(f"Error parsing timestamps in {log_file}: {e}")

        if not lateness_hours:
            print("No lateness data found.")
            return

        # Plotting
        plt.figure(figsize=(12, 8))
        bins = [0, 1, 2, 4, 6, 12, 24, 48, 72, 120]  # Define bins in hours
        plt.hist(lateness_hours, bins=bins, edgecolor='black', alpha=0.7)

        plt.xlabel('Hours Late')
        plt.ylabel('Number of Settlements')
        plt.title('Distribution of Lateness Duration (in Hours)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def analyze_confidence_intervals_from_json(self, output_file="comparison_confidence_intervals.png",
                                               confidence_level=0.95):
        """
        Analyze and plot confidence intervals for instruction and value efficiency based on JSON statistics.

        Args:
            output_file (str): Path to save the confidence interval plot
            confidence_level (float): Confidence level for the intervals (default: 0.95)
        """
        if not self.config_runs:
            print("No configuration run data loaded.")
            return

        from scipy import stats

        config_names = sorted(self.config_runs.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        instruction_means = []
        instruction_errors = []
        value_means = []
        value_errors = []

        for config_name in config_names:
            runs = self.config_runs[config_name]
            instruction_eff = []
            value_eff = []

            for run_name, run_data in runs.items():
                instruction = run_data.get('instruction_efficiency', None)
                value = run_data.get('value_efficiency', None)

                if instruction is not None:
                    instruction_eff.append(instruction)
                if value is not None:
                    value_eff.append(value)

            # Compute means and confidence intervals
            if instruction_eff:
                mean_instr = sum(instruction_eff) / len(instruction_eff)
                ci_instr = stats.t.interval(confidence_level, len(instruction_eff) - 1, loc=mean_instr,
                                            scale=stats.sem(instruction_eff))
                error_instr = (ci_instr[1] - ci_instr[0]) / 2
            else:
                mean_instr = 0
                error_instr = 0

            if value_eff:
                mean_value = sum(value_eff) / len(value_eff)
                ci_value = stats.t.interval(confidence_level, len(value_eff) - 1, loc=mean_value,
                                            scale=stats.sem(value_eff))
                error_value = (ci_value[1] - ci_value[0]) / 2
            else:
                mean_value = 0
                error_value = 0

            instruction_means.append(mean_instr)
            instruction_errors.append(error_instr)
            value_means.append(mean_value)
            value_errors.append(error_value)

        # Plot
        plt.figure(figsize=(14, 8))
        x = np.arange(len(config_names))
        width = 0.35

        plt.bar(x - width / 2, instruction_means, width, yerr=instruction_errors, capsize=5,
                label="Instruction Efficiency", color="skyblue")
        plt.bar(x + width / 2, value_means, width, yerr=value_errors, capsize=5, label="Value Efficiency",
                color="lightgreen")

        plt.xticks(x, config_names, rotation=45, ha='right')
        plt.xlabel('Configuration')
        plt.ylabel('Efficiency (%)')
        plt.title('Efficiency Metrics with Confidence Intervals')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def analyze_all(self, output_base="partial_allowance_visualisations/", log_folder="partial_allowance_logs/"):
        """
        Run all analysis functions and generate visualizations and statistics.

        Args:
            output_base (str): Base directory for all output
            log_folder (str): Directory containing event log files (.jsonocel)
        """
        os.makedirs(output_base, exist_ok=True)

        # 1. Compare settlement timing (count and value)
        self.compare_settlement_times_by_instruction_count(
            output_file=os.path.join(output_base, "comparison_settlement_times_instruction_count.png")
        )
        self.compare_settlement_times_by_value_from_json(
            output_file=os.path.join(output_base, "comparison_settlement_times_value.png")
        )

        # 2. Completion rate by depth
        self.compare_depth_completion_rates(
            output_file=os.path.join(output_base, "comparison_depth_completion_rates.png")
        )

        # 3. Status distributions
        self.compare_status_distributions_grouped(
            output_file=os.path.join(output_base, "comparison_status_distribution_grouped.png")
        )

        # 4. Heatmap success rates
        self.create_heatmap_comparison(
            output_file=os.path.join(output_base, "comparison_success_rate_heatmap.png")
        )

        # 5. Total instructions and total value
        self.compare_total_instructions(
            output_file=os.path.join(output_base, "comparison_total_instructions.png")
        )
        self.compare_total_settled_value_from_json(
            output_file=os.path.join(output_base, "comparison_total_settled_value.png")
        )

        # 6. Analyze lateness patterns
        self.analyze_lateness_from_depth_stats(
            output_file=os.path.join(output_base, "comparison_late_settlement_percentage.png")
        )

        # 7. Analyze runtime durations
        self.analyze_runtime_from_json(
            output_file=os.path.join(output_base, "comparison_runtime.png")
        )

        # 8. Analyze lateness hours from event logs
        self.analyze_lateness_hours(
            log_folder=log_folder,
            output_file=os.path.join(output_base, "lateness_distribution_hours.png")
        )

        # 9. Analyze RTP vs Batch settlement behavior
        self.analyze_rtp_vs_batch_from_logs(
            log_folder=log_folder,
            output_file=os.path.join(output_base, "comparison_rtp_vs_batch.png")
        )

        # 10. Analyze confidence intervals
        self.analyze_confidence_intervals_from_json(
            output_file=os.path.join(output_base, "comparison_confidence_intervals.png")
        )

        print(f"[✓] Full analysis completed. Results saved to {output_base}")













