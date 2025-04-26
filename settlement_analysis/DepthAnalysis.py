import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class DepthAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "depth_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        configs = self._aggregate_statistics()
        self._analyze_per_config(configs)
        self._analyze_comparative(configs)
        self._export_summary(configs)

    def _aggregate_statistics(self):
        config_data = defaultdict(lambda: {"depth_counts": defaultdict(float), "depth_status_counts": defaultdict(lambda: defaultdict(float))})

        for filename, data in self.suite.statistics.items():
            config_key = filename.split("_config")[1].split("_run")[0]
            if "depth_counts" in data and "depth_status_counts" in data:
                for depth, count in data["depth_counts"].items():
                    config_data[config_key]["depth_counts"][depth] += count
                for depth, statuses in data["depth_status_counts"].items():
                    for status, value in statuses.items():
                        config_data[config_key]["depth_status_counts"][depth][status] += value

        for config_key in config_data:
            run_count = len([fname for fname in self.suite.statistics if f"config{config_key}_" in fname])
            for depth in config_data[config_key]["depth_counts"]:
                config_data[config_key]["depth_counts"][depth] /= run_count
            for depth in config_data[config_key]["depth_status_counts"]:
                for status in config_data[config_key]["depth_status_counts"][depth]:
                    config_data[config_key]["depth_status_counts"][depth][status] /= run_count

        return config_data

    def _analyze_per_config(self, configs):
        for config_key, data in configs.items():
            self._plot_depth_distribution(data, config_key)
            self._plot_status_by_depth(data, config_key)
            self._plot_success_rate_by_depth(data, config_key)

    def _analyze_comparative(self, configs):
        self._plot_comparative_depth_distribution(configs)
        self._plot_comparative_success_rate(configs)
        self._plot_comparative_status_distribution(configs)
        self._plot_total_instruction_comparison(configs)
        self._plot_normalized_completion_rate(configs)
        self._plot_success_rate_heatmap(configs)

    def _export_summary(self, configs):
        summary = {}
        for config_key, data in configs.items():
            total_instructions = sum(float(count) for count in data["depth_counts"].values())
            depth_status = data["depth_status_counts"]
            total_successful = 0
            for depth, statuses in depth_status.items():
                total_successful += sum(statuses.get(s, 0) for s in ["Settled on time", "Settled late"])
            max_depth = max([int(d) for d in data["depth_counts"].keys()]) if data["depth_counts"] else 0
            summary[config_key] = {
                "total_instructions": total_instructions,
                "success_rate": (total_successful / total_instructions * 100) if total_instructions else 0,
                "max_depth": max_depth
            }
        with open(os.path.join(self.output_dir, "depth_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

    def _plot_depth_distribution(self, data, config_name):
        depth_counts = data["depth_counts"]
        depths = sorted(int(d) for d in depth_counts.keys())
        counts = [depth_counts[str(d)] for d in depths]
        plt.figure(figsize=(10, 6))
        plt.bar(depths, counts, color='skyblue')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Average Instructions')
        plt.title(f'{config_name} - Depth Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{config_name}_depth_distribution.png"))
        plt.close()

    def _plot_status_by_depth(self, data, config_name):
        status_groups = {
            "Settled on time": ["Settled on time"],
            "Settled late": ["Settled late"],
            "Cancelled": ["Cancelled due to timeout", "Cancelled due to partial settlement", "Cancelled due to error"],
            "In process": ["Matched", "Validated", "Pending", "Exists"]
        }
        depth_status_counts = data["depth_status_counts"]
        depths = sorted(int(d) for d in depth_status_counts.keys())
        grouped = {group: [] for group in status_groups}
        for depth in depths:
            for group, statuses in status_groups.items():
                grouped[group].append(sum(depth_status_counts.get(str(depth), {}).get(status, 0) for status in statuses))
        plt.figure(figsize=(12, 8))
        bottom = [0] * len(depths)
        colors = {"Settled on time": "green", "Settled late": "orange", "Cancelled": "red", "In process": "lightgray"}
        for group, counts in grouped.items():
            plt.bar(depths, counts, bottom=bottom, label=group, color=colors.get(group, 'grey'))
            bottom = [b + c for b, c in zip(bottom, counts)]
        plt.xlabel('Instruction Depth')
        plt.ylabel('Instructions')
        plt.title(f'{config_name} - Status by Depth')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{config_name}_status_by_depth.png"))
        plt.close()

    def _plot_success_rate_by_depth(self, data, config_name):
        depth_status_counts = data["depth_status_counts"]
        depths = sorted(int(d) for d in depth_status_counts.keys())
        success_rates = []
        for depth in depths:
            total = sum(depth_status_counts.get(str(depth), {}).values())
            successful = sum(depth_status_counts.get(str(depth), {}).get(status, 0) for status in ("Settled on time", "Settled late"))
            success_rates.append((successful / total) * 100 if total else 0)
        plt.figure(figsize=(10, 6))
        plt.plot(depths, success_rates, 'o-', color='blue')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Success Rate (%)')
        plt.title(f'{config_name} - Success Rate by Depth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{config_name}_success_rate.png"))
        plt.close()

    def _plot_comparative_depth_distribution(self, config_data):
        plt.figure(figsize=(12, 8))
        for config_key, data in sorted(config_data.items(), key=lambda x: int(x[0])):
            depths = sorted(int(d) for d in data["depth_counts"].keys())
            counts = [data["depth_counts"][str(d)] for d in depths]
            plt.plot(depths, counts, marker='o', label=f'config{config_key}')
        plt.xlabel('Depth Level')
        plt.ylabel('Average Number of Instructions')
        plt.title('Average Instruction Depth Distribution Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "depth_distribution_comparison.png"))
        plt.close()

    def _plot_comparative_success_rate(self, config_data):
        plt.figure(figsize=(12, 8))
        for config_key, data in sorted(config_data.items(), key=lambda x: int(x[0])):
            depth_status = data["depth_status_counts"]
            depths = sorted(int(d) for d in depth_status.keys())
            success_rates = []
            for depth in depths:
                statuses = depth_status.get(str(depth), {})
                total = sum(statuses.values())
                successful = sum(statuses.get(status, 0) for status in ["Settled on time", "Settled late"])
                success_rates.append((successful / total * 100) if total > 0 else 0)
            plt.plot(depths, success_rates, 'o-', linewidth=2, markersize=6, label=f'config{config_key}')
        plt.xlabel('Depth Level')
        plt.ylabel('Success Rate (%)')
        plt.title('Average Settlement Success Rate Comparison Across Configurations')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "success_rate_comparison.png"))
        plt.close()

    def _plot_comparative_status_distribution(self, config_data):
        status_groups = {
            "Settled on time": ["Settled on time"],
            "Settled late": ["Settled late"],
            "Cancelled": ["Cancelled due to timeout", "Cancelled due to partial settlement", "Cancelled due to error"],
            "In process": ["Matched", "Validated", "Pending", "Exists"]
        }
        config_names = []
        status_data = {group: [] for group in status_groups}
        for config_key, data in sorted(config_data.items(), key=lambda x: int(x[0])):
            config_names.append(f"config{config_key}")
            totals = {group: 0 for group in status_groups}
            total_instructions = 0
            for depth, statuses in data["depth_status_counts"].items():
                for group, keys in status_groups.items():
                    count = sum(statuses.get(k, 0) for k in keys)
                    totals[group] += count
                    total_instructions += count
            for group in status_groups:
                status_data[group].append((totals[group] / total_instructions) * 100 if total_instructions else 0)

        plt.figure(figsize=(12, 8))
        bottom = np.zeros(len(config_names))
        colors = {"Settled on time": "green", "Settled late": "orange", "Cancelled": "red", "In process": "lightgray"}
        for group, values in status_data.items():
            plt.bar(config_names, values, bottom=bottom, label=group, color=colors.get(group, 'gray'))
            bottom += np.array(values)
        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Instructions')
        plt.title('Overall Status Distribution Across Configurations')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "status_distribution_comparison.png"))
        plt.close()

    def _plot_total_instruction_comparison(self, config_data):
        config_names = []
        totals = []
        for config_key, data in sorted(config_data.items(), key=lambda x: int(x[0])):
            config_names.append(f"config{config_key}")
            totals.append(sum(data["depth_counts"].values()))
        plt.figure(figsize=(12, 8))
        bars = plt.bar(config_names, totals, color='steelblue')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1, f'{height:.1f}', ha='center', va='bottom')
        plt.xlabel('Configuration')
        plt.ylabel('Average Number of Instructions')
        plt.title('Total Instructions Processed per Configuration')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "total_instructions_comparison.png"))
        plt.close()

    def _plot_normalized_completion_rate(self, config_data):
        plt.figure(figsize=(12, 8))
        for config_key, data in sorted(config_data.items(), key=lambda x: int(x[0])):
            rates = []
            depth_status = data["depth_status_counts"]
            max_depth = max(int(d) for d in depth_status.keys())
            for depth in range(max_depth + 1):
                statuses = depth_status.get(str(depth), {})
                total = sum(statuses.values())
                successful = sum(statuses.get(s, 0) for s in ["Settled on time", "Settled late"])
                rate = (successful / total * 100) if total else 0
                rates.append(rate)
            if rates and rates[0] > 0:
                normalized = [r / rates[0] * 100 for r in rates]
                plt.plot(range(len(normalized)), normalized, 'o-', label=f"config{config_key}")
        plt.xlabel('Depth Level')
        plt.ylabel('Completion Rate (% of Depth 0)')
        plt.title('Normalized Settlement Completion Rate by Depth')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "depth_completion_rates.png"))
        plt.close()

    def _plot_success_rate_heatmap(self, config_data):
        import seaborn as sns
        import pandas as pd
        rows = []
        for config_key, data in sorted(config_data.items(), key=lambda x: int(x[0])):
            row = {}
            for depth, statuses in data["depth_status_counts"].items():
                total = sum(statuses.values())
                successful = sum(statuses.get(s, 0) for s in ["Settled on time", "Settled late"])
                rate = (successful / total * 100) if total else 0
                row[int(depth)] = rate
            rows.append(pd.Series(row, name=f"config{config_key}"))
        df = pd.DataFrame(rows).fillna(0)
        plt.figure(figsize=(14, 10))
        sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100)
        plt.title("Settlement Success Rate Heatmap")
        plt.xlabel("Depth Level")
        plt.ylabel("Configuration")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "success_rate_heatmap.png"))
        plt.close()