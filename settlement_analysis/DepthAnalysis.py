import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

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

    def _plot_depth_distribution(self):
        plt.figure(figsize=(14, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.depth_data)))

        for idx, (config, counts) in enumerate(sorted(self.depth_data.items(), key=lambda x: int(x[0].replace('config', '')))):
            plt.bar(range(len(counts)), counts, color=colors[idx % len(colors)], alpha=0.7, label=config)

        plt.xlabel('Depth Level', fontsize=14)
        plt.ylabel('Average Instructions', fontsize=14)
        plt.title('Depth Distribution per Configuration', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Configuration', fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "depth_distribution.png"), dpi=300)
        plt.close()

    def _plot_status_by_depth(self, status_data):
        plt.figure(figsize=(14, 8))
        statuses = ['Settled on time', 'Settled late', 'Cancelled', 'In process']
        colors = {'Settled on time': 'green', 'Settled late': 'orange', 'Cancelled': 'red', 'In process': 'lightgray'}

        depths = sorted(status_data.keys())
        bottom = np.zeros(len(depths))

        for status in statuses:
            counts = [status_data[d].get(status, 0) for d in depths]
            plt.bar(depths, counts, bottom=bottom, label=status, color=colors.get(status, 'gray'))
            bottom += np.array(counts)

        plt.xlabel('Instruction Depth', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Status Distribution by Depth', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Status', fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "status_by_depth.png"), dpi=300)
        plt.close()

    def _plot_success_rate_by_depth(self, success_data):
        plt.figure(figsize=(14, 8))
        depths = sorted(success_data.keys())
        success_rates = [success_data[d] for d in depths]

        plt.plot(depths, success_rates, 'o-', markersize=6, linewidth=2.5, color='blue')
        plt.xlabel('Instruction Depth', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.title('Success Rate by Depth', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "success_rate_by_depth.png"), dpi=300)
        plt.close()

    def _plot_comparative_depth_distribution(self):
        plt.figure(figsize=(14, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.depth_data)))

        for idx, (config, counts) in enumerate(sorted(self.depth_data.items(), key=lambda x: int(x[0].replace('config', '')))):
            plt.plot(range(len(counts)), counts, '^-', markersize=6, linewidth=2.5, label=config, color=colors[idx % len(colors)])

        plt.xlabel('Depth Level', fontsize=14)
        plt.ylabel('Average Number of Instructions', fontsize=14)
        plt.title('Average Instruction Depth Distribution Comparison', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Configuration', fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "depth_distribution_comparison.png"), dpi=300)
        plt.close()

    def _plot_comparative_success_rate(self):
        plt.figure(figsize=(14, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.depth_data)))

        for idx, (config, success_rates) in enumerate(
                sorted(self.depth_data.items(), key=lambda x: int(x[0].replace('config', '')))):
            plt.plot(range(len(success_rates)), success_rates, 's-', markersize=6, linewidth=2.5, label=config,
                     color=colors[idx % len(colors)])

        plt.xlabel('Depth Level', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.title('Average Settlement Success Rate Comparison Across Configurations', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Configuration', fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "success_rate_comparison.png"), dpi=300)
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

    def _plot_total_instruction_comparison(self):
        configs = list(self.depth_data.keys())
        totals = [sum(data) for data in self.depth_data.values()]
        sorted_indices = np.argsort([int(config.replace('config', '')) for config in configs])

        sorted_configs = [configs[i] for i in sorted_indices]
        sorted_totals = [totals[i] for i in sorted_indices]

        plt.figure(figsize=(14, 8))
        bars = plt.bar(sorted_configs, sorted_totals, color='steelblue')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1, f'{int(height):,}', ha='center', va='bottom')

        plt.xlabel('Configuration', fontsize=14)
        plt.ylabel('Average Number of Instructions', fontsize=14)
        plt.title('Total Instructions Processed per Configuration', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "total_instructions_comparison.png"), dpi=300)
        plt.close()

    def _plot_normalized_completion_rate(self):
        plt.figure(figsize=(14, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.depth_data)))

        for idx, (config, rates) in enumerate(sorted(self.depth_data.items(), key=lambda x: int(x[0].replace('config', '')))):
            normalized_rates = [rate / rates[0] * 100 if rates[0] else 0 for rate in rates]
            plt.plot(range(len(normalized_rates)), normalized_rates, 'o-', markersize=6, linewidth=2.5, label=config, color=colors[idx % len(colors)])

        plt.xlabel('Depth Level', fontsize=14)
        plt.ylabel('Completion Rate (% of Depth 0)', fontsize=14)
        plt.title('Normalized Settlement Completion Rate by Depth', fontsize=16)
        plt.ylim(0, 140)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Configuration', fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "depth_completion_rates.png"), dpi=300)
        plt.close()

    def _plot_success_rate_heatmap(self):
        configs = sorted(self.depth_data.keys(), key=lambda x: int(x.replace('config', '')))
        max_depth = 0
        for config in configs:
            depths = len(self.depth_data[config])
            max_depth = max(max_depth, depths)

        success_matrix = []
        for config in configs:
            counts = self.depth_data[config]
            rates = []
            for depth in range(max_depth):
                if depth < len(counts):
                    total = counts[depth]
                    success = total
                    rate = (success / total * 100) if total else 0
                else:
                    rate = 0
                rates.append(rate)
            success_matrix.append(rates)

        success_matrix = np.array(success_matrix)

        plt.figure(figsize=(14, 10))
        sns.heatmap(success_matrix, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100, linewidths=0.5, linecolor='white')
        plt.title('Settlement Success Rate Heatmap', fontsize=16)
        plt.xlabel('Depth Level', fontsize=14)
        plt.ylabel('Configuration', fontsize=14)
        plt.yticks(ticks=np.arange(len(configs)) + 0.5, labels=configs, rotation=0)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "success_rate_heatmap.png"), dpi=300)
        plt.close()

