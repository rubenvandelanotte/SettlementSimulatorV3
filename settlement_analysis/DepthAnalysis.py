import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

class DepthAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "depth_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

        # [PATCH] Local fix only, don't overwrite suite.statistics
        fixed_statistics = self._fix_statistics_format(self.suite.statistics)
        self.statistics = self._aggregate_statistics(fixed_statistics)

    def _fix_statistics_format(self, statistics):
        fixed_statistics = {}
        for stat_file, data in statistics.items():
            fixed_data = {}
            if "depth_counts" in data:
                fixed_data["depth_counts"] = {str(k): int(v) for k, v in data["depth_counts"].items()}
            if "depth_status_counts" in data:
                fixed_status_counts = {}
                for depth, statuses in data["depth_status_counts"].items():
                    fixed_status_counts[str(depth)] = {str(status): int(count) for status, count in statuses.items()}
                fixed_data["depth_status_counts"] = fixed_status_counts
            fixed_statistics[stat_file] = fixed_data
        return fixed_statistics

    # [PATCH] Accept statistics as argument
    def _aggregate_statistics(self, statistics):
        aggregated_stats = {}
        for stat_file, data in statistics.items():
            config_match = re.search(r'config(\d+)', stat_file)
            if config_match:
                config_num = int(config_match.group(1))
                config_name = f"Config {config_num}"
            else:
                config_name = "Unknown"

            if config_name not in aggregated_stats:
                aggregated_stats[config_name] = {
                    "depth_counts": defaultdict(int),
                    "depth_status_counts": defaultdict(lambda: defaultdict(int))
                }

            depth_counts = data.get("depth_counts", {})
            if not depth_counts:
                print(f"[WARNING] No depth_counts found in {stat_file}")

            for depth, count in depth_counts.items():
                aggregated_stats[config_name]["depth_counts"][depth] += count

            depth_status_counts = data.get("depth_status_counts", {})
            if not depth_status_counts:
                print(f"[WARNING] No depth_status_counts found in {stat_file}")

            for depth, statuses in depth_status_counts.items():
                for status, count in statuses.items():
                    aggregated_stats[config_name]["depth_status_counts"][depth][status] += count

        return aggregated_stats

    def run(self):
        print("[INFO] Running DepthAnalyzer with the following configurations:")
        for config_name, data in self.statistics.items():
            print(f"Config: {config_name}, Depths: {list(data['depth_counts'].keys())}")
            self._analyze_single_config(config_name, data)

        self._compare_success_rates()
        self._compare_depth_distributions()
        self._compare_status_distributions()
        self._compare_total_instructions()
        self._compare_normalized_completion_rate()
        self._plot_success_rate_heatmap()
        self._export_summary()

    def _safe_legend(self):
        handles, labels = plt.gca().get_legend_handles_labels()
        if any(label and not label.startswith('_') for label in labels):
            plt.legend()

    def _analyze_single_config(self, config_name, data):
        config_output_dir = os.path.join(self.output_dir, config_name)
        os.makedirs(config_output_dir, exist_ok=True)

        self._depth_distribution(data, config_name, os.path.join(config_output_dir, "depth_distribution.png"))
        self._status_by_depth(data, config_name, os.path.join(config_output_dir, "status_by_depth.png"))
        self._success_rate_by_depth(data, config_name, os.path.join(config_output_dir, "success_rate.png"))

    def _depth_distribution(self, data, config_name, output_file):
        depth_counts = data["depth_counts"]
        depths = sorted(int(d) for d in depth_counts.keys())
        counts = [depth_counts[str(d)] for d in depths]

        plt.figure(figsize=(10, 6))
        plt.bar(depths, counts, color='skyblue')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Average Number of Instructions')
        plt.title(f'{config_name}: Depth Distribution')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def _status_by_depth(self, data, config_name, output_file):
        depth_status_counts = data["depth_status_counts"]
        depths = sorted(int(d) for d in depth_status_counts.keys())

        status_groups = {
            "Settled on time": ["Settled on time"],
            "Settled late": ["Settled late"],
            "Cancelled": ["Cancelled"],
            "In process": ["In process"]
        }
        group_colors = {
            "Settled on time": "green",
            "Settled late": "orange",
            "Cancelled": "red",
            "In process": "gray"
        }

        group_data = {}
        for group, statuses in status_groups.items():
            group_data[group] = []
            for depth in depths:
                count = sum(depth_status_counts.get(str(depth), {}).get(status, 0) for status in statuses)
                group_data[group].append(count)

        plt.figure(figsize=(12, 8))
        bottom = [0] * len(depths)

        for group, counts in group_data.items():
            plt.bar(depths, counts, bottom=bottom, label=group, color=group_colors.get(group, 'gray'))
            bottom = [b + c for b, c in zip(bottom, counts)]

        plt.xlabel('Instruction Depth')
        plt.ylabel('Average Number of Instructions')
        plt.title(f'{config_name}: Status by Depth')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def _success_rate_by_depth(self, data, config_name, output_file):
        depth_status_counts = data["depth_status_counts"]
        depths = sorted(int(d) for d in depth_status_counts.keys())

        success_rates = []
        for depth in depths:
            statuses = depth_status_counts.get(str(depth), {})
            total = sum(statuses.values())
            successful = statuses.get("Settled on time", 0) + statuses.get("Settled late", 0)
            success_rate = (successful / total * 100) if total > 0 else 0
            success_rates.append(success_rate)

        plt.figure(figsize=(10, 6))
        plt.plot(depths, success_rates, 'o-', linewidth=2, markersize=8, color='blue')
        plt.xlabel('Depth Level')
        plt.ylabel('Success Rate (%)')
        plt.title(f'{config_name}: Success Rate by Depth')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def _compare_success_rates(self):
        plt.figure(figsize=(14, 8))
        for config_name, data in self.statistics.items():
            depth_status_counts = data["depth_status_counts"]
            depths = sorted(int(d) for d in depth_status_counts.keys())

            success_rates = []
            for depth in depths:
                statuses = depth_status_counts.get(str(depth), {})
                total = sum(statuses.values())
                successful = statuses.get("Settled on time", 0) + statuses.get("Settled late", 0)
                success_rate = (successful / total * 100) if total > 0 else 0
                success_rates.append(success_rate)

            plt.plot(depths, success_rates, marker='o', label=config_name)

        plt.xlabel('Depth Level')
        plt.ylabel('Success Rate (%)')
        plt.title('Comparative Success Rate Across Configurations')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparative_success_rate.png"))
        plt.close()

    def _compare_depth_distributions(self):
        plt.figure(figsize=(14, 8))
        for config_name, data in self.statistics.items():
            depth_counts = data["depth_counts"]
            depths = sorted(int(d) for d in depth_counts.keys())
            counts = [depth_counts[str(d)] for d in depths]
            plt.plot(depths, counts, marker='o', label=config_name)

        plt.xlabel('Instruction Depth')
        plt.ylabel('Average Number of Instructions')
        plt.title('Comparative Depth Distribution Across Configurations')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparative_depth_distribution.png"))
        plt.close()

    def _compare_status_distributions(self):
        plt.figure(figsize=(14, 8))
        depths = range(30)
        status_groups = ["Settled on time", "Settled late", "Cancelled", "In process"]
        colors = ['green', 'orange', 'red', 'gray']

        for config_name, data in self.statistics.items():
            depth_status_counts = data["depth_status_counts"]
            for idx, status in enumerate(status_groups):
                counts = [depth_status_counts.get(str(d), {}).get(status, 0) for d in depths]
                plt.plot(depths, counts, label=f'{config_name} - {status}', color=colors[idx % len(colors)])

        plt.xlabel('Depth Level')
        plt.ylabel('Instruction Count')
        plt.title('Comparative Status Distribution Across Configurations')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparative_status_distribution.png"))
        plt.close()

    def _compare_total_instructions(self):
        configs = []
        totals = []
        for config_name, data in self.statistics.items():
            total_instructions = sum(data["depth_counts"].values())
            configs.append(config_name)
            totals.append(total_instructions)

        plt.figure(figsize=(14, 8))
        plt.bar(configs, totals, color='skyblue')
        plt.xticks(rotation=45)
        plt.xlabel('Configuration')
        plt.ylabel('Total Instructions')
        plt.title('Total Instructions per Configuration')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "total_instructions_comparison.png"))
        plt.close()

    def _compare_normalized_completion_rate(self):
        plt.figure(figsize=(14, 8))
        configs = []
        rates = []

        for config_name, data in self.statistics.items():
            depth_status_counts = data["depth_status_counts"]
            total = sum([sum(depth_status_counts.get(str(d), {}).values()) for d in range(30)])
            successful = sum([
                depth_status_counts.get(str(d), {}).get("Settled on time", 0) +
                depth_status_counts.get(str(d), {}).get("Settled late", 0) for d in range(30)
            ])
            rate = (successful / total * 100) if total > 0 else 0
            configs.append(config_name)
            rates.append(rate)

        plt.bar(configs, rates, color='lightgreen')
        plt.xticks(rotation=45)
        plt.xlabel('Configuration')
        plt.ylabel('Completion Rate (%)')
        plt.title('Normalized Completion Rate Across Configurations')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "normalized_completion_rate.png"))
        plt.close()

    def _plot_success_rate_heatmap(self):
        configs = sorted(self.statistics.keys())
        depths = range(30)
        heatmap_data = np.full((len(depths), len(configs)), np.nan)

        for j, config_name in enumerate(configs):
            depth_status_counts = self.statistics[config_name]["depth_status_counts"]
            for i in range(len(depths)):
                statuses = depth_status_counts.get(str(i), {})
                total = sum(statuses.values())
                successful = statuses.get("Settled on time", 0) + statuses.get("Settled late", 0)
                if total > 0:
                    heatmap_data[i, j] = (successful / total) * 100

        if np.any(~np.isnan(heatmap_data)):
            plt.figure(figsize=(14, 10))
            sns.heatmap(heatmap_data, annot=False, cmap="YlGnBu", xticklabels=configs, yticklabels=depths)
            plt.xlabel('Configuration')
            plt.ylabel('Instruction Depth')
            plt.title('Heatmap of Success Rate by Depth and Configuration')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "success_rate_heatmap.png"))
            plt.close()
        else:
            print("[WARNING] No valid data to plot in heatmap, skipping.")

    def _export_summary(self):
        summary_data = []
        for config_name, data in self.statistics.items():
            total_instructions = sum(data["depth_counts"].values())
            success = 0
            total = 0

            for depth, statuses in data["depth_status_counts"].items():
                total += sum(statuses.values())
                success += statuses.get("Settled on time", 0) + statuses.get("Settled late", 0)

            success_rate = (success / total * 100) if total > 0 else 0
            max_depth = max([int(d) for d in data["depth_counts"].keys()]) if data["depth_counts"] else 0

            summary_data.append({
                "Configuration": config_name,
                "Total Instructions": total_instructions,
                "Success Rate (%)": success_rate,
                "Max Depth": max_depth
            })

        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.output_dir, "depth_summary.csv"), index=False)
        with open(os.path.join(self.output_dir, "depth_summary.json"), 'w') as f:
            json.dump(summary_data, f, indent=2)
