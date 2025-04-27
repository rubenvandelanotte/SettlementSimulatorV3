import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

class LatenessHoursAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "lateness_hours_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

        self.config_lateness = defaultdict(list)
        self.lateness_by_depth = defaultdict(list)
        self.all_lateness_hours = []
        self.log_files = []

    def run(self):
        self._load_lateness_data()
        if not self.all_lateness_hours:
            print("[WARNING] No lateness hours found. Exiting.")
            return

        self._plot_lateness_statistics()
        self._plot_lateness_boxplot()
        self._plot_lateness_histogram()
        self._plot_lateness_by_depth()
        self._plot_violin_by_depth()
        self._plot_lateness_heatmap()
        self._plot_lateness_scatter()
        self._plot_lateness_time_buckets()

    def _load_lateness_data(self):
        print("[INFO] Loading lateness hours from logs...")
        log_folder = os.path.join(self.input_dir, "logs")

        if not os.path.exists(log_folder):
            print(f"[WARNING] Log folder '{log_folder}' does not exist.")
            return

        self.log_files = [f for f in os.listdir(log_folder) if f.endswith(".jsonocel") and "simulation" in f]

        if not self.log_files:
            print(f"[WARNING] No simulation log files found in {log_folder}")
            return

        for log_file in self.log_files:
            config_match = re.search(r'config(\d+)', log_file)
            if not config_match:
                continue

            config_num = int(config_match.group(1))
            config_name = f"Config {config_num}"

            try:
                with open(os.path.join(log_folder, log_file), 'r') as f:
                    log_data = json.load(f)

                events = log_data.get("ocel:events", log_data.get("events", {}))

                for event_id, event in (events.items() if isinstance(events, dict) else enumerate(events)):
                    lateness_hours = None
                    depth = None

                    attributes = event.get("ocel:attributes", event.get("attributes", {}))

                    if isinstance(attributes, dict):
                        lateness_hours = attributes.get("lateness_hours")
                        depth = attributes.get("depth")
                    elif isinstance(attributes, list):
                        for attr in attributes:
                            if attr.get("name") == "lateness_hours":
                                lateness_hours = attr.get("value")
                            if attr.get("name") == "depth":
                                depth = attr.get("value")

                    if lateness_hours is not None:
                        try:
                            lateness_hours = float(lateness_hours)
                            self.config_lateness[config_name].append(lateness_hours)
                            self.all_lateness_hours.append(lateness_hours)

                            if depth is not None:
                                self.lateness_by_depth[int(depth)].append(lateness_hours)
                        except Exception:
                            pass

            except Exception as e:
                print(f"[ERROR] Failed to process {log_file}: {e}")

        print(f"[INFO] Loaded lateness data from {len(self.log_files)} files.")

    def _plot_lateness_statistics(self):
        configs = sorted(self.config_lateness.keys(), key=lambda x: int(x.split()[1]))
        avg_lateness = [np.mean(self.config_lateness[cfg]) for cfg in configs]
        median_lateness = [np.median(self.config_lateness[cfg]) for cfg in configs]
        max_lateness = [np.max(self.config_lateness[cfg]) for cfg in configs]

        plt.figure(figsize=(14, 8))
        bar_width = 0.25
        x = np.arange(len(configs))

        plt.bar(x - bar_width, avg_lateness, width=bar_width, label='Average', color='skyblue')
        plt.bar(x, median_lateness, width=bar_width, label='Median', color='green')
        plt.bar(x + bar_width, max_lateness, width=bar_width, label='Maximum', color='salmon')

        plt.xlabel('Configuration')
        plt.ylabel('Hours Late')
        plt.title('Lateness Statistics by Configuration')
        plt.xticks(x, configs)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_statistics.png"))
        plt.close()

    def _plot_lateness_boxplot(self):
        configs = sorted(self.config_lateness.keys(), key=lambda x: int(x.split()[1]))
        box_data = [self.config_lateness[cfg] for cfg in configs]

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=box_data)
        for i, config_data in enumerate(box_data):
            jitter = np.random.normal(i, 0.08, size=len(config_data))
            plt.scatter(jitter, config_data, alpha=0.3, color='black', s=10)
        plt.xticks(np.arange(len(configs)), configs)
        plt.title('Distribution of Settlement Lateness Hours by Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Hours Late')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_boxplot.png"))
        plt.close()

    def _plot_lateness_histogram(self):
        plt.figure(figsize=(14, 8))
        plt.hist(self.all_lateness_hours, bins=30, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(self.all_lateness_hours), color='red', linestyle='--', label='Mean')
        plt.axvline(np.median(self.all_lateness_hours), color='green', linestyle='--', label='Median')
        plt.title('Overall Distribution of Settlement Lateness Hours')
        plt.xlabel('Hours Late')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_histogram.png"))
        plt.close()

    def _plot_lateness_by_depth(self):
        if not self.lateness_by_depth:
            return

        depths = sorted(self.lateness_by_depth.keys())
        avg_lateness = [np.mean(self.lateness_by_depth[d]) for d in depths]

        plt.figure(figsize=(14, 8))
        plt.plot(depths, avg_lateness, 'o-', color='blue')
        plt.title('Average Lateness Hours by Instruction Depth')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Hours Late')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_by_depth.png"))
        plt.close()

    def _plot_violin_by_depth(self):
        if not self.lateness_by_depth:
            return

        depths = sorted(self.lateness_by_depth.keys())
        violin_data = [self.lateness_by_depth[d] for d in depths]

        plt.figure(figsize=(14, 8))
        sns.violinplot(data=violin_data)
        plt.xticks(np.arange(len(depths)), depths)
        plt.title('Distribution of Lateness Hours by Instruction Depth')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Hours Late')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_violin.png"))
        plt.close()

    def _plot_lateness_heatmap(self):
        if not self.lateness_by_depth or not self.config_lateness:
            return

        lateness_by_config_depth = defaultdict(lambda: defaultdict(list))
        for config, lateness_list in self.config_lateness.items():
            for lateness in lateness_list:
                lateness_by_config_depth[config][0].append(lateness)

        configs = sorted(lateness_by_config_depth.keys(), key=lambda x: int(x.split()[1]))
        depths = sorted({depth for conf in lateness_by_config_depth.values() for depth in conf.keys()})

        if not depths or not configs:
            return

        heatmap_data = np.zeros((len(depths), len(configs)))
        for i, depth in enumerate(depths):
            for j, config in enumerate(configs):
                hours = lateness_by_config_depth[config].get(depth, [])
                heatmap_data[i, j] = np.mean(hours) if hours else 0

        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Avg Hours Late'})
        plt.title('Average Lateness Hours by Depth and Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Instruction Depth')
        plt.xticks(np.arange(len(configs)) + 0.5, configs, rotation=45, ha='right')
        plt.yticks(np.arange(len(depths)) + 0.5, depths, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_heatmap.png"))
        plt.close()

    def _plot_lateness_scatter(self):
        if not self.lateness_by_depth:
            return

        x, y = [], []
        for depth, hours_list in self.lateness_by_depth.items():
            for hours in hours_list:
                x.append(depth)
                y.append(hours)

        plt.figure(figsize=(14, 8))
        plt.scatter(x, y, alpha=0.5, color='blue')
        if x and y:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(sorted(x), p(sorted(x)), "r--", label=f"Trend: y = {z[0]:.2f}x + {z[1]:.2f}")
            plt.legend()
        plt.title('Scatter Plot of Lateness Hours vs Instruction Depth')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Hours Late')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_scatter.png"))
        plt.close()

    def _plot_lateness_time_buckets(self):
        if not self.config_lateness:
            return

        configs = sorted(self.config_lateness.keys(), key=lambda x: int(x.split()[1]))

        buckets = [(0, 1, "< 1h"), (1, 6, "1-6h"), (6, 12, "6-12h"), (12, 24, "12-24h"), (24, 48, "1-2d"), (48, float('inf'), ">2d")]

        bucket_counts = {config: [0] * len(buckets) for config in configs}
        for config in configs:
            for hours in self.config_lateness[config]:
                for idx, (low, high, _) in enumerate(buckets):
                    if low <= hours < high:
                        bucket_counts[config][idx] += 1
                        break

        bucket_pcts = {config: [(count / sum(bucket_counts[config]) * 100) if sum(bucket_counts[config]) else 0 for count in counts] for config, counts in bucket_counts.items()}

        plt.figure(figsize=(14, 8))
        bottom = np.zeros(len(configs))
        bucket_colors = ['green', 'yellowgreen', 'gold', 'orange', 'darkorange', 'red']

        for i, (_, _, label) in enumerate(buckets):
            heights = [bucket_pcts[cfg][i] for cfg in configs]
            plt.bar(configs, heights, bottom=bottom, label=label, color=bucket_colors[i])
            bottom += np.array(heights)

        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Late Settlements')
        plt.title('Lateness Time Categories by Configuration')
        plt.legend(title='Lateness Category')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_categories.png"))
        plt.close()
