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

        # Global & per-config lateness
        self.config_lateness = defaultdict(list)
        # Per-depth lateness across all configs
        self.lateness_by_depth = defaultdict(list)
        # Per-(config, depth) lateness
        self.lateness_by_cfg_depth = defaultdict(lambda: defaultdict(list))
        # All lateness values
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

        self.log_files = [f for f in os.listdir(log_folder)
                          if f.endswith(".jsonocel") or f.endswith(".json")]

        if not self.log_files:
            print(f"[WARNING] No simulation log files found in {log_folder}")
            return

        for log_file in self.log_files:
            config_match = re.search(r'(?:config|truecount)(\d+)', log_file, re.IGNORECASE)
            if not config_match:
                continue

            config_num = int(config_match.group(1))
            config_name = f"Config {config_num}"

            try:
                with open(os.path.join(log_folder, log_file), 'r') as f:
                    log_data = json.load(f)

                events = log_data.get("ocel:events", log_data.get("events", {}))
                iterable = events.values() if isinstance(events, dict) else events

                for event in iterable:
                    # extract attributes container
                    attrs = event.get("ocel:attributes", event.get("attributes", {}))
                    if isinstance(attrs, dict):
                        lateness = attrs.get("lateness_hours")
                        depth = attrs.get("depth")
                    else:
                        lateness = next((a["value"] for a in attrs if a.get("name")=="lateness_hours"), None)
                        depth = next((a["value"] for a in attrs if a.get("name")=="depth"), None)

                    if lateness is not None:
                        try:
                            h = float(lateness)
                            d = int(depth) if depth is not None else None

                            # global and per-config
                            self.config_lateness[config_name].append(h)
                            self.all_lateness_hours.append(h)

                            # per-depth across configs
                            if d is not None:
                                self.lateness_by_depth[d].append(h)

                            # per-(config, depth)
                            if d is not None:
                                self.lateness_by_cfg_depth[config_name][d].append(h)
                        except ValueError:
                            pass

            except Exception as e:
                print(f"[ERROR] Failed to process {log_file}: {e}")

        print(f"[INFO] Loaded lateness data from {len(self.log_files)} files.")

    def _plot_lateness_statistics(self):
        configs = sorted(self.config_lateness.keys(), key=lambda x: int(x.split()[1]))
        avg_lateness = [np.mean(self.config_lateness[c]) for c in configs]
        median_lateness = [np.median(self.config_lateness[c]) for c in configs]
        max_lateness = [np.max(self.config_lateness[c]) for c in configs]

        plt.figure(figsize=(14, 8))
        bar_w = 0.25
        x = np.arange(len(configs))

        bars_avg = plt.bar(x - bar_w, avg_lateness, width=bar_w, label='Average', color='skyblue')
        bars_med = plt.bar(x, median_lateness, width=bar_w, label='Median', color='green')
        bars_max = plt.bar(x + bar_w, max_lateness, width=bar_w, label='Maximum', color='salmon')

        # annotate each bar
        for bars in (bars_avg, bars_med, bars_max):
            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.2f}",
                         ha='center', va='bottom', fontsize=8)

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
        data = [self.config_lateness[c] for c in configs]

        plt.figure(figsize=(14, 8))
        bp = sns.boxplot(data=data)
        # overlay raw points
        for i, d in enumerate(data):
            jitter = np.random.normal(i, 0.08, size=len(d))
            plt.scatter(jitter, d, alpha=0.3, color='black', s=10)

        # annotate medians
        medians = [np.median(d) for d in data]
        for i, m in enumerate(medians):
            plt.text(i, m + 0.5, f"{m:.2f}", ha='center', va='bottom', fontsize=8, color='white',
                     bbox=dict(facecolor='black', alpha=0.6, pad=1))

        plt.xticks(range(len(configs)), configs)
        plt.title('Distribution of Settlement Lateness Hours by Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Hours Late')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_boxplot.png"))
        plt.close()

    def _plot_lateness_histogram(self):
        all_h = self.all_lateness_hours
        mean_h = np.mean(all_h)
        median_h = np.median(all_h)

        plt.figure(figsize=(14, 8))
        n, bins, patches = plt.hist(all_h, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(mean_h, color='red', linestyle='--', label=f'Mean: {mean_h:.2f}')
        plt.axvline(median_h, color='green', linestyle='--', label=f'Median: {median_h:.2f}')

        # annotate histogram bar heights
        for count, edge in zip(n, bins[:-1]):
            if count > 0:
                plt.text(edge + (bins[1] - bins[0]) / 2, count + 0.5, str(int(count)),
                         ha='center', va='bottom', fontsize=7)

        plt.title('Overall Distribution of Settlement Lateness Hours Over All Configurations')
        plt.xlabel('Hours Late')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_histogram.png"))
        plt.close()

    def _plot_lateness_by_depth(self):
        depths = sorted(self.lateness_by_depth.keys())
        avg_lateness = [np.mean(self.lateness_by_depth[d]) for d in depths]

        plt.figure(figsize=(14, 8))
        plt.plot(depths, avg_lateness, 'o-', linewidth=2, markersize=6, color='blue')

        # Annotate each point
        for d, h in zip(depths, avg_lateness):
            plt.text(d, h + 0.5, f"{h:.2f}", ha='center', va='bottom', fontsize=8)

        # **Ensure every depth** appears on x-axis
        plt.xticks(depths, [str(d) for d in depths])

        plt.title('Average Lateness Hours by Instruction Depth')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Hours Late')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_by_depth.png"))
        plt.close()



    def _plot_violin_by_depth(self):
        depths = sorted(self.lateness_by_depth.keys())
        data = [self.lateness_by_depth[d] for d in depths]

        plt.figure(figsize=(14, 8))
        vp = sns.violinplot(data=data)
        # annotate median on each violin
        for i, d in enumerate(data):
            med = np.median(d)
            plt.text(i, med + 0.5, f"{med:.2f}", ha='center', va='bottom', fontsize=8, color='black')

        plt.xticks(range(len(depths)), depths)
        plt.title('Distribution of Lateness Hours by Instruction Depth')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Hours Late')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_violin.png"))
        plt.close()

    def _plot_lateness_heatmap(self):
        # sort configs numerically
        configs = sorted(
            self.lateness_by_cfg_depth.keys(),
            key=lambda name: int(name.split()[1])
        )
        # collect all depths seen under any config
        depths = sorted({
            d for cfg in configs
            for d in self.lateness_by_cfg_depth[cfg].keys()
        })

        # build matrix of averages (NaN if missing)
        heatmap_data = np.full((len(depths), len(configs)), np.nan)
        for i, d in enumerate(depths):
            for j, cfg in enumerate(configs):
                vals = self.lateness_by_cfg_depth[cfg].get(d, [])
                if vals:
                    heatmap_data[i, j] = np.mean(vals)

        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar_kws={'label': 'Avg Hours Late'},
            xticklabels=configs,
            yticklabels=depths
        )
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_yticklabels(depths, rotation=0)

        plt.title('Average Lateness Hours by Depth and Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Instruction Depth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_heatmap.png"))
        plt.close()

    def _plot_lateness_scatter(self):
        x = []
        y = []
        for depth, lst in self.lateness_by_depth.items():
            x += [depth] * len(lst)
            y += lst

        # Determine all depth levels present
        depths = sorted(self.lateness_by_depth.keys())

        plt.figure(figsize=(14, 8))
        plt.scatter(x, y, alpha=0.5, s=30)

        # Add a linear trend line if there's data
        if x and y:
            m, b = np.polyfit(x, y, 1)
            xs = np.array(depths)
            plt.plot(xs, m * xs + b, "r--", label=f"Trend: y = {m:.2f}x + {b:.2f}")

        # Annotate a sample of points to avoid clutter
        for xi, yi in list(zip(x, y))[:20]:
            plt.text(xi, yi, f"{yi:.1f}", fontsize=6, alpha=0.7)

        # Ensure every depth appears on the x-axis
        plt.xticks(depths)

        plt.title('Scatter Plot of Lateness Hours vs Instruction Depth')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Hours Late')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_scatter.png"))
        plt.close()

    def _plot_lateness_time_buckets(self):
        configs = sorted(self.config_lateness.keys(), key=lambda x: int(x.split()[1]))
        buckets = [
            (1, 12, "1-12h"),
            (12, 24, "12-24h"),
            (24, 48, "1-2d"),
            (48, 120, "2-5d"),
            (120, float('inf'), ">5d")
        ]

        # Define a color gradient from green to orange to red
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']  # green->yellow->orange->red

        # count
        counts = {cfg: [0] * len(buckets) for cfg in configs}
        for cfg in configs:
            for h in self.config_lateness[cfg]:
                for i, (lo, hi, _) in enumerate(buckets):
                    if lo <= h < hi:
                        counts[cfg][i] += 1
                        break

        # percentages
        pcts = {
            cfg: [
                (cnt / sum(counts[cfg]) * 100 if sum(counts[cfg]) > 0 else 0)
                for cnt in counts[cfg]
            ]
            for cfg in configs
        }

        bottom = np.zeros(len(configs))
        plt.figure(figsize=(14, 8))

        for i, (_, _, label) in enumerate(buckets):
            heights = [pcts[cfg][i] for cfg in configs]
            bars = plt.bar(configs, heights, bottom=bottom, label=label, color=colors[i])

            # annotate each segment
            for j, bar in enumerate(bars):
                h = bar.get_height()
                if h > 0:
                    plt.text(bar.get_x() + bar.get_width() / 2, bottom[j] + h / 2,
                             f"{h:.1f}%", ha='center', va='center', fontsize=7)
            bottom += heights

        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Late Settlements')
        plt.title('Lateness Time Categories by Configuration')
        plt.legend(title='Category')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_categories.png"))
        plt.close()

