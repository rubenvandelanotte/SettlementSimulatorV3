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

        # grab both .jsonocel and .json exports
        self.log_files = [
            f for f in os.listdir(log_folder)
            if f.endswith(".jsonocel") or f.endswith(".json")
        ]

        if not self.log_files:
            print(f"[WARNING] No simulation log files found in {log_folder}")
            return

        for log_file in self.log_files:
            # match either “config12” or “truecount12” (case-insensitive)
            config_match = re.search(r'(?:config|truecount)(\d+)', log_file, re.IGNORECASE)
            if not config_match:
                continue

            config_num = int(config_match.group(1))
            config_name = f"Config {config_num}"

            try:
                with open(os.path.join(log_folder, log_file), 'r') as f:
                    log_data = json.load(f)

                events = log_data.get("ocel:events", log_data.get("events", {}))

                for event in (events.values() if isinstance(events, dict) else events):
                    # extract lateness and depth
                    attrs = event.get("ocel:attributes", event.get("attributes", {}))
                    if isinstance(attrs, dict):
                        lateness = attrs.get("lateness_hours")
                        depth = attrs.get("depth")
                    else:
                        lateness = next((a["value"] for a in attrs if a.get("name") == "lateness_hours"), None)
                        depth = next((a["value"] for a in attrs if a.get("name") == "depth"), None)

                    if lateness is not None:
                        try:
                            h = float(lateness)
                            self.config_lateness[config_name].append(h)
                            self.all_lateness_hours.append(h)
                            if depth is not None:
                                self.lateness_by_depth[int(depth)].append(h)
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

        plt.title('Overall Distribution of Settlement Lateness Hours')
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
        plt.plot(depths, avg_lateness, 'o-', color='blue')
        for d, h in zip(depths, avg_lateness):
            plt.text(d, h + 0.5, f"{h:.2f}", ha='center', va='bottom', fontsize=8)

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
        # build mean hours by config/depth
        lateness_md = defaultdict(lambda: defaultdict(list))
        for cfg, lst in self.config_lateness.items():
            for h in lst:
                lateness_md[cfg][0].append(h)

        configs = sorted(lateness_md.keys(), key=lambda x: int(x.split()[1]))
        depths = sorted({d for sub in lateness_md.values() for d in sub.keys()})

        heat = np.zeros((len(depths), len(configs)))
        for i, d in enumerate(depths):
            for j, cfg in enumerate(configs):
                vals = lateness_md[cfg].get(d, [])
                heat[i, j] = np.mean(vals) if vals else 0

        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd",
                         cbar_kws={'label': 'Avg Hours Late'})
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
        for d, lst in self.lateness_by_depth.items():
            x += [d] * len(lst)
            y += lst

        plt.figure(figsize=(14, 8))
        plt.scatter(x, y, alpha=0.5, s=30)
        if x and y:
            m, b = np.polyfit(x, y, 1)
            xs = np.array(sorted(set(x)))
            plt.plot(xs, m * xs + b, "r--", label=f"Trend: y = {m:.2f}x + {b:.2f}")
        # annotate a few random points to avoid clutter
        for xi, yi in list(zip(x, y))[:20]:
            plt.text(xi, yi, f"{yi:.1f}", fontsize=6, alpha=0.7)

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
            (0, 1, "<1h"), (1, 6, "1-6h"), (6, 12, "6-12h"),
            (12, 24, "12-24h"), (24, 48, "1-2d"), (48, float('inf'), ">2d")
        ]

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
            bars = plt.bar(configs, heights, bottom=bottom, label=label)
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

