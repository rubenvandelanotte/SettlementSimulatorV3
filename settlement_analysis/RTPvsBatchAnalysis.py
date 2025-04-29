import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

class RTPvsBatchAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "rtp_vs_batch")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        config_data = self._build_rtp_batch_data()
        if not config_data:
            print("[WARNING] No RTP vs Batch data available.")
            return

        self._plot_absolute_counts(config_data)
        self._plot_percentage_stack(config_data)
        self._plot_trend_lines(config_data)

    def _build_rtp_batch_data(self):
        config_data = defaultdict(lambda: {"rtp_sum": 0, "batch_sum": 0, "count": 0})

        for filename, stats in self.suite.statistics.items():
            config = self._extract_config_name(filename)
            if config == "Unknown":
                continue

            rtp_count = stats.get("settled_ontime_rtp", 0) + stats.get("settled_late_rtp", 0)
            batch_count = stats.get("settled_ontime_batch", 0) + stats.get("settled_late_batch", 0)

            config_data[config]["rtp_sum"] += rtp_count
            config_data[config]["batch_sum"] += batch_count
            config_data[config]["count"] += 1

        averaged_data = {}
        for config, values in config_data.items():
            count = values["count"]
            if count > 0:
                averaged_data[config] = {
                    "rtp": values["rtp_sum"] / count,
                    "batch": values["batch_sum"] / count
                }

        return averaged_data

    def _extract_config_name(self, filename):
        """
        Extract config number from filename.
        Supports both 'configX' and 'truecountX'.
        """
        match = re.search(r'(?:config|truecount)(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            print(f"[WARNING] Failed to parse config from {filename}")
            return None

    def _plot_absolute_counts(self, config_data):
        configs = sorted(config_data.keys())
        rtp_counts = [config_data[c]["rtp"] for c in configs]
        batch_counts = [config_data[c]["batch"] for c in configs]

        x = np.arange(len(configs))
        width = 0.35

        plt.figure(figsize=(12, 8))
        bars_rtp = plt.bar(x - width/2, rtp_counts, width, label='Real-Time', color='skyblue')
        bars_batch = plt.bar(x + width/2, batch_counts, width, label='Batch', color='salmon')

        # Add value labels on top of each bar
        for bar in bars_rtp + bars_batch:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + max(rtp_counts+batch_counts)*0.01,
                     f"{int(height)}", ha='center', va='bottom', fontsize=8)

        plt.xticks(x, [f"Config {c}" for c in configs])
        plt.xlabel('Configuration')
        plt.ylabel('Average Settled Instructions')
        plt.title('RTP vs Batch: Absolute Settlement Counts')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rtp_vs_batch_absolute.png"))
        plt.close()

    def _plot_percentage_stack(self, config_data):
        configs = sorted(config_data.keys())
        rtp_pct = []
        batch_pct = []
        for c in configs:
            r = config_data[c]["rtp"]
            b = config_data[c]["batch"]
            total = r + b
            if total > 0:
                rtp_pct.append(r/total*100)
                batch_pct.append(b/total*100)
            else:
                rtp_pct.append(0)
                batch_pct.append(0)

        x = np.arange(len(configs))
        plt.figure(figsize=(12, 8))
        bars_rtp = plt.bar(x, rtp_pct, label='Real-Time %', color='skyblue')
        bars_batch = plt.bar(x, batch_pct, bottom=rtp_pct, label='Batch %', color='salmon')

        # Annotate each segment
        for idx in range(len(configs)):
            # RTP segment
            plt.text(x[idx], rtp_pct[idx]/2, f"{rtp_pct[idx]:.1f}%", ha='center', va='center', fontsize=8)
            # Batch segment
            plt.text(x[idx], rtp_pct[idx] + batch_pct[idx]/2, f"{batch_pct[idx]:.1f}%",
                     ha='center', va='center', fontsize=8)

        plt.xticks(x, [f"Config {c}" for c in configs])
        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Settlements')
        plt.title('RTP vs Batch: Settlement Percentages')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rtp_vs_batch_percentage.png"))
        plt.close()

    def _plot_trend_lines(self, config_data):
        configs = sorted(config_data.keys())
        rtp = [config_data[c]["rtp"] for c in configs]
        batch = [config_data[c]["batch"] for c in configs]
        total = [r + b for r, b in zip(rtp, batch)]

        x = np.arange(len(configs))
        plt.figure(figsize=(12, 8))
        line_tot, = plt.plot(x, total, 'o-', label='Total', color='purple')
        line_rtp, = plt.plot(x, rtp, 's--', label='Real-Time', color='deepskyblue')
        line_batch, = plt.plot(x, batch, '^--', label='Batch', color='tomato')

        # Annotate each point
        for xi, yv in zip(x, total):
            plt.text(xi, yv + max(total)*0.01, f"{int(yv)}", ha='center', va='bottom', fontsize=8)
        for xi, yv in zip(x, rtp):
            plt.text(xi, yv - max(total)*0.02, f"{int(yv)}", ha='center', va='top', fontsize=8)
        for xi, yv in zip(x, batch):
            plt.text(xi, yv - max(total)*0.02, f"{int(yv)}", ha='center', va='top', fontsize=8)

        plt.xticks(x, [f"Config {c}" for c in configs])
        plt.xlabel('Configuration')
        plt.ylabel('Average Settled Instructions')
        plt.title('Settlement Trends by Processing Type')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rtp_vs_batch_trend.png"))
        plt.close()
