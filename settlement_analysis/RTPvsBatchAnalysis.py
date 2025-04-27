import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
        # Change: Track both SUM and COUNT
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

        # Now build the average counts
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
        parts = filename.split("_")
        for part in parts:
            if part.lower().startswith("config"):
                num_part = part.replace("config", "")
                if num_part.isdigit():
                    return int(num_part)
        print(f"[WARNING] Failed to parse config from {filename}")
        return "Unknown"

    def _plot_absolute_counts(self, config_data):
        configs = sorted([c for c in config_data.keys() if isinstance(c, int)])

        rtp_counts = [config_data[cfg]["rtp"] for cfg in configs]
        batch_counts = [config_data[cfg]["batch"] for cfg in configs]

        x = np.arange(len(configs))
        width = 0.35

        plt.figure(figsize=(12, 8))
        plt.bar(x - width/2, rtp_counts, width, label='Real-Time Processing', color='skyblue')
        plt.bar(x + width/2, batch_counts, width, label='Batch Processing', color='salmon')
        plt.xticks(x, configs)
        plt.xlabel('Configuration')
        plt.ylabel('Number of Settled Instructions')
        plt.title('RTP vs Batch Processing Settlements by Configuration')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rtp_vs_batch_absolute.png"))
        plt.close()

    def _plot_percentage_stack(self, config_data):
        configs = sorted([c for c in config_data.keys() if isinstance(c, int)])

        rtp_pct = []
        batch_pct = []

        for cfg in configs:
            total = config_data[cfg]["rtp"] + config_data[cfg]["batch"]
            if total > 0:
                rtp_pct.append((config_data[cfg]["rtp"] / total) * 100)
                batch_pct.append((config_data[cfg]["batch"] / total) * 100)
            else:
                rtp_pct.append(0)
                batch_pct.append(0)

        plt.figure(figsize=(12, 8))
        plt.bar(configs, rtp_pct, label='Real-Time Processing', color='skyblue')
        plt.bar(configs, batch_pct, bottom=rtp_pct, label='Batch Processing', color='salmon')
        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Settled Instructions')
        plt.title('RTP vs Batch Processing Settlements by Configuration')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rtp_vs_batch_percentage.png"))
        plt.close()

    def _plot_trend_lines(self, config_data):
        configs = sorted([c for c in config_data.keys() if isinstance(c, int)])

        rtp = [config_data[cfg]["rtp"] for cfg in configs]
        batch = [config_data[cfg]["batch"] for cfg in configs]
        total = [r + b for r, b in zip(rtp, batch)]

        x = range(len(configs))
        plt.figure(figsize=(12, 8))
        plt.plot(x, total, 'o-', color='purple', label='Total Settlements')
        plt.plot(x, rtp, 's--', color='deepskyblue', label='Real-Time Processing')
        plt.plot(x, batch, '^--', color='tomato', label='Batch Processing')
        plt.xticks(x, configs)
        plt.xlabel('Configuration')
        plt.ylabel('Number of Settled Instructions')
        plt.title('Settlement Trends by Processing Type Across Configurations')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rtp_vs_batch_trend.png"))
        plt.close()
