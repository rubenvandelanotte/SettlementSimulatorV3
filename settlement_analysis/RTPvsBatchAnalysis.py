import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, time

class RTPvsBatchAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        """
                input_dir: str - top-level folder containing simulation outputs (expects 'logs/' subfolder)
                output_dir: str - top-level folder to save visualizations
                suite: SettlementAnalysisSuite - gives access to preloaded logs and statistics
                """
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "rtp_vs_batch")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        config_data = defaultdict(lambda: {"rtp": 0, "batch": 0})

        for log in self.suite.logs:
            events = []
            if isinstance(log, dict):
                if "ocel:events" in log:
                    events = list(log.get("ocel:events", {}).values())
                elif "events" in log:
                    events = log.get("events", [])

            config_name = self._extract_config_name(log)
            if not config_name:
                continue

            for event in events:
                if event.get("type") in ["Settled On Time", "Settled Late"] or event.get("ocel:activity") in ["Settled On Time", "Settled Late"]:
                    timestamp = event.get("time") or event.get("ocel:timestamp")
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            event_time = dt.time()
                            if time(1, 30) <= event_time <= time(19, 30):
                                config_data[config_name]["rtp"] += 1
                            elif event_time >= time(22, 0):
                                config_data[config_name]["batch"] += 1
                        except Exception:
                            continue

        self._plot_percentage_stack(config_data)
        self._plot_absolute_counts(config_data)
        self._plot_trend_lines(config_data)

    def _extract_config_name(self, log):
        if isinstance(log, dict) and "log_name" in log:
            parts = log["log_name"].split("_")
        elif isinstance(log, dict) and "meta" in log and "name" in log["meta"]:
            parts = log["meta"]["name"].split("_")
        else:
            return None

        for part in parts:
            if part.startswith("config"):
                return part
        return None

    def _plot_percentage_stack(self, config_data):
        configs = sorted(config_data.keys(), key=lambda x: int(x.replace("config", "")))
        rtp_pct = []
        batch_pct = []
        for cfg in configs:
            total = config_data[cfg]["rtp"] + config_data[cfg]["batch"]
            rtp_pct.append((config_data[cfg]["rtp"] / total) * 100 if total else 0)
            batch_pct.append((config_data[cfg]["batch"] / total) * 100 if total else 0)

        plt.figure(figsize=(12, 8))
        plt.bar(configs, rtp_pct, label='Real-Time Processing', color='skyblue')
        plt.bar(configs, batch_pct, bottom=rtp_pct, label='Batch Processing', color='salmon')
        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Settled Instructions')
        plt.title('RTP vs Batch Processing Settlements by Configuration')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rtp_vs_batch_percentage.png"), dpi=300)
        plt.close()

    def _plot_absolute_counts(self, config_data):
        configs = sorted(config_data.keys(), key=lambda x: int(x.replace("config", "")))
        rtp_counts = [config_data[cfg]["rtp"] for cfg in configs]
        batch_counts = [config_data[cfg]["batch"] for cfg in configs]

        x = range(len(configs))
        width = 0.35

        plt.figure(figsize=(12, 8))
        bars1 = plt.bar([i - width / 2 for i in x], rtp_counts, width, label='Real-Time Processing', color='skyblue')
        bars2 = plt.bar([i + width / 2 for i in x], batch_counts, width, label='Batch Processing', color='salmon')
        plt.xticks(x, configs)
        plt.xlabel('Configuration')
        plt.ylabel('Number of Settled Instructions')
        plt.title('RTP vs Batch Processing Settlements by Configuration')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        for i, (r, b) in enumerate(zip(rtp_counts, batch_counts)):
            plt.text(i - width / 2, r + 100, f"{r}", ha='center', fontsize=8)
            plt.text(i + width / 2, b + 100, f"{b}", ha='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rtp_vs_batch_absolute.png"), dpi=300)
        plt.close()

    def _plot_trend_lines(self, config_data):
        configs = sorted(config_data.keys(), key=lambda x: int(x.replace("config", "")))
        rtp = [config_data[cfg]["rtp"] for cfg in configs]
        batch = [config_data[cfg]["batch"] for cfg in configs]
        total = [r + b for r, b in zip(rtp, batch)]

        x = range(len(configs))
        plt.figure(figsize=(12, 8))
        plt.plot(x, total, 'o-', color='purple', label='Total Settlements')
        plt.plot(x, rtp, 's--', color='deepskyblue', label='Real-Time Processing')
        plt.plot(x, batch, 'd--', color='tomato', label='Batch Processing')
        for i, (t, r, b) in enumerate(zip(total, rtp, batch)):
            plt.text(i, t + 200, f"{t}", ha='center', fontsize=8, color='purple')
        plt.xticks(x, configs)
        plt.xlabel('Configuration')
        plt.ylabel('Number of Settled Instructions')
        plt.title('Settlement Trends by Processing Type Across Configurations')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rtp_vs_batch_trend.png"), dpi=300)
        plt.close()