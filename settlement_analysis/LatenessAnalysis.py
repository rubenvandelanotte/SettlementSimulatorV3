import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class LatenessAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "lateness_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        df = self._build_dataframe()
        if df.empty:
            print("[WARNING] No valid statistics data to analyze.")
            return

        self._plot_late_percentage_per_config(df)
        self._plot_lateness_by_depth(df)
        self._plot_lateness_heatmap(df)
        self._plot_ontime_vs_late_amounts(df)
        self._plot_ontime_vs_late_counts(df)
        self._plot_settlement_amount_trends(df)

    def _build_dataframe(self):
        records = []
        for stat_file, stat_data in self.suite.statistics.items():
            config_key = "Unknown"
            if "_config" in stat_file:
                config_key = stat_file.split("_config")[-1].split("_run")[0]

            depth_status = stat_data.get("depth_status_counts", {})
            depth_value = stat_data.get("depth_value_status_counts", {})

            settled_on_time = 0
            settled_late = 0
            ontime_amount = 0
            late_amount = 0
            depth_records = {}

            for depth, statuses in depth_status.items():
                depth = int(depth)
                late = statuses.get("Settled late", 0)
                ontime = statuses.get("Settled on time", 0)
                settled_late += late
                settled_on_time += ontime
                depth_records.setdefault(depth, {"late": 0, "ontime": 0})
                depth_records[depth]["late"] += late
                depth_records[depth]["ontime"] += ontime

            for depth, values in depth_value.items():
                late_amount += values.get("Settled late", 0)
                ontime_amount += values.get("Settled on time", 0)

            records.append({
                "config": config_key,
                "settled_on_time": settled_on_time,
                "settled_late": settled_late,
                "ontime_amount": ontime_amount,
                "late_amount": late_amount,
                "depth_records": depth_records
            })

        return pd.DataFrame(records)

    def _plot_late_percentage_per_config(self, df):
        df_grouped = df.groupby("config")[["settled_on_time", "settled_late"]].sum()
        lateness_percentage = df_grouped["settled_late"] / (df_grouped["settled_on_time"] + df_grouped["settled_late"]) * 100

        plt.figure(figsize=(12, 8))
        bars = plt.bar(lateness_percentage.index, lateness_percentage, color='orange', alpha=0.7)
        for bar, pct in zip(bars, lateness_percentage):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{pct:.1f}%', ha='center')
        plt.plot(lateness_percentage.index, np.poly1d(np.polyfit(range(len(lateness_percentage)), lateness_percentage, 1))(range(len(lateness_percentage))), 'r--')
        plt.title('Late Settlement Percentage by Configuration')
        plt.ylabel('Percentage of Settlements that were Late')
        plt.xlabel('Configuration')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "late_settlement_percentage.png"))
        plt.close()

    def _plot_lateness_by_depth(self, df):
        depth_totals = {}
        for _, row in df.iterrows():
            for depth, counts in row["depth_records"].items():
                depth_totals.setdefault(depth, {"late": 0, "ontime": 0})
                depth_totals[depth]["late"] += counts["late"]
                depth_totals[depth]["ontime"] += counts["ontime"]

        depth_list = sorted(depth_totals.keys())
        lateness_pct = [(depth_totals[d]["late"] / (depth_totals[d]["late"] + depth_totals[d]["ontime"]) * 100) if (depth_totals[d]["late"] + depth_totals[d]["ontime"]) > 0 else 0 for d in depth_list]

        plt.figure(figsize=(14, 8))
        plt.plot(depth_list, lateness_pct, 'o-', color='purple')
        for d, pct in zip(depth_list, lateness_pct):
            plt.text(d, pct + 1, f'{pct:.1f}%', ha='center')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Late Settlement Percentage')
        plt.title('Late Settlement Percentage by Instruction Depth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_by_depth.png"))
        plt.close()

    def _plot_lateness_heatmap(self, df):
        heatmap_data = {}
        for _, row in df.iterrows():
            config = row["config"]
            for depth, counts in row["depth_records"].items():
                heatmap_data.setdefault(depth, {})
                total = counts["late"] + counts["ontime"]
                if total > 0:
                    heatmap_data[depth][config] = counts["late"] / total * 100
                else:
                    heatmap_data[depth][config] = 0

        heatmap_df = pd.DataFrame(heatmap_data).T.sort_index()

        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="RdYlGn_r", vmin=0, vmax=100)
        plt.title('Late Settlement Percentage by Depth and Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Instruction Depth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_depth_config_heatmap.png"))
        plt.close()

    def _plot_ontime_vs_late_amounts(self, df):
        df_grouped = df.groupby("config")[["ontime_amount", "late_amount"]].sum()

        plt.figure(figsize=(14, 8))
        plt.bar(df_grouped.index, df_grouped["ontime_amount"], label="Settled On Time", color='green')
        plt.bar(df_grouped.index, df_grouped["late_amount"], bottom=df_grouped["ontime_amount"], label="Settled Late", color='orange')
        plt.title('On-Time vs Late Settlement Amounts by Configuration')
        plt.ylabel('Settlement Amount')
        plt.xlabel('Configuration')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ontime_vs_late_amounts.png"))
        plt.close()

    def _plot_ontime_vs_late_counts(self, df):
        df_grouped = df.groupby("config")[["settled_on_time", "settled_late"]].sum()

        plt.figure(figsize=(14, 8))
        plt.bar(df_grouped.index, df_grouped["settled_on_time"], label="Settled On Time", color='green')
        plt.bar(df_grouped.index, df_grouped["settled_late"], bottom=df_grouped["settled_on_time"], label="Settled Late", color='orange')
        plt.title('On-Time vs Late Settlements by Configuration')
        plt.ylabel('Number of Settlements')
        plt.xlabel('Configuration')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ontime_vs_late_counts.png"))
        plt.close()

    def _plot_settlement_amount_trends(self, df):
        df_grouped = df.groupby("config")[["ontime_amount", "late_amount"]].sum()
        df_grouped["total_amount"] = df_grouped["ontime_amount"] + df_grouped["late_amount"]

        plt.figure(figsize=(14, 8))
        plt.plot(df_grouped.index, df_grouped["total_amount"], 'o-', label="Total Amount", color='blue')
        plt.plot(df_grouped.index, df_grouped["ontime_amount"], 's--', label="On-Time Amount", color='green')
        plt.plot(df_grouped.index, df_grouped["late_amount"], '^--', label="Late Amount", color='orange')
        plt.title('Settlement Amount Trends by Configuration')
        plt.ylabel('Settlement Amount')
        plt.xlabel('Configuration')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "settlement_amount_trends.png"))
        plt.close()
