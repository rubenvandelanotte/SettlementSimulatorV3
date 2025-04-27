import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LatenessAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "lateness_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        df, total_intended_amounts = self._build_lateness_dataframe()
        print(f"[DEBUG] LatenessAnalyzer DataFrame shape: {df.shape}")

        if df.empty:
            print("[WARNING] No lateness data available.")
            return

        self._plot_late_percentage_by_config(df)
        self._plot_lateness_by_depth(df)
        self._plot_lateness_depth_config_heatmap(df)
        self._plot_ontime_vs_late_amounts_fixed(df)
        self._plot_ontime_vs_late_counts(df)
        self._plot_settlement_amount_trends_fixed(df, total_intended_amounts)

    def _build_lateness_dataframe(self):
        records = []
        total_intended_amounts = {}

        for filename, stats in self.suite.statistics.items():
            config = self._extract_config_name(filename)

            # Safely fallback
            ontime_count = stats.get("settled_ontime_rtp", 0) + stats.get("settled_ontime_batch", 0)
            late_count = stats.get("settled_late_rtp", 0) + stats.get("settled_late_batch", 0)

            ontime_amount = stats.get("settled_on_time_amount", 0)
            late_amount = stats.get("settled_late_amount", 0)

            depth_counts = stats.get("depth_counts", {})

            for depth, count in depth_counts.items():
                try:
                    depth_int = int(depth)
                except ValueError:
                    continue

                records.append({
                    "config": config,
                    "depth": depth_int,
                    "total_count": count,  # How many instructions at this depth
                    # Additional fields for later aggregation
                    "ontime_count": ontime_count,
                    "late_count": late_count,
                    "ontime_amount": ontime_amount,
                    "late_amount": late_amount,
                })

            # Save total intended amount for normalized trends
            intended = stats.get("intended_amount", 0)
            total_intended_amounts[config] = intended

        df = pd.DataFrame(records)
        return df, total_intended_amounts

    def _extract_config_name(self, filename):
        parts = filename.split("_")
        for part in parts:
            if part.lower().startswith("config"):
                try:
                    return int(part.replace("config", ""))
                except ValueError:
                    return "Unknown"
        return "Unknown"

    def _plot_late_percentage_by_config(self, df):
        df_grouped = df.groupby("config")[["ontime_count", "late_count"]].sum().sort_index()
        df_grouped["late_percentage"] = df_grouped["late_count"] / (df_grouped["ontime_count"] + df_grouped["late_count"]) * 100

        plt.figure(figsize=(14, 8))
        plt.bar(df_grouped.index, df_grouped["late_percentage"], color='tomato')
        plt.title('Late Settlement Percentage by Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Settlements that were Late')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "late_settlement_percentage.png"))
        plt.close()

    def _plot_lateness_by_depth(self, df):
        df_depth = df.groupby("depth")[["ontime_count", "late_count"]].sum()
        df_depth["late_percentage"] = df_depth["late_count"] / (df_depth["ontime_count"] + df_depth["late_count"]) * 100

        plt.figure(figsize=(14, 8))
        plt.plot(df_depth.index, df_depth["late_percentage"], marker='o', color='purple')
        for idx, val in enumerate(df_depth["late_percentage"]):
            plt.text(idx, val + 2, f"{val:.1f}%", ha='center', fontsize=8)

        plt.title('Late Settlement Percentage by Instruction Depth')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Late Settlement Percentage')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_by_depth.png"))
        plt.close()

    def _plot_lateness_depth_config_heatmap(self, df):
        grouped = df.groupby(["depth", "config"])[["ontime_count", "late_count"]].sum()
        grouped["late_percentage"] = grouped["late_count"] / (grouped["ontime_count"] + grouped["late_count"]) * 100

        pivot = grouped["late_percentage"].unstack(fill_value=0)

        plt.figure(figsize=(16, 10))
        ax = sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=0.5)

        plt.title('Late Settlement Percentage by Depth and Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Instruction Depth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_depth_config_heatmap.png"))
        plt.close()

    def _plot_ontime_vs_late_amounts_fixed(self, df):
        df_grouped = df.groupby("config")[["ontime_amount", "late_amount"]].sum().sort_index()

        plt.figure(figsize=(16, 8))
        plt.bar(df_grouped.index - 0.15, df_grouped["ontime_amount"], width=0.3, label="Settled On Time (€)", color='green')
        plt.bar(df_grouped.index + 0.15, df_grouped["late_amount"], width=0.3, label="Settled Late (€)", color='orange')
        plt.xlabel('Configuration')
        plt.ylabel('Settlement Amount (€)')
        plt.title('On-Time vs Late Settlement Amounts by Configuration (Fixed)')
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ontime_vs_late_amounts_fixed.png"))
        plt.close()

    def _plot_ontime_vs_late_counts(self, df):
        df_grouped = df.groupby("config")[["ontime_count", "late_count"]].sum().sort_index()

        plt.figure(figsize=(16, 8))
        plt.bar(df_grouped.index - 0.15, df_grouped["ontime_count"], width=0.3, label="Settled On Time", color='green')
        plt.bar(df_grouped.index + 0.15, df_grouped["late_count"], width=0.3, label="Settled Late", color='orange')
        plt.xlabel('Configuration')
        plt.ylabel('Number of Settlements')
        plt.title('On-Time vs Late Settlements by Configuration')
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ontime_vs_late_counts.png"))
        plt.close()

    def _plot_settlement_amount_trends_fixed(self, df, total_intended_amounts):
        df_grouped = df.groupby("config")[["ontime_amount", "late_amount"]].sum().sort_index()

        configs = df_grouped.index
        total_settled = df_grouped["ontime_amount"] + df_grouped["late_amount"]
        total_intended = [total_intended_amounts.get(cfg, 1) for cfg in configs]

        normalized_total = (total_settled / total_intended) * 100
        normalized_ontime = (df_grouped["ontime_amount"] / total_intended) * 100
        normalized_late = (df_grouped["late_amount"] / total_intended) * 100

        plt.figure(figsize=(16, 8))
        plt.plot(configs, normalized_total, marker='o', label='Total Settled (%)', color='blue')
        plt.plot(configs, normalized_ontime, marker='s', linestyle='--', label='On-Time Settled (%)', color='green')
        plt.plot(configs, normalized_late, marker='^', linestyle='--', label='Late Settled (%)', color='orange')

        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Intended Amount Settled')
        plt.title('Normalized Settlement Amount Trends by Configuration (Fixed)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "settlement_amount_trends_fixed.png"))
        plt.close()
