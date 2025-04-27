import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LatenessHoursAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "lateness_hours_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        df = self._build_lateness_dataframe()
        print(f"[DEBUG] Lateness DataFrame shape: {df.shape}")
        if df.empty:
            print("[WARNING] No lateness data found.")
            return

        self._plot_lateness_statistics(df)
        self._plot_lateness_categories(df)
        self._plot_lateness_boxplot(df)
        self._plot_lateness_histogram(df)
        self._plot_lateness_heatmap(df)

    def _build_lateness_dataframe(self):
        records = []
        for filename, stats in self.suite.statistics.items():
            config = self._extract_config_name(filename)
            lateness_by_depth = stats.get("lateness_by_depth", {})

            for depth_str, avg_lateness in lateness_by_depth.items():
                try:
                    avg_lateness = float(avg_lateness)
                except Exception:
                    continue
                records.append({
                    "config": config,
                    "depth": int(depth_str),
                    "lateness_hours": avg_lateness
                })

        return pd.DataFrame(records)

    def _extract_config_name(self, filename):
        parts = filename.split("_")
        for part in parts:
            if part.lower().startswith("config"):
                try:
                    return int(part.replace("config", ""))
                except ValueError:
                    return "Unknown"
        return "Unknown"

    def _plot_lateness_statistics(self, df):
        grouped = df.groupby("config")["lateness_hours"]
        summary_df = pd.DataFrame({
            "Average": grouped.mean(),
            "Median": grouped.median(),
            "Maximum": grouped.max()
        }).reset_index()

        x = np.arange(len(summary_df))
        width = 0.25

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.bar(x - width, summary_df["Average"], width, label='Average', color='skyblue')
        ax.bar(x, summary_df["Median"], width, label='Median', color='green')
        ax.bar(x + width, summary_df["Maximum"], width, label='Maximum', color='salmon')

        for i, v in enumerate(summary_df["Maximum"]):
            ax.text(i + width, v + 5, f"{v:.1f}", ha='center', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(["Config " + str(c) for c in summary_df["config"]])
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Hours Late")
        ax.set_title("Lateness Statistics by Configuration")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_statistics.png"))
        plt.close()

    def _plot_lateness_categories(self, df):
        bins = [0, 1, 6, 12, 24, 48, np.inf]
        labels = ['< 1 hour', '1-6 hours', '6-12 hours', '12-24 hours', '1-2 days', '> 2 days']
        df['lateness_category'] = pd.cut(df['lateness_hours'], bins=bins, labels=labels, right=False)

        category_counts = df.groupby(["config", "lateness_category"], observed=False).size().unstack(fill_value=0)


        # Normalize to percentages
        category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(16, 8))
        bottom = np.zeros(len(category_percentages))

        for category in labels:
            ax.bar(
                category_percentages.index,
                category_percentages[category],
                label=category,
                bottom=bottom
            )
            bottom += category_percentages[category]

        for i, total in enumerate(bottom):
            if total > 100:
                bottom[i] = 100

        for idx, row in category_percentages.iterrows():
            cumulative = 0
            for category in labels:
                value = row[category]
                if value > 5:
                    ax.text(idx, cumulative + value/2, f"{value:.1f}%", ha='center', va='center', fontsize=8)
                cumulative += value

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Percentage of Late Settlements')
        ax.set_title('Lateness Time Categories by Configuration')
        ax.legend(title="Lateness Category")
        ax.set_xticks(category_percentages.index)
        ax.set_xticklabels(["Config " + str(c) for c in category_percentages.index])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_categories.png"))
        plt.close()

    def _plot_lateness_boxplot(self, df):
        plt.figure(figsize=(16, 10))
        sns.boxplot(x="config", y="lateness_hours", data=df, showfliers=True, boxprops=dict(alpha=.6))
        sns.stripplot(x="config", y="lateness_hours", data=df, color="black", size=2, jitter=True, alpha=0.3)
        plt.title('Distribution of Settlement Lateness Hours by Configuration')
        plt.ylabel('Hours Late')
        plt.xlabel('Configuration')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_boxplot.png"))
        plt.close()

    def _plot_lateness_histogram(self, df):
        plt.figure(figsize=(14, 8))
        plt.hist(df['lateness_hours'], bins=30, color='skyblue', edgecolor='black')
        plt.axvline(df['lateness_hours'].mean(), color='red', linestyle='--', label=f'Mean: {df["lateness_hours"].mean():.2f} hours')
        plt.axvline(df['lateness_hours'].median(), color='green', linestyle='--', label=f'Median: {df["lateness_hours"].median():.2f} hours')
        plt.title('Overall Distribution of Settlement Lateness Hours')
        plt.xlabel('Hours Late')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_histogram.png"))
        plt.close()

    def _plot_lateness_heatmap(self, df):
        if df.empty or "depth" not in df.columns:
            print("[WARNING] No depth data available for lateness heatmap.")
            return

        heatmap_data = df.pivot_table(index="depth", columns="config", values="lateness_hours", aggfunc="mean")

        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5)
        plt.title('Average Lateness Hours by Depth and Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Instruction Depth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_heatmap.png"))
        plt.close()
