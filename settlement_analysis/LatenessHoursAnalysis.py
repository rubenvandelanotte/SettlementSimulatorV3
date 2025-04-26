import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class LatenessHoursAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "lateness_hours_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        df = self._build_lateness_dataframe()
        if df.empty:
            print("[WARNING] No lateness data found.")
            return

        self._plot_lateness_categories(df)
        self._plot_lateness_boxplot(df)
        self._plot_lateness_heatmap(df)
        self._plot_lateness_histogram(df)
        self._plot_lateness_statistics(df)

    def _build_lateness_dataframe(self):
        records = []
        for log in self.suite.logs:
            events = []
            if isinstance(log, dict):
                if "ocel:events" in log:
                    events = list(log.get("ocel:events", {}).values())
                elif "events" in log:
                    events = log.get("events", [])

            config = self._extract_config_name(log)

            for event in events:
                event_type = event.get("type") or event.get("ocel:activity")
                if event_type == "Settled Late":
                    lateness_hours = event.get("lateness_hours")
                    depth = event.get("depth", None)
                    if lateness_hours is not None:
                        records.append({
                            "config": config,
                            "lateness_hours": lateness_hours,
                            "depth": depth
                        })

        return pd.DataFrame(records)

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

    def _plot_lateness_categories(self, df):
        bins = [0, 1, 6, 12, 24, 48, np.inf]
        labels = ['< 1 hour', '1-6 hours', '6-12 hours', '12-24 hours', '1-2 days', '> 2 days']
        df['category'] = pd.cut(df['lateness_hours'], bins=bins, labels=labels, right=False)

        grouped = df.groupby(['config', 'category']).size().unstack(fill_value=0)
        grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100

        colors = sns.color_palette("YlOrRd", len(labels))

        plt.figure(figsize=(14, 8))
        grouped.plot(kind='bar', stacked=True, color=colors, ax=plt.gca())
        plt.title('Lateness Time Categories by Configuration')
        plt.ylabel('Percentage of Late Settlements')
        plt.xlabel('Configuration')
        plt.legend(title='Lateness Category')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_categories.png"))
        plt.close()

    def _plot_lateness_boxplot(self, df):
        plt.figure(figsize=(16, 10))
        sns.boxplot(x="config", y="lateness_hours", data=df, showfliers=False)
        sns.stripplot(x="config", y="lateness_hours", data=df, color="black", size=2, jitter=True)
        plt.title('Distribution of Settlement Lateness Hours by Configuration')
        plt.ylabel('Hours Late')
        plt.xlabel('Configuration')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_boxplot.png"))
        plt.close()

    def _plot_lateness_heatmap(self, df):
        if "depth" not in df.columns:
            print("[WARNING] No depth data available for lateness heatmap.")
            return

        heatmap_data = df.dropna(subset=["depth"]).groupby(["depth", "config"])['lateness_hours'].mean().unstack()

        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5)
        plt.title('Average Lateness Hours by Depth and Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Instruction Depth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_heatmap.png"))
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

    def _plot_lateness_statistics(self, df):
        grouped = df.groupby("config")["lateness_hours"].agg(["mean", "median", "max"])

        x = np.arange(len(grouped.index))
        width = 0.25

        plt.figure(figsize=(16, 10))
        plt.bar(x - width, grouped["mean"], width, label='Average')
        plt.bar(x, grouped["median"], width, label='Median')
        plt.bar(x + width, grouped["max"], width, label='Maximum')

        plt.xticks(x, grouped.index, rotation=45)
        plt.ylabel('Hours Late')
        plt.title('Lateness Statistics by Configuration')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_statistics.png"))
        plt.close()
