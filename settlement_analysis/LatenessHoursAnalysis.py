import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class LatenessHoursAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "lateness_hours_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        df = self._build_lateness_dataframe()
        print(f"[DEBUG] Dataframe shape: {df.shape}")  # Debug line
        if df.empty:
            print("[WARNING] No lateness data found.")
            return

        self._plot_lateness_boxplot(df)
        self._plot_lateness_histogram(df)
        self._plot_lateness_heatmap(df)

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
                if event_type and "settled" in event_type.lower() and "late" in event_type.lower():
                    lateness_hours = event.get("lateness_hours")
                    if lateness_hours is not None:
                        records.append({
                            "config": config,
                            "lateness_hours": lateness_hours,
                            "depth": event.get("depth", None)
                        })

        print(f"[DEBUG] Total late events found: {len(records)}")  # Debug line
        return pd.DataFrame(records)

    def _extract_config_name(self, log):
        if isinstance(log, dict):
            if "log_name" in log:
                parts = log["log_name"].split("_")
            elif "meta" in log and "name" in log["meta"]:
                parts = log["meta"]["name"].split("_")
            else:
                return "Unknown"

            for part in parts:
                if part.lower().startswith("config"):
                    try:
                        return int(part.replace("config", ""))
                    except ValueError:
                        return "Unknown"
        return "Unknown"

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
        if "depth" not in df.columns or df['depth'].isnull().all():
            print("[WARNING] No depth data available for lateness heatmap.")
            return

        heatmap_data = df.dropna(subset=["depth"]).groupby(["depth", "config"])["lateness_hours"].mean().unstack()

        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5)
        plt.title('Average Lateness Hours by Depth and Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Instruction Depth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_hours_heatmap.png"))
        plt.close()
