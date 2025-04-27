import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RuntimeAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "runtime_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        df = self._build_runtime_dataframe()
        print(f"[DEBUG] RuntimeAnalyzer DataFrame shape: {df.shape}")

        if df.empty:
            print("[WARNING] No runtime data available.")
            return

        self._plot_runtime_and_efficiency(df)
        self._plot_runtime_distribution(df)
        self._plot_runtime_bars_mean_median(df)
        self._plot_runtime_boxplot(df)
        self._plot_runtime_vs_efficiency_scatter(df, "instruction_efficiency")
        self._plot_runtime_vs_efficiency_scatter(df, "value_efficiency")
        self._plot_correlation_heatmap(df)

    def _build_runtime_dataframe(self):
        records = []
        results_dir = os.path.join(self.input_dir, "results_all_analysis")

        if not hasattr(self.suite, 'runtime_data') or not self.suite.runtime_data:
            print("[WARNING] No runtime data found inside suite object.")
            return pd.DataFrame()

        for entry in self.suite.runtime_data:
            config_info = entry.get("config", {})
            true_count = config_info.get("true_count", None)
            run_number = config_info.get("run_number", None)
            execution_time = entry.get("execution_time_seconds", None)

            instruction_efficiency = None
            value_efficiency = None

            if true_count is not None and run_number is not None:
                stats_filename = f"results_partial_config{true_count}_run{run_number}.json"
                stats_path = os.path.join(results_dir, stats_filename)

                if os.path.exists(stats_path):
                    try:
                        with open(stats_path, "r") as f:
                            stats_data = json.load(f)
                            instruction_efficiency = stats_data.get("instruction_efficiency", None)
                            value_efficiency = stats_data.get("value_efficiency", None)
                    except Exception as e:
                        print(f"[WARNING] Failed to read {stats_path}: {e}")

            if true_count is not None and execution_time is not None:
                records.append({
                    "config": int(true_count),
                    "run_number": int(run_number) if run_number is not None else None,
                    "execution_time": execution_time,
                    "instruction_efficiency": instruction_efficiency,
                    "value_efficiency": value_efficiency
                })

        df = pd.DataFrame(records)
        if df.empty:
            print("[WARNING] RuntimeAnalyzer built empty dataframe.")
        return df

    def _plot_runtime_and_efficiency(self, df):
        grouped = df.groupby("config").mean().sort_index()

        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Execution Time (seconds)', color='tab:blue')
        ax1.plot(grouped.index, grouped["execution_time"], 'o-', color='tab:blue', label='Execution Time')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Efficiency (%)')
        ax2.plot(grouped.index, grouped["instruction_efficiency"], 's--', color='tab:green', label='Instruction Efficiency')
        ax2.plot(grouped.index, grouped["value_efficiency"], '^--', color='tab:orange', label='Value Efficiency')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')

        plt.title('Runtime and Efficiencies by Configuration')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_and_efficiency.png"))
        plt.close()

    def _plot_runtime_distribution(self, df):
        plt.figure(figsize=(14, 8))
        plt.hist(df['execution_time'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(df['execution_time'].mean(), color='red', linestyle='--', label=f'Mean: {df["execution_time"].mean():.2f}s')
        plt.axvline(df['execution_time'].median(), color='green', linestyle='--', label=f'Median: {df["execution_time"].median():.2f}s')
        plt.title('Global Runtime Distribution')
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_histogram.png"))
        plt.close()

    def _plot_runtime_bars_mean_median(self, df):
        grouped = df.groupby("config")["execution_time"]

        mean_runtime = grouped.mean()
        median_runtime = grouped.median()
        stddev_runtime = grouped.std()

        configs = mean_runtime.index
        x = np.arange(len(configs))
        width = 0.35

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.bar(x - width/2, mean_runtime, width, yerr=stddev_runtime, capsize=5, label='Mean Runtime', color='skyblue')
        ax.bar(x + width/2, median_runtime, width, label='Median Runtime', color='lightgreen')

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Mean and Median Runtime per Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Config {cfg}" for cfg in configs])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_mean_median_bars.png"))
        plt.close()

    def _plot_runtime_boxplot(self, df):
        plt.figure(figsize=(14, 8))
        sns.boxplot(x="config", y="execution_time", data=df)
        plt.title('Runtime Distribution per Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Runtime (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_boxplot.png"))
        plt.close()

    def _plot_runtime_vs_efficiency_scatter(self, df, metric):
        df_valid = df.dropna(subset=["execution_time", metric])

        if df_valid.empty:
            print(f"[WARNING] No valid data for runtime vs {metric} scatter plot.")
            return

        plt.figure(figsize=(12, 8))
        plt.scatter(df_valid["execution_time"], df_valid[metric], s=100, alpha=0.7, color='skyblue', edgecolors='black')

        for idx, row in df_valid.iterrows():
            plt.annotate(f"Config {row['config']}", (row["execution_time"], row[metric]), xytext=(5, 5), textcoords='offset points', fontsize=8)

        m, b = np.polyfit(df_valid["execution_time"], df_valid[metric], 1)
        x_trend = np.linspace(df_valid["execution_time"].min(), df_valid["execution_time"].max(), 100)
        plt.plot(x_trend, m*x_trend + b, "r--", label=f"Trend: y = {m:.2f}x + {b:.2f}")

        corr = df_valid["execution_time"].corr(df_valid[metric])
        plt.figtext(0.5, 0.01, f"Correlation: {corr:.4f}", ha='center', fontsize=12, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

        plt.xlabel('Execution Time (seconds)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Runtime vs {metric.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"runtime_vs_{metric}.png"))
        plt.close()

    def _plot_correlation_heatmap(self, df):
        corr = df[["execution_time", "instruction_efficiency", "value_efficiency"]].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, linewidths=0.5, fmt=".2f")
        plt.title('Correlation between Runtime and Efficiencies')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_correlation_heatmap.png"))
        plt.close()
