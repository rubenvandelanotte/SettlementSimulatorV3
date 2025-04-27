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
        self._plot_runtime_bars(df)
        self._plot_runtime_boxplot(df)
        self._plot_correlation_heatmap(df)
        self._plot_runtime_vs_instruction_efficiency(df)
        self._plot_runtime_vs_value_efficiency(df)

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

            # Try to match to results file
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
            print("[WARNING] RuntimeAnalyzer built empty dataframe — check JSON contents.")
        return df

    def _plot_runtime_and_efficiency(self, df):
        grouped = df.groupby("config").mean().sort_index()

        fig, ax1 = plt.subplots(figsize=(16, 8))

        color = 'tab:blue'
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Execution Time (seconds)', color=color)
        ax1.plot(grouped.index, grouped["execution_time"], 'o-', color=color, label='Execution Time (s)')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Efficiency (%)', color=color)
        ax2.plot(grouped.index, grouped["instruction_efficiency"], 's--', color='tab:green', label='Instruction Efficiency')
        ax2.plot(grouped.index, grouped["value_efficiency"], '^--', color='tab:orange', label='Value Efficiency')
        ax2.tick_params(axis='y', labelcolor=color)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

        plt.title('Runtime and Efficiencies by Configuration')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_and_efficiency.png"))
        plt.close()

    def _plot_runtime_distribution(self, df):
        plt.figure(figsize=(14, 8))
        sns.histplot(df["execution_time"], bins=20, kde=False, color='skyblue', edgecolor='black')
        plt.title('Global Runtime Distribution')
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_histogram.png"))
        plt.close()

    def _plot_runtime_bars(self, df):
        grouped = df.groupby("config")["execution_time"]

        mean_runtime = grouped.mean()
        median_runtime = grouped.median()
        stddev_runtime = grouped.std()

        configs = mean_runtime.index

        x = np.arange(len(configs))  # Config positions
        width = 0.35

        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot mean runtimes
        ax.bar(x - width / 2, mean_runtime, width, yerr=stddev_runtime, capsize=5, label='Mean Runtime (s)',
               color='skyblue')

        # Plot median runtimes
        ax.bar(x + width / 2, median_runtime, width, label='Median Runtime (s)', color='lightgreen')

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

    def _plot_correlation_heatmap(self, df):
        corr = df[["execution_time", "instruction_efficiency", "value_efficiency"]].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, linewidths=0.5)
        plt.title('Correlation between Runtime and Efficiencies')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_correlation_heatmap.png"))
        plt.close()

    def _plot_runtime_vs_instruction_efficiency(self, df):
        df_valid = df.dropna(subset=["execution_time", "instruction_efficiency"])

        if df_valid.empty:
            print("[WARNING] No valid data for runtime vs instruction efficiency plot.")
            return

        plt.figure(figsize=(10, 8))
        plt.scatter(df_valid["execution_time"], df_valid["instruction_efficiency"], color='blue')

        m, b = np.polyfit(df_valid["execution_time"], df_valid["instruction_efficiency"], 1)
        plt.plot(df_valid["execution_time"], m*df_valid["execution_time"] + b, color='red', linestyle='--', label=f"Trend: y = {m:.4f}x + {b:.2f}")

        corr = df_valid["execution_time"].corr(df_valid["instruction_efficiency"])
        plt.text(0.05, 0.9, f"Correlation: {corr:.4f}", transform=plt.gca().transAxes)

        plt.title('Runtime vs Instruction Efficiency (%)')
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Instruction Efficiency (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_vs_instruction_efficiency.png"))
        plt.close()

    def _plot_runtime_vs_value_efficiency(self, df):
        df_valid = df.dropna(subset=["execution_time", "value_efficiency"])

        if df_valid.empty:
            print("[WARNING] No valid data for runtime vs value efficiency plot.")
            return

        plt.figure(figsize=(10, 8))
        plt.scatter(df_valid["execution_time"], df_valid["value_efficiency"], color='green')

        m, b = np.polyfit(df_valid["execution_time"], df_valid["value_efficiency"], 1)
        plt.plot(df_valid["execution_time"], m*df_valid["execution_time"] + b, color='red', linestyle='--', label=f"Trend: y = {m:.4f}x + {b:.2f}")

        corr = df_valid["execution_time"].corr(df_valid["value_efficiency"])
        plt.text(0.05, 0.9, f"Correlation: {corr:.4f}", transform=plt.gca().transAxes)

        plt.title('Runtime vs Value Efficiency (%)')
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Value Efficiency (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_vs_value_efficiency.png"))
        plt.close()
