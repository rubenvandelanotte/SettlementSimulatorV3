import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class RuntimeAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "runtime_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        runtime_df = self._build_runtime_dataframe()
        if runtime_df.empty:
            print("[WARNING] No valid runtime entries to analyze.")
            return

        self._plot_runtime_boxplot(runtime_df)
        self._plot_avg_runtime_with_error(runtime_df)
        self._plot_correlation_heatmap(runtime_df)
        self._plot_runtime_histogram(runtime_df)
        self._plot_runtime_vs_instruction_efficiency(runtime_df)
        self._plot_runtime_vs_value_efficiency(runtime_df)
        self._plot_runtime_efficiency_multiline(runtime_df)
        self._plot_avg_runtime_bar(runtime_df)

    def _build_runtime_dataframe(self):
        records = []
        for entry in self.suite.runtime_data:
            run_label = entry.get("run_label")
            exec_time = entry.get("execution_time_seconds")
            config_key = "Unknown"

            if run_label and "_" in run_label:
                parts = run_label.split("_")
                for p in parts:
                    if p.lower().startswith("config"):
                        try:
                            config_key = int(p.replace("config", ""))
                        except ValueError:
                            pass

            instruction_eff = None
            value_eff = None

            for stat_file, stat_data in self.suite.statistics.items():
                if run_label in stat_file or run_label.replace("Run", "run") in stat_file:
                    instruction_eff = stat_data.get("instruction_efficiency")
                    value_eff = stat_data.get("value_efficiency")
                    break

            records.append({
                "config": config_key,
                "run_label": run_label,
                "execution_time": exec_time,
                "instruction_efficiency": instruction_eff,
                "value_efficiency": value_eff
            })

        return pd.DataFrame(records)

    def _plot_runtime_boxplot(self, df):
        plt.figure(figsize=(14, 8))
        sns.boxplot(x="config", y="execution_time", data=df)
        plt.ylabel('Runtime (seconds)')
        plt.title('Runtime Distribution per Configuration')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_boxplot.png"))
        plt.close()

    def _plot_avg_runtime_with_error(self, df):
        avg_runtime = df.groupby("config")["execution_time"].mean()
        std_runtime = df.groupby("config")["execution_time"].std()

        configs = avg_runtime.index.tolist()
        x = np.arange(len(configs))

        plt.figure(figsize=(12, 8))
        plt.errorbar(x, avg_runtime, yerr=std_runtime, fmt='o', ecolor='red', capsize=5)
        plt.xticks(x, configs)
        plt.xlabel('Configuration')
        plt.ylabel('Average Runtime (seconds)')
        plt.title('Average Runtime per Configuration with Error Bars')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_by_config.png"))
        plt.close()

    def _plot_correlation_heatmap(self, df):
        plt.figure(figsize=(10, 8))
        corr = df[["execution_time", "instruction_efficiency", "value_efficiency"]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title('Correlation between Runtime and Efficiencies')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_correlation_heatmap.png"))
        plt.close()

    def _plot_runtime_histogram(self, df):
        plt.figure(figsize=(12, 8))
        plt.hist(df["execution_time"], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Frequency')
        plt.title('Global Runtime Distribution')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_histogram.png"))
        plt.close()

    def _plot_runtime_vs_instruction_efficiency(self, df):
        plt.figure(figsize=(10, 8))
        sns.regplot(x="execution_time", y="instruction_efficiency", data=df, scatter_kws={'s':50}, line_kws={'color':'red'})
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Instruction Efficiency')
        plt.title('Runtime vs Instruction Efficiency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_vs_instruction_efficiency.png"))
        plt.close()

    def _plot_runtime_vs_value_efficiency(self, df):
        plt.figure(figsize=(10, 8))
        sns.regplot(x="execution_time", y="value_efficiency", data=df, scatter_kws={'s':50}, line_kws={'color':'green'})
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Value Efficiency')
        plt.title('Runtime vs Value Efficiency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_vs_value_efficiency.png"))
        plt.close()

    def _plot_runtime_efficiency_multiline(self, df):
        avg_data = df.groupby("config")[["execution_time", "instruction_efficiency", "value_efficiency"]].mean()
        plt.figure(figsize=(14, 8))
        plt.plot(avg_data.index, avg_data["execution_time"], 'o-', label='Execution Time (s)')
        plt.plot(avg_data.index, avg_data["instruction_efficiency"], 's-', label='Instruction Efficiency')
        plt.plot(avg_data.index, avg_data["value_efficiency"], 'd-', label='Value Efficiency')
        plt.xlabel('Configuration')
        plt.title('Runtime and Efficiencies by Configuration')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(avg_data.index)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_and_efficiency.png"))
        plt.close()

    def _plot_avg_runtime_bar(self, df):
        avg_runtime = df.groupby("config")["execution_time"].mean()
        std_runtime = df.groupby("config")["execution_time"].std()

        plt.figure(figsize=(12, 8))
        avg_runtime.plot(kind='bar', yerr=std_runtime, capsize=4, color='steelblue', error_kw={'elinewidth':2})
        plt.ylabel('Average Runtime (seconds)')
        plt.title('Average Runtime with Std Dev per Configuration')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_bars.png"))
        plt.close()
