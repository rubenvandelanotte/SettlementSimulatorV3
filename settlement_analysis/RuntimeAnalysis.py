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

    def _find_stats_file(self, true_count, run_number):
        results_dir = os.path.join(self.input_dir, "results_all_analysis")
        if not os.path.exists(results_dir):
            print(f"[ERROR] Results directory '{results_dir}' not found.")
            return None

        for file in os.listdir(results_dir):
            if file.endswith(".json") and f"truecount{true_count}_" in file and f"run{run_number}" in file:
                return os.path.join(results_dir, file)

        print(f"[WARNING] No stats file found for true_count={true_count}, run_number={run_number}")
        return None

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
                stats_path = self._find_stats_file(true_count, run_number)

                if stats_path and os.path.exists(stats_path):
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
        cfgs = grouped.index.values
        exec_times = grouped["execution_time"].values
        instr_eff = grouped["instruction_efficiency"].values
        val_eff = grouped["value_efficiency"].values

        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Execution Time (s)', color='tab:blue')
        line1, = ax1.plot(cfgs, exec_times, 'o-', label='Exec Time', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        # annotate exec times
        for x, y in zip(cfgs, exec_times):
            ax1.text(x, y + max(exec_times)*0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Efficiency (%)')
        line2, = ax2.plot(cfgs, instr_eff, 's--', label='Instr Eff', color='tab:green')
        line3, = ax2.plot(cfgs, val_eff, '^--', label='Value Eff', color='tab:orange')
        # annotate efficiencies
        for x, y in zip(cfgs, instr_eff):
            ax2.text(x, y - max(instr_eff)*0.01, f"{y:.1f}%", ha='center', va='top', fontsize=8)
        for x, y in zip(cfgs, val_eff):
            ax2.text(x, y - max(val_eff)*0.01, f"{y:.1f}%", ha='center', va='top', fontsize=8)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')
        plt.title('Runtime and Efficiencies by Configuration')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_and_efficiency.png"))
        plt.close()

    def _plot_runtime_distribution(self, df):
        data = df['execution_time'].values
        mean_v = np.mean(data)
        median_v = np.median(data)
        plt.figure(figsize=(14, 8))
        n, bins, patches = plt.hist(data, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(mean_v, color='red', linestyle='--', label=f'Mean: {mean_v:.2f}')
        plt.axvline(median_v, color='green', linestyle='--', label=f'Median: {median_v:.2f}')
        # annotate counts
        for count, edge in zip(n, bins[:-1]):
            if count > 0:
                plt.text(edge + (bins[1]-bins[0])/2, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=7)
        plt.title('Runtime Distribution')
        plt.xlabel('Execution Time (s)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_histogram.png"))
        plt.close()

    def _plot_runtime_bars_mean_median(self, df):
        grouped = df.groupby('config')['execution_time']
        mean_rt = grouped.mean().values
        med_rt = grouped.median().values
        std_rt = grouped.std().values
        cfgs = grouped.mean().index.values
        x = np.arange(len(cfgs))
        width = 0.35

        fig, ax = plt.subplots(figsize=(16, 8))
        bars1 = ax.bar(x - width/2, mean_rt, width, yerr=std_rt, capsize=5, label='Mean', color='skyblue')
        bars2 = ax.bar(x + width/2, med_rt, width, label='Median', color='lightgreen')
        # annotate bars
        for bar in bars1 + bars2:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h + max(mean_rt+med_rt)*0.01, f"{h:.2f}", ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Config {c}" for c in cfgs])
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Runtime (s)')
        ax.set_title('Mean and Median Runtime per Configuration')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_mean_median_bars.png"))
        plt.close()

    def _plot_runtime_boxplot(self, df):
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='config', y='execution_time', data=df)
        # annotate medians
        meds = df.groupby('config')['execution_time'].median().values
        cfgs = sorted(df['config'].unique())
        for i, m in enumerate(meds):
            plt.text(i, m + 0.5, f"{m:.2f}", ha='center', va='bottom', fontsize=8)
        plt.title('Runtime Distribution per Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Runtime (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_boxplot.png"))
        plt.close()

    def _plot_runtime_vs_efficiency_scatter(self, df, metric):
        df_valid = df.dropna(subset=['execution_time', metric])
        if df_valid.empty:
            return
        plt.figure(figsize=(12, 8))
        plt.scatter(df_valid['execution_time'], df_valid[metric], s=100, alpha=0.7)
        # annotate points with config
        for _, row in df_valid.iterrows():
            plt.annotate(f"Cfg {row['config']}", (row['execution_time'], row[metric]), fontsize=8)
        m, b = np.polyfit(df_valid['execution_time'], df_valid[metric], 1)
        xs = np.linspace(df_valid['execution_time'].min(), df_valid['execution_time'].max(), 100)
        plt.plot(xs, m*xs + b, 'r--', label=f"Trend: y={m:.2f}x+{b:.2f}")
        corr = df_valid['execution_time'].corr(df_valid[metric])
        plt.figtext(0.5, 0.01, f"Corr: {corr:.3f}", ha='center', fontsize=12)
        plt.title(f'Runtime vs {metric.replace("_", " ").title()}')
        plt.xlabel('Execution Time (s)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"runtime_vs_{metric}.png"))
        plt.close()

    def _plot_correlation_heatmap(self, df):
        corr = df[['execution_time', 'instruction_efficiency', 'value_efficiency']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt='.2f')
        plt.title('Correlation between Runtime and Efficiencies')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "runtime_correlation_heatmap.png"))
        plt.close()