import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ConfidenceIntervalAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "confidence_interval_analysis_fixed")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        df = self._build_dataframe()
        if df.empty:
            print("[WARNING] No valid statistics data to analyze.")
            return

        self._plot_instruction_efficiency(df)
        self._plot_value_efficiency(df)
        self._plot_value_efficiency_ontime_only(df)
        self._plot_combined_efficiency(df)



    def _build_dataframe(self):
        import re
        records = []
        for stat_file, stat_data in self.suite.statistics.items():
            # Extract config number using regex
            match = re.search(r'(?:config|truecount)(\d+)', stat_file)
            config_key = match.group(1) if match else "Unknown"

            instruction_eff = stat_data.get("instruction_efficiency")
            value_eff = stat_data.get("value_efficiency")
            mothers_effectively_settled = stat_data.get("mothers_effectively_settled", 0)
            settled_on_time_amount = stat_data.get("settled_on_time_amount", 0)
            settled_late_amount = stat_data.get("settled_late_amount", 0)
            settled_amount = settled_on_time_amount + settled_late_amount

            records.append({
                "config": config_key,
                "instruction_efficiency": instruction_eff,
                "value_efficiency": value_eff,
                "mothers_effectively_settled": mothers_effectively_settled,
                "settled_on_time_amount": settled_on_time_amount,
                "settled_amount": settled_amount
            })

        df = pd.DataFrame(records)

        # Drop any rows missing critical numeric values
        df = df.dropna(subset=["instruction_efficiency", "value_efficiency"])

        # Ensure config is treated as numeric for sorting
        df["config"] = pd.to_numeric(df["config"], errors="coerce")
        df = df.dropna(subset=["config"])
        df["config"] = df["config"].astype(int)

        return df

    def _plot_instruction_efficiency(self, df):
        grouped = df.groupby("config").agg(
            instruction_efficiency_mean=("instruction_efficiency", "mean"),
            instruction_efficiency_std=("instruction_efficiency", "std"),
            mothers_settled_mean=("mothers_effectively_settled", "mean")
        ).sort_index()

        configs = grouped.index.tolist()
        x = np.arange(len(configs))
        ci_multiplier = 1.96 / np.sqrt(len(df.groupby("config")))

        fig, ax1 = plt.subplots(figsize=(14, 8))

        y_err = grouped['instruction_efficiency_std'] * ci_multiplier
        ax1.errorbar(x, grouped['instruction_efficiency_mean'].values, yerr=y_err.values,
                     fmt='o', color='deepskyblue', capsize=5, label='Instruction Efficiency')
        ax1.set_ylabel('Instruction Efficiency (%)', color='deepskyblue')
        ax1.tick_params(axis='y', labelcolor='deepskyblue')

        ax2 = ax1.twinx()
        ax2.plot(x, grouped['mothers_settled_mean'].values, 's-', color='green', label='Mothers Settled On Time')
        ax2.set_ylabel('Mothers Settled On Time', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        for i in range(len(x)):
            ax1.text(x[i], grouped['instruction_efficiency_mean'].values[i] + y_err.values[i] + 0.5,
                     f"{grouped['instruction_efficiency_mean'].values[i]:.2f}%", ha='center', fontsize=9,
                     color='deepskyblue')
            ax2.text(x[i], grouped['mothers_settled_mean'].values[i] + 1,
                     f"{int(grouped['mothers_settled_mean'].values[i])}", ha='center', fontsize=9, color='green')

        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Config {cfg}" for cfg in configs], rotation=45, ha='right')
        ax1.set_title('Instruction Efficiency (95% CI) vs Mothers Settled (On Time or Late)')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "instruction_efficiency_ci.png"), dpi=300)
        plt.close()

    def _plot_value_efficiency(self, df):
        grouped = df.groupby("config").agg(
            value_efficiency_mean=("value_efficiency", "mean"),
            value_efficiency_std=("value_efficiency", "std"),
            settled_amount_mean=("settled_amount", "mean")
        ).sort_index()

        configs = grouped.index.tolist()
        x = np.arange(len(configs))
        ci_multiplier = 1.96 / np.sqrt(len(df.groupby("config")))

        fig, ax1 = plt.subplots(figsize=(14, 8))

        y_err = grouped['value_efficiency_std'] * ci_multiplier
        ax1.errorbar(x, grouped['value_efficiency_mean'].values, yerr=y_err.values,
                     fmt='o', color='salmon', capsize=5, label='Value Efficiency')
        ax1.set_ylabel('Value Efficiency (%)', color='salmon')
        ax1.tick_params(axis='y', labelcolor='salmon')

        for i, (mean, err) in enumerate(zip(grouped['value_efficiency_mean'].values, y_err.values)):
            ax1.text(x[i], mean + err + 0.5, f"{mean:.2f}%", ha='center', fontsize=9, color='salmon')

        ax2 = ax1.twinx()
        total_bil = grouped['settled_amount_mean'].values / 1e9

        ax2.plot(x, total_bil, 's-', color='darkgreen', label='Total Settled')
        ax2.set_ylabel('Settled Amount (Billions €)', color='darkgreen')
        ax2.tick_params(axis='y', labelcolor='darkgreen')

        for i in range(len(x)):
            ax2.text(x[i], total_bil[i] + 0.05, f"{total_bil[i]:.2f}B", ha='center', fontsize=9, color='darkgreen')

        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Config {cfg}" for cfg in configs], rotation=45, ha='right')
        plt.title('Value Efficiency (95% CI) vs Total Settled Amount (On Time or Late)')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "value_efficiency_ci.png"), dpi=300)
        plt.close()

    def _plot_value_efficiency_ontime_only(self, df):
        grouped = df.groupby("config").agg(
            value_efficiency_mean=("value_efficiency", "mean"),
            value_efficiency_std=("value_efficiency", "std"),
            settled_ontime_mean=("settled_on_time_amount", "mean")
        ).sort_index()

        configs = grouped.index.tolist()
        x = np.arange(len(configs))
        ci_multiplier = 1.96 / np.sqrt(len(df.groupby("config")))

        fig, ax1 = plt.subplots(figsize=(14, 8))

        y_err = grouped['value_efficiency_std'] * ci_multiplier
        ax1.errorbar(x, grouped['value_efficiency_mean'].values, yerr=y_err.values,
                     fmt='o', color='salmon', capsize=5, label='Value Efficiency')
        ax1.set_ylabel('Value Efficiency (%)', color='salmon')
        ax1.tick_params(axis='y', labelcolor='salmon')

        for i, (mean, err) in enumerate(zip(grouped['value_efficiency_mean'].values, y_err.values)):
            ax1.text(x[i], mean + err + 0.5, f"{mean:.2f}%", ha='center', fontsize=9, color='salmon')

        ax2 = ax1.twinx()
        ontime_bil = grouped['settled_ontime_mean'].values / 1e9

        ax2.plot(x, ontime_bil, 'o--', color='teal', label='Settled On Time')
        ax2.set_ylabel('Settled On Time Amount (Billions €)', color='teal')
        ax2.tick_params(axis='y', labelcolor='teal')

        for i in range(len(x)):
            ax2.text(x[i], ontime_bil[i] + 0.05, f"{ontime_bil[i]:.2f}B", ha='center', fontsize=9, color='teal')

        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Config {cfg}" for cfg in configs], rotation=45, ha='right')
        plt.title('Value Efficiency (95% CI) vs Settled On Time Amount')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "value_efficiency_ontime_ci.png"), dpi=300)
        plt.close()


    def _plot_combined_efficiency(self, df):
        grouped = df.groupby("config").agg(
            instruction_efficiency_mean=("instruction_efficiency", "mean"),
            instruction_efficiency_std=("instruction_efficiency", "std"),
            mothers_settled_mean=("mothers_effectively_settled", "mean"),
            value_efficiency_mean=("value_efficiency", "mean"),
            value_efficiency_std=("value_efficiency", "std"),
            settled_ontime_mean=("settled_on_time_amount", "mean")
        ).sort_index()

        configs = grouped.index.tolist()
        x = np.arange(len(configs))
        ci_multiplier = 1.96 / np.sqrt(len(df.groupby("config")))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

        y_err1 = grouped['instruction_efficiency_std'] * ci_multiplier
        ax1.errorbar(x, grouped['instruction_efficiency_mean'].values, yerr=y_err1.values,
                     fmt='o', color='deepskyblue', capsize=5, label='Instruction Efficiency')
        ax1.set_ylabel('Instruction Efficiency (%)', color='deepskyblue')
        ax1.tick_params(axis='y', labelcolor='deepskyblue')

        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, grouped['mothers_settled_mean'].values, 's-', color='green', label='Mothers Settled On Time')
        ax1_twin.set_ylabel('Mothers Settled (On Time or Late)', color='green')
        ax1_twin.tick_params(axis='y', labelcolor='green')

        for i in range(len(x)):
            ax1.text(x[i], grouped['instruction_efficiency_mean'].values[i] + y_err1.values[i] + 0.5,
                     f"{grouped['instruction_efficiency_mean'].values[i]:.2f}%", ha='center', fontsize=9, color='deepskyblue')
            ax1_twin.text(x[i], grouped['mothers_settled_mean'].values[i] + 1,
                          f"{int(grouped['mothers_settled_mean'].values[i])}", ha='center', fontsize=9, color='green')

        ax1.set_title('Instruction Efficiency (95% CI) vs Mothers Settled (On Time or Late)')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        y_err2 = grouped['value_efficiency_std'] * ci_multiplier
        ax2.errorbar(x, grouped['value_efficiency_mean'].values, yerr=y_err2.values,
                     fmt='o', color='salmon', capsize=5, label='Value Efficiency')
        ax2.set_ylabel('Value Efficiency (%)', color='salmon')
        ax2.tick_params(axis='y', labelcolor='salmon')

        ax2_twin = ax2.twinx()
        ontime_bil = grouped['settled_ontime_mean'].values / 1e9
        ax2_twin.plot(x, ontime_bil, 'o--', color='teal', label='Settled On Time')
        ax2_twin.set_ylabel('Settled On Time Amount (Billions €)', color='teal')
        ax2_twin.tick_params(axis='y', labelcolor='teal')

        for i in range(len(x)):
            ax2.text(x[i], grouped['value_efficiency_mean'].values[i] + y_err2.values[i] + 0.5,
                     f"{grouped['value_efficiency_mean'].values[i]:.2f}%", ha='center', fontsize=9, color='salmon')
            ax2_twin.text(x[i], ontime_bil[i] + 0.05, f"{ontime_bil[i]:.2f}B", ha='center', fontsize=9, color='teal')

        ax2.set_title('Value Efficiency (95% CI) vs Settled On Time Amount')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Config {cfg}" for cfg in configs], rotation=45, ha='right')
        ax2.grid(axis='y', linestyle='--', alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax2_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "combined_efficiency_ci.png"), dpi=300)
        plt.close()

