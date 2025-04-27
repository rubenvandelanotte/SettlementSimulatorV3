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
        self._plot_combined_efficiency(df)

    def _build_dataframe(self):
        records = []
        for stat_file, stat_data in self.suite.statistics.items():
            config_key = "Unknown"
            if "_config" in stat_file:
                config_key = stat_file.split("_config")[-1].split("_run")[0]

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
                "settled_amount": settled_amount
            })

        return pd.DataFrame(records)

    def _plot_instruction_efficiency(self, df):
        grouped = df.groupby("config").agg(
            instruction_efficiency_mean=("instruction_efficiency", "mean"),
            instruction_efficiency_std=("instruction_efficiency", "std"),
            mothers_settled_mean=("mothers_effectively_settled", "mean")
        )

        configs = grouped.index.tolist()
        x = np.arange(len(configs))
        ci_multiplier = 1.96 / np.sqrt(len(df.groupby("config")))

        fig, ax1 = plt.subplots(figsize=(14, 8))

        ax1.errorbar(x, grouped['instruction_efficiency_mean'],
                     yerr=grouped['instruction_efficiency_std'] * ci_multiplier,
                     fmt='o', color='deepskyblue', capsize=5)
        ax1.set_ylabel('Instruction Efficiency (%)', color='deepskyblue')
        ax1.tick_params(axis='y', labelcolor='deepskyblue')

        ax2 = ax1.twinx()
        ax2.plot(x, grouped['mothers_settled_mean'], 's-', color='green')
        ax2.set_ylabel('Mothers Effectively Settled', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        plt.title('Instruction Efficiency (95% CI) vs Mothers Effectively Settled')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "instruction_efficiency_ci.png"), dpi=300)
        plt.close()

    def _plot_value_efficiency(self, df):
        grouped = df.groupby("config").agg(
            value_efficiency_mean=("value_efficiency", "mean"),
            value_efficiency_std=("value_efficiency", "std"),
            settled_amount_mean=("settled_amount", "mean")
        )

        configs = grouped.index.tolist()
        x = np.arange(len(configs))
        ci_multiplier = 1.96 / np.sqrt(len(df.groupby("config")))

        fig, ax1 = plt.subplots(figsize=(14, 8))

        ax1.errorbar(x, grouped['value_efficiency_mean'],
                     yerr=grouped['value_efficiency_std'] * ci_multiplier,
                     fmt='o', color='salmon', capsize=5)
        ax1.set_ylabel('Value Efficiency (%)', color='salmon')
        ax1.tick_params(axis='y', labelcolor='salmon')

        ax2 = ax1.twinx()
        ax2.plot(x, grouped['settled_amount_mean'], 's-', color='darkgreen')
        ax2.set_ylabel('Settled Amount', color='darkgreen')
        ax2.tick_params(axis='y', labelcolor='darkgreen')

        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        plt.title('Value Efficiency (95% CI) vs Settled Amount')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "value_efficiency_ci.png"), dpi=300)
        plt.close()

    def _plot_combined_efficiency(self, df):
        grouped = df.groupby("config").agg(
            instruction_efficiency_mean=("instruction_efficiency", "mean"),
            instruction_efficiency_std=("instruction_efficiency", "std"),
            mothers_settled_mean=("mothers_effectively_settled", "mean"),
            value_efficiency_mean=("value_efficiency", "mean"),
            value_efficiency_std=("value_efficiency", "std"),
            settled_amount_mean=("settled_amount", "mean")
        )

        configs = grouped.index.tolist()
        x = np.arange(len(configs))
        ci_multiplier = 1.96 / np.sqrt(len(df.groupby("config")))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

        ax1.errorbar(x, grouped['instruction_efficiency_mean'],
                     yerr=grouped['instruction_efficiency_std'] * ci_multiplier,
                     fmt='o', color='deepskyblue', capsize=5)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, grouped['mothers_settled_mean'], 's-', color='green')
        ax1.set_ylabel('Instruction Efficiency (%)', color='deepskyblue')
        ax1_twin.set_ylabel('Mothers Effectively Settled', color='green')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.set_title('Instruction Efficiency (95% CI) vs Mothers Effectively Settled')

        ax2.errorbar(x, grouped['value_efficiency_mean'],
                     yerr=grouped['value_efficiency_std'] * ci_multiplier,
                     fmt='o', color='salmon', capsize=5)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, grouped['settled_amount_mean'], 's-', color='darkgreen')
        ax2.set_ylabel('Value Efficiency (%)', color='salmon')
        ax2_twin.set_ylabel('Settled Amount', color='darkgreen')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.set_title('Value Efficiency (95% CI) vs Settled Amount')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "combined_efficiency_ci.png"), dpi=300)
        plt.close()
