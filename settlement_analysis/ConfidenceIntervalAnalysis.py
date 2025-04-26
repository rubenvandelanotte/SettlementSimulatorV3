import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ConfidenceIntervalAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "confidence_interval_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        df = self._build_dataframe()
        if df.empty:
            print("[WARNING] No valid statistics data to analyze.")
            return

        self._plot_instruction_efficiency_vs_settled(df)
        self._plot_value_efficiency_vs_settled(df)
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

    def _plot_instruction_efficiency_vs_settled(self, df):
        grouped = df.groupby("config").agg({
            "instruction_efficiency": ["mean", "std"],
            "mothers_effectively_settled": "mean"
        })
        grouped.columns = ['instruction_efficiency_mean', 'instruction_efficiency_std', 'mothers_effectively_settled_mean']
        configs = grouped.index.tolist()
        x = np.arange(len(configs))

        fig, ax1 = plt.subplots(figsize=(14, 8))

        ax1.errorbar(x, grouped['instruction_efficiency_mean'], yerr=grouped['instruction_efficiency_std'], fmt='o', color='deepskyblue', capsize=5, label='Instruction Efficiency (%)')
        ax1.set_ylabel('Instruction Efficiency (%)', color='deepskyblue')
        ax1.tick_params(axis='y', labelcolor='deepskyblue')

        ax2 = ax1.twinx()
        ax2.plot(x, grouped['mothers_effectively_settled_mean'], 's-', color='green', label='Mothers Effectively Settled')
        ax2.set_ylabel('Mothers Effectively Settled', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        plt.title('Confidence Intervals for Instruction Efficiency by Configuration')
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "instruction_efficiency_ci.png"), dpi=300)
        plt.close()

    def _plot_value_efficiency_vs_settled(self, df):
        grouped = df.groupby("config").agg({
            "value_efficiency": ["mean", "std"],
            "settled_amount": "mean"
        })
        grouped.columns = ['value_efficiency_mean', 'value_efficiency_std', 'settled_amount_mean']
        configs = grouped.index.tolist()
        x = np.arange(len(configs))

        fig, ax1 = plt.subplots(figsize=(14, 8))

        ax1.errorbar(x, grouped['value_efficiency_mean'], yerr=grouped['value_efficiency_std'], fmt='o', color='salmon', capsize=5, label='Value Efficiency (%)')
        ax1.set_ylabel('Value Efficiency (%)', color='salmon')
        ax1.tick_params(axis='y', labelcolor='salmon')

        ax2 = ax1.twinx()
        ax2.plot(x, grouped['settled_amount_mean'], 's-', color='darkgreen', label='Settled Amount (€)')
        ax2.set_ylabel('Settled Amount (€)', color='darkgreen')
        ax2.tick_params(axis='y', labelcolor='darkgreen')

        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        plt.title('Confidence Intervals for Value Efficiency by Configuration')
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "value_efficiency_ci.png"), dpi=300)
        plt.close()

    def _plot_combined_efficiency(self, df):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

        # First subplot: Instruction Efficiency
        grouped_instr = df.groupby("config").agg({
            "instruction_efficiency": ["mean", "std"],
            "mothers_effectively_settled": "mean"
        })
        grouped_instr.columns = ['instruction_efficiency_mean', 'instruction_efficiency_std', 'mothers_effectively_settled_mean']
        configs = grouped_instr.index.tolist()
        x = np.arange(len(configs))

        ax1.errorbar(x, grouped_instr['instruction_efficiency_mean'], yerr=grouped_instr['instruction_efficiency_std'], fmt='o', color='deepskyblue', capsize=5)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, grouped_instr['mothers_effectively_settled_mean'], 's-', color='green')
        ax1.set_ylabel('Instruction Efficiency (%)', color='deepskyblue')
        ax1_twin.set_ylabel('Mothers Effectively Settled', color='green')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.set_title('Instruction Efficiency vs Mothers Effectively Settled')

        # Second subplot: Value Efficiency
        grouped_val = df.groupby("config").agg({
            "value_efficiency": ["mean", "std"],
            "settled_amount": "mean"
        })
        grouped_val.columns = ['value_efficiency_mean', 'value_efficiency_std', 'settled_amount_mean']

        ax2.errorbar(x, grouped_val['value_efficiency_mean'], yerr=grouped_val['value_efficiency_std'], fmt='o', color='salmon', capsize=5)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, grouped_val['settled_amount_mean'], 's-', color='darkgreen')
        ax2.set_ylabel('Value Efficiency (%)', color='salmon')
        ax2_twin.set_ylabel('Settled Amount (€)', color='darkgreen')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.set_title('Value Efficiency vs Settled Amount')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "combined_efficiency_ci.png"), dpi=300)
        plt.close()
