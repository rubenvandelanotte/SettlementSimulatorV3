import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

class EfficiencyPerParticipantAnalyzer:
    """
    Analyze and visualize settlement efficiencies (instruction & value) per participant
    across scenarios with varying numbers of institutions allowing partial settlements.

    Provides separate methods for data creation and visualization per metric.
    """
    DEFAULT_FIGSIZE = (12, 7)
    DEFAULT_DPI = 200

    def __init__(self, input_dir, output_dir, suite):
        self.statistics = suite.statistics
        self.output_dir = os.path.join(output_dir, "partial")
        os.makedirs(self.output_dir, exist_ok=True)

    def create_dataframe(self):
        """
        Build DataFrame for instruction efficiency per participant:
          - partialsallowed_count (int)
          - participant_id (str or int)
          - efficiency (float)
        """
        return self._create_metric_dataframe('instruction_efficiency')

    def create_value_dataframe(self):
        """
        Build DataFrame for value efficiency per participant:
          - partialsallowed_count (int)
          - participant_id (str or int)
          - efficiency (float)
        """
        return self._create_metric_dataframe('value_efficiency')

    def _create_metric_dataframe(self, metric_key):
        records = []
        for fname, data in self.statistics.items():
            meta = data.get("config_metadata", {})
            count = meta.get("config_id")
            if count is None:
                warnings.warn(f"Missing config_id in {fname}, skipping.")
                continue

            part_metrics = data.get("participant_metrics") or {}
            if not part_metrics:
                warnings.warn(f"No participant_metrics in {fname}, skipping.")
                continue

            for part_id, metrics in part_metrics.items():
                eff = metrics.get(metric_key)
                if eff is None:
                    continue
                records.append({
                    "partialsallowed_count": count,
                    "participant_id": part_id,
                    "efficiency": eff
                })

        df = pd.DataFrame(records)
        if df.empty:
            return df
        # Ensure consistent types
        df['participant_id'] = df['participant_id'].astype(str)
        df.sort_values(["partialsallowed_count", "participant_id"], inplace=True)
        return df

    def _format_axes(self, ax):
        """
        Apply common formatting: grid, fonts.
        """
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

    def plot_mean_bar(self, df, metric_name='Instruction'):
        """
        Plot mean efficiency per scenario (partialsallowed_count) as a bar chart.
        """
        mean_df = df.groupby('partialsallowed_count')['efficiency'].mean().reset_index()
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        bars = ax.bar(
            mean_df['partialsallowed_count'].astype(str),
            mean_df['efficiency'],
            edgecolor='black'
        )
        ax.set_xlabel('Allowed Partials Count', fontsize=12)
        ax.set_ylabel(f'Avg {metric_name} Efficiency (%)', fontsize=12)
        ax.set_title(f'Mean {metric_name} Efficiency per Scenario', fontsize=14)
        self._format_axes(ax)
        # Annotate
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=10)
        fig.tight_layout()
        fname = f'participant_mean_bar_{metric_name.lower()}.png'
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, dpi=self.DEFAULT_DPI)
        plt.close(fig)
        print(f"[INFO] Saved mean bar chart for {metric_name.lower()} to {path}")

    def plot_distribution(self, df, metric_name='Instruction'):
        """
        Plot distribution of participant efficiencies across scenarios using boxplots.
        """
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        # Prepare data for boxplot: list of lists per scenario
        order = sorted(df['partialsallowed_count'].unique())
        data = [
            df[df['partialsallowed_count']==cnt]['efficiency'].values
            for cnt in order
        ]
        ax.boxplot(data, labels=[str(cnt) for cnt in order], patch_artist=True)
        ax.set_xlabel('Allowed Partials Count', fontsize=12)
        ax.set_ylabel(f'{metric_name} Efficiency (%)', fontsize=12)
        ax.set_title(f'Distribution of {metric_name} Efficiency per Scenario', fontsize=14)
        self._format_axes(ax)
        fig.tight_layout()
        fname = f'participant_distribution_box_{metric_name.lower()}.png'
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, dpi=self.DEFAULT_DPI)
        plt.close(fig)
        print(f"[INFO] Saved distribution boxplot for {metric_name.lower()} to {path}")

    def plot_heatmap(self, df, metric_name='Instruction'):
        """
        Plot heatmap of participant efficiencies: rows=participant, cols=scenario.
        """
        pivot = df.pivot(
            index='participant_id',
            columns='partialsallowed_count',
            values='efficiency'
        )
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        c = ax.pcolormesh(pivot.columns.astype(float),
                          range(len(pivot.index)),
                          pivot.values, shading='auto')
        fig.colorbar(c, ax=ax, label='Efficiency (%)')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel('Allowed Partials Count', fontsize=12)
        ax.set_ylabel('Participant ID', fontsize=12)
        ax.set_title(f'{metric_name} Efficiency Heatmap by Participant and Scenario', fontsize=14)
        self._format_axes(ax)
        fig.tight_layout()
        fname = f'participant_efficiency_heatmap_{metric_name.lower()}.png'
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, dpi=self.DEFAULT_DPI)
        plt.close(fig)
        print(f"[INFO] Saved heatmap for {metric_name.lower()} to {path}")

    def run(self):
        # Instruction efficiency plots
        df_inst = self.create_dataframe()
        if df_inst.empty:
            raise RuntimeError("No instruction efficiency data for participants.")
        self.plot_mean_bar(df_inst, metric_name='Instruction')
        self.plot_distribution(df_inst, metric_name='Instruction')
        self.plot_heatmap(df_inst, metric_name='Instruction')

        # Value efficiency plots
        df_val = self.create_value_dataframe()
        if df_val.empty:
            warnings.warn("No value efficiency data for participants.")
        else:
            self.plot_mean_bar(df_val, metric_name='Value')
            self.plot_distribution(df_val, metric_name='Value')
            self.plot_heatmap(df_val, metric_name='Value')

# End of EfficiencyPerParticipantAnalyzer