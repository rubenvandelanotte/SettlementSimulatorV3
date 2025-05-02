import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

class EfficiencyPerDayAnalyzer:
    """
    Analyze and visualize daily settlement efficiencies (instruction & value)
    across scenarios with varying numbers of institutions allowing partial settlements.

    Provides separate methods for data creation and visualization per metric.
    """
    DEFAULT_FIGSIZE = (12, 7)
    DEFAULT_DPI = 200
    DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, input_dir, output_dir, suite):
        self.statistics = suite.statistics
        self.output_dir = os.path.join(output_dir, "partial")
        os.makedirs(self.output_dir, exist_ok=True)

    def create_dataframe(self):
        """
        Build DataFrame for instruction efficiency:
          - partialsallowed_count (int)
          - day (datetime)
          - efficiency (float)
        """
        return self._create_metric_dataframe('instruction_efficiency')

    def create_value_dataframe(self):
        """
        Build DataFrame for value efficiency:
          - partialsallowed_count (int)
          - day (datetime)
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

            daily = data.get("daily_metrics") or {}
            if not daily:
                warnings.warn(f"No daily_metrics in {fname}, skipping.")
                continue

            for day_str, effs in daily.items():
                eff = effs.get(metric_key)
                if eff is None:
                    continue
                records.append({
                    "partialsallowed_count": count,
                    "day": pd.to_datetime(day_str),
                    "efficiency": eff
                })

        df = pd.DataFrame(records)
        if df.empty:
            return df
        df.sort_values(["partialsallowed_count", "day"], inplace=True)
        return df

    def _format_axes(self, ax):
        """
        Apply common formatting: date formatting, grid, fonts.
        """
        locator = AutoDateLocator()
        formatter = DateFormatter(self.DATE_FORMAT)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

    def plot_time_series(self, df, metric_name='Instruction'):
        """
        Plot daily efficiency time series by partials count.
        metric_name: label for chart and filename prefix.
        """
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        for count, grp in df.groupby('partialsallowed_count'):
            ax.plot(grp['day'], grp['efficiency'], marker='o', linewidth=2,
                    label=f"{count} partials")

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{metric_name} Efficiency (%)', fontsize=12)
        ax.set_title(f'Daily {metric_name} Efficiency by Partial Settlement Allowance', fontsize=14)
        ax.legend(title='Allowed Partials', fontsize=10, title_fontsize=11)
        self._format_axes(ax)
        fig.tight_layout()

        fname = f'efficiency_time_series_{metric_name.lower()}.png'
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, dpi=self.DEFAULT_DPI)
        plt.close(fig)
        print(f"[INFO] Saved {metric_name.lower()} time series to {path}")

    def plot_average_bar(self, df, metric_name='Instruction'):
        """
        Plot average efficiency per partials count as bar chart.
        """
        avg = df.groupby('partialsallowed_count')['efficiency'].mean().reset_index()
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        bars = ax.bar(avg['partialsallowed_count'].astype(str), avg['efficiency'], edgecolor='black')

        ax.set_xlabel('Allowed Partials Count', fontsize=12)
        ax.set_ylabel(f'Avg {metric_name} Efficiency (%)', fontsize=12)
        ax.set_title(f'Average Daily {metric_name} Efficiency by Partial Settlement Allowance', fontsize=14)
        self._format_axes(ax)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)

        fig.tight_layout()
        fname = f'efficiency_avg_bar_{metric_name.lower()}.png'
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, dpi=self.DEFAULT_DPI)
        plt.close(fig)
        print(f"[INFO] Saved avg bar chart for {metric_name.lower()} to {path}")

    def plot_heatmap(self, df, metric_name='Instruction'):
        """
        Plot pivot heatmap: rows=day, cols=partials count.
        """
        pivot = df.pivot(index='day', columns='partialsallowed_count', values='efficiency')
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        c = ax.pcolormesh(pivot.index, pivot.columns, pivot.T, shading='auto')
        fig.colorbar(c, ax=ax, label='Efficiency (%)')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Allowed Partials', fontsize=12)
        ax.set_title(f'{metric_name} Efficiency Heatmap by Day and Partials Allowance', fontsize=14)
        ax.set_yticks(pivot.columns)
        ax.set_yticklabels([str(int(y)) for y in pivot.columns], fontsize=10)
        self._format_axes(ax)

        fig.tight_layout()
        fname = f'efficiency_heatmap_{metric_name.lower()}.png'
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, dpi=self.DEFAULT_DPI)
        plt.close(fig)
        print(f"[INFO] Saved heatmap for {metric_name.lower()} to {path}")

    def run(self):
        # Instruction efficiency
        df_inst = self.create_dataframe()
        if df_inst.empty:
            raise RuntimeError("No instruction efficiency data to plot.")
        self.plot_time_series(df_inst, metric_name='Instruction')
        self.plot_average_bar(df_inst, metric_name='Instruction')
        self.plot_heatmap(df_inst, metric_name='Instruction')

        # Value efficiency
        df_val = self.create_value_dataframe()
        if df_val.empty:
            warnings.warn("No value efficiency data to plot.")
        else:
            self.plot_time_series(df_val, metric_name='Value')
            self.plot_average_bar(df_val, metric_name='Value')
            self.plot_heatmap(df_val, metric_name='Value')

# End of EfficiencyPerDayAnalyzer