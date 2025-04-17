import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


class MinSettlementAmountVisualizer:
    """
    A class to visualize the results of the minimum settlement amount analysis simulations.
    Focuses on comparing different minimum settlement percentage configurations averaged across runs.
    """

    def __init__(self, results_csv="min_settlement_files/min_settlement_amount_final_results.csv",
                 output_dir="min_settlement_visualizations"):
        """
        Initialize the visualizer with the path to the results CSV file.

        Args:
            results_csv: Path to the CSV file containing simulation results
            output_dir: Directory where visualizations will be saved
        """
        self.results_file = results_csv
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load and prepare data
        self.load_data()

    def load_data(self):
        try:
            # Load the CSV file
            df = pd.read_csv(self.results_file)
            print(f"Loaded data from {self.results_file}: {len(df)} rows")

            # Print column information to help debug
            print("Columns in the dataframe:")
            for col in df.columns:
                print(f"  - {col}: {df[col].dtype}")

            # Group by min_settlement_percentage
            grouped = df.groupby('min_settlement_percentage')

            # Create a dictionary to hold our statistics
            stats_dict = {}

            # Calculate statistics for each column we need
            columns_to_process = [
                'instruction_efficiency', 'value_efficiency', 'runtime_seconds',
                'settled_count', 'settled_amount', 'partial_settlements',
                'memory_usage_mb'
            ]

            for col in columns_to_process:
                if col in df.columns:
                    stats_dict[f'{col}_mean'] = grouped[col].mean()
                    stats_dict[f'{col}_std'] = grouped[col].std()
                    stats_dict[f'{col}_count'] = grouped[col].count()

            # Convert dictionary to dataframe
            self.config_stats = pd.DataFrame(stats_dict)

            # Save the raw data for other analyses
            self.df = df

            print(f"Calculated statistics for {len(self.config_stats)} configurations")
            print("Columns in self.config_stats:")
            for col in self.config_stats.columns:
                print(f"  - {col}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def plot_runtime_vs_efficiency(self):
        """
        Plot the relationship between runtime and efficiency metrics.
        Creates two plots: one for instruction efficiency and one for value efficiency.
        """
        # Prepare the data for plotting
        x = self.config_stats['runtime_seconds_mean']

        # Plotting variables
        percentages = self.config_stats.index.tolist()

        # Create figure for instruction efficiency
        plt.figure(figsize=(12, 8))

        # Calculate error bars (standard deviation)
        y = self.config_stats['instruction_efficiency_mean']
        yerr = self.config_stats['instruction_efficiency_std']

        # Create scatter plot with error bars
        plt.errorbar(x, y, yerr=yerr, fmt='o-', linewidth=2, markersize=10,
                     capsize=5, color='blue', label='Instruction Efficiency')

        # Add percentage labels to each point
        for i, pct in enumerate(percentages):
            plt.annotate(f"Min %: {pct * 100:.1f}%",
                         (x.iloc[i], y.iloc[i]),
                         xytext=(10, 5),
                         textcoords='offset points',
                         fontsize=12)

        # Add trendline
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.7,
                 label=f"Trend: y = {z[0]:.4f}x + {z[1]:.2f}")

        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]

        # Customize plot
        plt.title(f'Runtime vs Instruction Efficiency\nCorrelation: {correlation:.4f}', fontsize=16)
        plt.xlabel('Average Runtime (seconds)', fontsize=14)
        plt.ylabel('Average Instruction Efficiency (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'runtime_vs_instruction_efficiency.png'), dpi=300)
        plt.close()

        # Create figure for value efficiency
        plt.figure(figsize=(12, 8))

        # Calculate error bars (standard deviation)
        y = self.config_stats['value_efficiency_mean']
        yerr = self.config_stats['value_efficiency_std']

        # Create scatter plot with error bars
        plt.errorbar(x, y, yerr=yerr, fmt='o-', linewidth=2, markersize=10,
                     capsize=5, color='green', label='Value Efficiency')

        # Add percentage labels to each point
        for i, pct in enumerate(percentages):
            plt.annotate(f"Min %: {pct * 100:.1f}%",
                         (x.iloc[i], y.iloc[i]),
                         xytext=(10, 5),
                         textcoords='offset points',
                         fontsize=12)

        # Add trendline
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.7,
                 label=f"Trend: y = {z[0]:.4f}x + {z[1]:.2f}")

        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]

        # Customize plot
        plt.title(f'Runtime vs Value Efficiency\nCorrelation: {correlation:.4f}', fontsize=16)
        plt.xlabel('Average Runtime (seconds)', fontsize=14)
        plt.ylabel('Average Value Efficiency (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'runtime_vs_value_efficiency.png'), dpi=300)
        plt.close()

        print(f"Created runtime vs efficiency plots in {self.output_dir}")

    def plot_efficiency_vs_percentage(self):
        """
        Plot instruction and value efficiency against minimum settlement percentage.
        Shows if there's a point of diminishing returns.
        """
        # Prepare the data for plotting
        percentages = self.config_stats.index.tolist()
        # Format percentages for display
        percentage_labels = [f"{pct * 100:.1f}%" for pct in percentages]

        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # First axis for instruction efficiency
        instruction_eff = self.config_stats['instruction_efficiency_mean']
        instruction_err = self.config_stats['instruction_efficiency_std']

        # Plot instruction efficiency
        color = 'blue'
        ax1.errorbar(percentages, instruction_eff, yerr=instruction_err, fmt='o-',
                     linewidth=2, markersize=10, capsize=5, color=color,
                     label='Instruction Efficiency')
        ax1.set_xlabel('Minimum Settlement Percentage', fontsize=14)
        ax1.set_ylabel('Instruction Efficiency (%)', color=color, fontsize=14)
        ax1.tick_params(axis='y', labelcolor=color)

        # Format x-axis tick labels as percentages
        ax1.set_xticks(percentages)
        ax1.set_xticklabels(percentage_labels)

        # Second y-axis for value efficiency
        ax2 = ax1.twinx()
        value_eff = self.config_stats['value_efficiency_mean']
        value_err = self.config_stats['value_efficiency_std']

        # Plot value efficiency
        color = 'green'
        ax2.errorbar(percentages, value_eff, yerr=value_err, fmt='s--',
                     linewidth=2, markersize=10, capsize=5, color=color,
                     label='Value Efficiency')
        ax2.set_ylabel('Value Efficiency (%)', color=color, fontsize=14)
        ax2.tick_params(axis='y', labelcolor=color)

        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)

        # Add title
        plt.title('Efficiency Metrics vs Minimum Settlement Percentage', fontsize=16)
        plt.grid(True, alpha=0.3)

        # Add data point labels
        for i, pct in enumerate(percentages):
            # Label instruction efficiency points
            ax1.annotate(f"{instruction_eff.iloc[i]:.2f}%",
                         (pct, instruction_eff.iloc[i]),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', fontsize=10, color='blue')

            # Label value efficiency points
            ax2.annotate(f"{value_eff.iloc[i]:.2f}%",
                         (pct, value_eff.iloc[i]),
                         xytext=(0, -15), textcoords='offset points',
                         ha='center', fontsize=10, color='green')

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'efficiency_vs_percentage.png'), dpi=300)
        plt.close()

        print(f"Created efficiency vs percentage plot in {self.output_dir}")

    def plot_settled_amount_vs_percentage(self):
        """
        Plot the total settled amount against minimum settlement percentage.
        Shows how settlement magnitude changes with different percentage limits.
        """
        if 'settled_amount_mean' not in self.config_stats.columns:
            print("Warning: settled_amount data not available. Skipping this plot.")
            return

        # Prepare the data for plotting
        percentages = self.config_stats.index.tolist()
        percentage_labels = [f"{pct * 100:.1f}%" for pct in percentages]
        values = self.config_stats['settled_amount_mean']
        errors = self.config_stats['settled_amount_std']

        # Create the figure
        plt.figure(figsize=(12, 8))

        # Create bar chart with error bars
        bars = plt.bar(percentage_labels, values, yerr=errors, capsize=5,
                       color='skyblue', edgecolor='navy', alpha=0.7)

        # Add data labels on top of bars
        def format_value(val):
            if val >= 1e12:
                return f'{val / 1e12:.2f}T'
            elif val >= 1e9:
                return f'{val / 1e9:.2f}B'
            elif val >= 1e6:
                return f'{val / 1e6:.2f}M'
            else:
                return f'{val:.0f}'

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     format_value(height),
                     ha='center', va='bottom', fontsize=12)

        # Customize plot
        plt.xlabel('Minimum Settlement Percentage', fontsize=14)
        plt.ylabel('Average Settled Amount', fontsize=14)
        plt.title('Settlement Amount vs Minimum Settlement Percentage', fontsize=16)
        plt.grid(axis='y', alpha=0.3)

        # Add a line showing the trend
        plt.plot(range(len(percentages)), values, 'ro-', linewidth=2, alpha=0.7)

        # Format y-axis to show abbreviated large numbers
        from matplotlib.ticker import FuncFormatter
        def y_fmt(val, pos):
            if val >= 1e12:
                return f'{val / 1e12:.1f}T'
            elif val >= 1e9:
                return f'{val / 1e9:.1f}B'
            elif val >= 1e6:
                return f'{val / 1e6:.1f}M'
            else:
                return f'{val:.0f}'

        plt.gca().yaxis.set_major_formatter(FuncFormatter(y_fmt))

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'settled_amount_vs_percentage.png'), dpi=300)
        plt.close()

        print(f"Created settled amount vs percentage plot in {self.output_dir}")

    def plot_partial_settlements_vs_percentage(self):
        """
        Plot the number of partial settlements against minimum settlement percentage.
        Shows how the frequency of partial settlements changes with different percentage limits.
        """
        if 'partial_settlements_mean' not in self.config_stats.columns:
            print("Warning: partial_settlements data not available. Skipping this plot.")
            return

        # Prepare the data for plotting
        percentages = self.config_stats.index.tolist()
        percentage_labels = [f"{pct * 100:.1f}%" for pct in percentages]
        values = self.config_stats['partial_settlements_mean']
        errors = self.config_stats['partial_settlements_std']

        # Create the figure
        plt.figure(figsize=(12, 8))

        # Create bar chart with error bars
        bars = plt.bar(percentage_labels, values, yerr=errors, capsize=5,
                       color='lightgreen', edgecolor='darkgreen', alpha=0.7)

        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.0f}',
                     ha='center', va='bottom', fontsize=12)

        # Customize plot
        plt.xlabel('Minimum Settlement Percentage', fontsize=14)
        plt.ylabel('Average Number of Partial Settlements', fontsize=14)
        plt.title('Partial Settlements vs Minimum Settlement Percentage', fontsize=16)
        plt.grid(axis='y', alpha=0.3)

        # Add a line showing the trend
        plt.plot(range(len(percentages)), values, 'ro-', linewidth=2, alpha=0.7)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'partial_settlements_vs_percentage.png'), dpi=300)
        plt.close()

        print(f"Created partial settlements vs percentage plot in {self.output_dir}")

    def plot_memory_usage_vs_percentage(self):
        """
        Plot memory usage against minimum settlement percentage.
        Shows how computational resources scale with percentage configuration.
        """
        if 'memory_usage_mb_mean' not in self.config_stats.columns:
            print("Warning: memory_usage_mb data not available. Skipping this plot.")
            return

        # Prepare the data for plotting
        percentages = self.config_stats.index.tolist()
        percentage_labels = [f"{pct * 100:.1f}%" for pct in percentages]
        values = self.config_stats['memory_usage_mb_mean']
        errors = self.config_stats['memory_usage_mb_std']

        # Create the figure
        plt.figure(figsize=(12, 8))

        # Create scatter plot with error bars
        plt.errorbar(percentages, values, yerr=errors, fmt='o-',
                     linewidth=2, markersize=10, capsize=5,
                     color='orange', label='Memory Usage')

        # Add data labels
        for i, pct in enumerate(percentages):
            plt.annotate(f"{values.iloc[i]:.1f} MB",
                         (pct, values.iloc[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=12)

        # Customize plot
        plt.xlabel('Minimum Settlement Percentage', fontsize=14)
        plt.xticks(percentages, percentage_labels)
        plt.ylabel('Average Memory Usage (MB)', fontsize=14)
        plt.title('Memory Usage vs Minimum Settlement Percentage', fontsize=16)
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'memory_usage_vs_percentage.png'), dpi=300)
        plt.close()

        print(f"Created memory usage vs percentage plot in {self.output_dir}")

    def plot_runtime_scaling(self):
        """
        Plot the relationship between minimum settlement percentage and runtime.
        Helps identify how computation time scales with the percentage threshold.
        """
        # Prepare the data for plotting
        percentages = self.config_stats.index.tolist()
        percentage_labels = [f"{pct * 100:.1f}%" for pct in percentages]
        runtimes = self.config_stats['runtime_seconds_mean']

        # Create the figure
        plt.figure(figsize=(12, 8))

        # Create scatter plot
        plt.plot(percentages, runtimes, 'o-', linewidth=2, markersize=10, color='red')

        # Add data labels
        for i, pct in enumerate(percentages):
            plt.annotate(f"{runtimes.iloc[i]:.2f}s",
                         (pct, runtimes.iloc[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=12)

        # Customize plot
        plt.xlabel('Minimum Settlement Percentage', fontsize=14)
        plt.xticks(percentages, percentage_labels)
        plt.ylabel('Runtime (seconds)', fontsize=14)
        plt.title('Runtime vs Minimum Settlement Percentage', fontsize=16)
        plt.grid(True, alpha=0.3)

        # Add polynomial fit
        try:
            # Polynomial fit
            from scipy.optimize import curve_fit

            def poly_func(x, a, b, c):
                return a * (x ** 2) + b * x + c

            x_data = np.array(percentages)
            y_data = np.array(runtimes)

            # Try polynomial fit
            try:
                poly_params, _ = curve_fit(poly_func, x_data, y_data)
                x_fit = np.linspace(min(x_data), max(x_data), 100)
                y_fit = poly_func(x_fit, *poly_params)
                plt.plot(x_fit, y_fit, 'g--',
                         label=f'Quadratic: a*x²+b*x+c\na={poly_params[0]:.4f}, b={poly_params[1]:.4f}, c={poly_params[2]:.2f}')
            except:
                print("Warning: Could not fit polynomial curve to runtime data")

            plt.legend(fontsize=12)

        except ImportError:
            print("Warning: scipy not available. Skipping curve fitting.")

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'runtime_scaling.png'), dpi=300)
        plt.close()

        print(f"Created runtime scaling plot in {self.output_dir}")

    def plot_elbow_analysis(self):
        """
        Create elbow plots to find the optimal minimum settlement percentage.
        Balances efficiency gains against computational costs.
        """
        # Prepare the data
        percentages = self.config_stats.index.tolist()
        percentage_labels = [f"{pct * 100:.1f}%" for pct in percentages]

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Value Efficiency vs Percentage
        ax = axes[0, 0]
        y = self.config_stats['value_efficiency_mean']
        ax.plot(percentages, y, 'o-', linewidth=2, markersize=10, color='green')

        # Add data labels
        for i, pct in enumerate(percentages):
            ax.annotate(f"{y.iloc[i]:.2f}%",
                        (pct, y.iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10)

        ax.set_xlabel('Minimum Settlement Percentage')
        ax.set_xticks(percentages)
        ax.set_xticklabels(percentage_labels)
        ax.set_ylabel('Value Efficiency (%)')
        ax.set_title('Value Efficiency vs Percentage')
        ax.grid(True, alpha=0.3)

        # 2. Runtime vs Percentage
        ax = axes[0, 1]
        y = self.config_stats['runtime_seconds_mean']
        ax.plot(percentages, y, 'o-', linewidth=2, markersize=10, color='red')

        # Add data labels
        for i, pct in enumerate(percentages):
            ax.annotate(f"{y.iloc[i]:.2f}s",
                        (pct, y.iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10)

        ax.set_xlabel('Minimum Settlement Percentage')
        ax.set_xticks(percentages)
        ax.set_xticklabels(percentage_labels)
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Runtime vs Percentage')
        ax.grid(True, alpha=0.3)

        # 3. Efficiency per Second vs Percentage
        ax = axes[1, 0]

        # Calculate efficiency per second
        y = self.config_stats['value_efficiency_mean'] / self.config_stats['runtime_seconds_mean']
        ax.plot(percentages, y, 'o-', linewidth=2, markersize=10, color='purple')

        # Add data labels
        for i, pct in enumerate(percentages):
            ax.annotate(f"{y.iloc[i]:.4f}",
                        (pct, y.iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10)

        ax.set_xlabel('Minimum Settlement Percentage')
        ax.set_xticks(percentages)
        ax.set_xticklabels(percentage_labels)
        ax.set_ylabel('Value Efficiency per Second')
        ax.set_title('Efficiency/Runtime Ratio vs Percentage')
        ax.grid(True, alpha=0.3)

        # Identify the "elbow point" (maximum)
        best_pct = percentages[y.argmax()]
        ax.axvline(x=best_pct, color='k', linestyle='--', alpha=0.5,
                   label=f'Optimal Percentage: {best_pct * 100:.1f}%')
        ax.legend()

        # 4. Normalized metrics to show tradeoffs
        ax = axes[1, 1]

        # Normalize metrics to 0-1 scale for comparison
        runtime_norm = (self.config_stats['runtime_seconds_mean'] - self.config_stats['runtime_seconds_mean'].min()) / \
                       (self.config_stats['runtime_seconds_mean'].max() - self.config_stats[
                           'runtime_seconds_mean'].min()) if len(
            self.config_stats['runtime_seconds_mean'].unique()) > 1 else self.config_stats['runtime_seconds_mean'] * 0

        efficiency_norm = (self.config_stats['value_efficiency_mean'] - self.config_stats[
            'value_efficiency_mean'].min()) / \
                          (self.config_stats['value_efficiency_mean'].max() - self.config_stats[
                              'value_efficiency_mean'].min()) if len(
            self.config_stats['value_efficiency_mean'].unique()) > 1 else self.config_stats['value_efficiency_mean'] * 0

        # Plot normalized metrics
        ax.plot(percentages, runtime_norm, 'o-', linewidth=2, markersize=8, color='red', label='Runtime (normalized)')
        ax.plot(percentages, efficiency_norm, 'o-', linewidth=2, markersize=8, color='green',
                label='Efficiency (normalized)')

        # Calculate and plot the difference (efficiency gain - runtime cost)
        tradeoff = efficiency_norm - runtime_norm
        ax.plot(percentages, tradeoff, 'o-', linewidth=2, markersize=8, color='blue', label='Efficiency - Runtime')

        # Identify optimal percentage based on tradeoff
        optimal_pct = percentages[tradeoff.argmax()]
        ax.axvline(x=optimal_pct, color='k', linestyle='--', alpha=0.5,
                   label=f'Optimal Percentage: {optimal_pct * 100:.1f}%')

        ax.set_xlabel('Minimum Settlement Percentage')
        ax.set_xticks(percentages)
        ax.set_xticklabels(percentage_labels)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Efficiency vs Runtime Tradeoff')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add a note with recommendations
        plt.figtext(0.5, 0.01,
                    f"Recommended min_settlement_percentage based on efficiency/runtime ratio: {best_pct * 100:.1f}%\n"
                    f"Recommended min_settlement_percentage based on efficiency-runtime tradeoff: {optimal_pct * 100:.1f}%",
                    ha="center", fontsize=14,
                    bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle('Elbow Analysis for Optimal Minimum Settlement Percentage', fontsize=18)

        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'elbow_analysis.png'), dpi=300)
        plt.close()

        print(f"Created elbow analysis plot in {self.output_dir}")

    def plot_3d_surface(self):
        """
        Create a 3D surface plot showing the relationship between
        minimum settlement percentage, runtime, and efficiency.
        """
        try:
            # Prepare data for 3D plotting
            percentages = self.config_stats.index.tolist()
            runtimes = self.config_stats['runtime_seconds_mean'].tolist()
            efficiencies = self.config_stats['value_efficiency_mean'].tolist()

            # Create figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Create scatter plot
            scatter = ax.scatter(percentages, runtimes, efficiencies,
                                 c=efficiencies, s=100, cmap='viridis')

            # Add color bar
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Value Efficiency (%)', fontsize=12)

            # Connect points with lines
            ax.plot(percentages, runtimes, efficiencies, '-', color='gray', alpha=0.5)

            # Add text labels for each point
            for i, (p, r, e) in enumerate(zip(percentages, runtimes, efficiencies)):
                ax.text(p, r, e, f'Min %: {p * 100:.1f}%\nEff: {e:.2f}%', fontsize=10)

            # Add a surface fit (if more than 3 data points)
            if len(percentages) > 3:
                try:
                    # Create a mesh grid
                    pct_range = np.linspace(min(percentages), max(percentages), 20)
                    runtime_range = np.linspace(min(runtimes), max(runtimes), 20)
                    X, Y = np.meshgrid(pct_range, runtime_range)

                    # Fit a 2D polynomial
                    from scipy.interpolate import griddata
                    Z = griddata((percentages, runtimes), efficiencies, (X, Y), method='cubic')

                    # Plot the surface
                    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6,
                                           linewidth=0, antialiased=True)
                except Exception as e:
                    print(f"Warning: Could not create surface fit: {e}")

            # Set labels and title
            ax.set_xlabel('Minimum Settlement Percentage', fontsize=12)
            ax.set_ylabel('Runtime (seconds)', fontsize=12)
            ax.set_zlabel('Value Efficiency (%)', fontsize=12)
            ax.set_title('3D Relationship: Percentage, Runtime, and Efficiency', fontsize=14)

            # Format x-axis tick labels
            from matplotlib.ticker import FuncFormatter
            def pct_fmt(x, pos):
                return f'{x * 100:.1f}%'

            ax.xaxis.set_major_formatter(FuncFormatter(pct_fmt))

            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, '3d_surface_plot.png'), dpi=300)
            plt.close()

            print(f"Created 3D surface plot in {self.output_dir}")
        except Exception as e:
            print(f"Warning: Could not create 3D surface plot: {e}")
            print("This might be due to missing dependencies or too few data points.")

    def plot_comparative_bar_chart(self):
        """
        Create a comparative bar chart showing how different metrics
        vary across minimum settlement percentage configurations.
        """
        if 'instruction_efficiency_mean' not in self.config_stats.columns or \
                'value_efficiency_mean' not in self.config_stats.columns or \
                'partial_settlements_mean' not in self.config_stats.columns:
            print("Warning: Required metrics not available. Skipping this plot.")
            return

        # Prepare the data for plotting
        percentages = self.config_stats.index.tolist()
        percentage_labels = [f"{pct * 100:.1f}%" for pct in percentages]

        # Metrics to include in the comparison
        metrics = [
            ('instruction_efficiency_mean', 'Instruction Efficiency (%)', 'blue'),
            ('value_efficiency_mean', 'Value Efficiency (%)', 'green'),
            ('partial_settlements_mean', 'Partial Settlements (count)', 'orange')
        ]

        # Normalize metrics for fair comparison
        normalized_data = {}
        for metric_name, label, color in metrics:
            values = self.config_stats[metric_name]
            if values.max() - values.min() > 0:  # Avoid division by zero
                norm_values = (values - values.min()) / (values.max() - values.min())
            else:
                norm_values = values * 0  # If all values are identical, set to zero
            normalized_data[metric_name] = norm_values

        # Create the figure
        plt.figure(figsize=(14, 8))

        # Set up the plot
        bar_width = 0.2
        index = np.arange(len(percentages))

        # Plot each metric as a grouped bar
        for i, (metric_name, label, color) in enumerate(metrics):
            offset = (i - 1) * bar_width
            plt.bar(index + offset, normalized_data[metric_name],
                    bar_width, label=label, color=color, alpha=0.7)

        # Customize plot
        plt.xlabel('Minimum Settlement Percentage', fontsize=14)
        plt.ylabel('Normalized Value (0-1 scale)', fontsize=14)
        plt.title('Comparative Analysis of Key Metrics Across Percentage Configurations', fontsize=16)
        plt.xticks(index, percentage_labels)
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparative_metrics.png'), dpi=300)
        plt.close()

        print(f"Created comparative bar chart in {self.output_dir}")

    def plot_efficiency_distribution(self):
        """
        Create a violin plot showing the distribution of efficiency values
        across different minimum settlement percentage configurations.
        """
        # Check if we have raw data with multiple runs per configuration
        if not hasattr(self, 'df'):
            print("Warning: Raw data not available. Skipping efficiency distribution plot.")
            return

        # Create the figure
        plt.figure(figsize=(14, 8))

        # Prepare data in the format needed for violin plot
        df_plot = self.df.copy()
        # Format the percentage for better display
        df_plot['min_pct_label'] = df_plot['min_settlement_percentage'].apply(lambda x: f"{x * 100:.1f}%")

        # Create violin plot for instruction efficiency
        ax1 = plt.subplot(1, 2, 1)
        sns.violinplot(x='min_pct_label', y='instruction_efficiency', data=df_plot,
                       palette="Blues", inner="points", ax=ax1)
        plt.title('Instruction Efficiency Distribution', fontsize=14)
        plt.xlabel('Minimum Settlement Percentage', fontsize=12)
        plt.ylabel('Instruction Efficiency (%)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        # Create violin plot for value efficiency
        ax2 = plt.subplot(1, 2, 2)
        sns.violinplot(x='min_pct_label', y='value_efficiency', data=df_plot,
                       palette="Greens", inner="points", ax=ax2)
        plt.title('Value Efficiency Distribution', fontsize=14)
        plt.xlabel('Minimum Settlement Percentage', fontsize=12)
        plt.ylabel('Value Efficiency (%)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        # Add overall title
        plt.suptitle('Efficiency Distributions by Minimum Settlement Percentage', fontsize=16)

        # Save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(os.path.join(self.output_dir, 'efficiency_distribution.png'), dpi=300)
        plt.close()

        print(f"Created efficiency distribution plot in {self.output_dir}")

    def generate_all_visualizations(self):
        """Generate all visualization plots."""
        # Base visualizations from initial request
        print("Generating runtime vs efficiency plots...")
        self.plot_runtime_vs_efficiency()

        # Additional visualizations
        print("Generating efficiency vs percentage plot...")
        self.plot_efficiency_vs_percentage()

        print("Generating settled amount vs percentage plot...")
        self.plot_settled_amount_vs_percentage()

        print("Generating partial settlements vs percentage plot...")
        self.plot_partial_settlements_vs_percentage()

        print("Generating memory usage vs percentage plot...")
        self.plot_memory_usage_vs_percentage()

        print("Generating runtime scaling plot...")
        self.plot_runtime_scaling()

        print("Generating elbow analysis plot...")
        self.plot_elbow_analysis()

        print("Generating 3D surface plot...")
        self.plot_3d_surface()

        print("Generating comparative bar chart...")
        self.plot_comparative_bar_chart()

        print("Generating efficiency distribution plot...")
        self.plot_efficiency_distribution()

        print("\nAll visualizations have been generated successfully.")
        print(f"Results are available in: {os.path.abspath(self.output_dir)}")


if __name__ == "__main__":
    # Create visualizer and generate all plots
    visualizer = MinSettlementAmountVisualizer()
    visualizer.generate_all_visualizations()