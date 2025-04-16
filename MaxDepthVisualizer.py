import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


class MaxDepthVisualizer:
    """
    A class to visualize the results of the max child depth analysis simulations.
    Focuses on comparing different max child depth configurations averaged across runs.
    """

    def __init__(self, results_csv = r"C:\Users\matth\Documents\GitHub\SettlementSimulatorV3\max_depth_stats\max_child_depth_final_results.csv"
, output_dir="max_depth_visualizations"):
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
        """Load the results data and calculate averages per configuration."""
        try:
            # Load the CSV file
            df = pd.read_csv(self.results_file)
            print(f"Loaded data from {self.results_file}: {len(df)} rows")

            # Check required columns
            required_columns = ['max_child_depth', 'instruction_efficiency',
                                'value_efficiency', 'runtime_seconds']

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                print(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Missing required columns in the data file")

            # Group by max_child_depth and calculate statistics
            self.config_stats = df.groupby('max_child_depth').agg({
                'instruction_efficiency': ['mean', 'std', 'count'],
                'value_efficiency': ['mean', 'std', 'count'],
                'runtime_seconds': ['mean', 'std', 'count'],
                'settled_count': ['mean', 'std'] if 'settled_count' in df.columns else None,
                'settled_amount': ['mean', 'std'] if 'settled_amount' in df.columns else None,
                'avg_tree_depth': ['mean', 'std'] if 'avg_tree_depth' in df.columns else None,
                'partial_settlements': ['mean', 'std'] if 'partial_settlements' in df.columns else None,
                'memory_usage_mb': ['mean', 'std'] if 'memory_usage_mb' in df.columns else None
            }).dropna(axis=1, how='all')

            # Flatten MultiIndex columns
            self.config_stats.columns = ['_'.join(col).strip() for col in self.config_stats.columns.values]

            # Save the raw data for other analyses
            self.df = df

            print(f"Calculated statistics for {len(self.config_stats)} configurations")

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
        depths = self.config_stats.index.tolist()

        # Create figure for instruction efficiency
        plt.figure(figsize=(12, 8))

        # Calculate error bars (standard deviation)
        y = self.config_stats['instruction_efficiency_mean']
        yerr = self.config_stats['instruction_efficiency_std']

        # Create scatter plot with error bars
        plt.errorbar(x, y, yerr=yerr, fmt='o-', linewidth=2, markersize=10,
                     capsize=5, color='blue', label='Instruction Efficiency')

        # Add depth labels to each point
        for i, depth in enumerate(depths):
            plt.annotate(f"Depth: {depth}",
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

        # Add depth labels to each point
        for i, depth in enumerate(depths):
            plt.annotate(f"Depth: {depth}",
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

    def plot_efficiency_vs_depth(self):
        """
        Plot instruction and value efficiency against maximum child depth.
        Shows if there's a point of diminishing returns.
        """
        # Prepare the data for plotting
        depths = self.config_stats.index.tolist()

        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # First axis for instruction efficiency
        instruction_eff = self.config_stats['instruction_efficiency_mean']
        instruction_err = self.config_stats['instruction_efficiency_std']

        # Plot instruction efficiency
        color = 'blue'
        ax1.errorbar(depths, instruction_eff, yerr=instruction_err, fmt='o-',
                     linewidth=2, markersize=10, capsize=5, color=color,
                     label='Instruction Efficiency')
        ax1.set_xlabel('Maximum Child Depth', fontsize=14)
        ax1.set_ylabel('Instruction Efficiency (%)', color=color, fontsize=14)
        ax1.tick_params(axis='y', labelcolor=color)

        # Second y-axis for value efficiency
        ax2 = ax1.twinx()
        value_eff = self.config_stats['value_efficiency_mean']
        value_err = self.config_stats['value_efficiency_std']

        # Plot value efficiency
        color = 'green'
        ax2.errorbar(depths, value_eff, yerr=value_err, fmt='s--',
                     linewidth=2, markersize=10, capsize=5, color=color,
                     label='Value Efficiency')
        ax2.set_ylabel('Value Efficiency (%)', color=color, fontsize=14)
        ax2.tick_params(axis='y', labelcolor=color)

        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)

        # Add title
        plt.title('Efficiency Metrics vs Maximum Child Depth', fontsize=16)
        plt.grid(True, alpha=0.3)

        # Add data point labels
        for i, depth in enumerate(depths):
            # Label instruction efficiency points
            ax1.annotate(f"{instruction_eff.iloc[i]:.2f}%",
                         (depth, instruction_eff.iloc[i]),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', fontsize=10, color='blue')

            # Label value efficiency points
            ax2.annotate(f"{value_eff.iloc[i]:.2f}%",
                         (depth, value_eff.iloc[i]),
                         xytext=(0, -15), textcoords='offset points',
                         ha='center', fontsize=10, color='green')

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'efficiency_vs_depth.png'), dpi=300)
        plt.close()

        print(f"Created efficiency vs depth plot in {self.output_dir}")

    def plot_settled_amount_vs_depth(self):
        """
        Plot the total settled amount against maximum child depth.
        Shows how settlement magnitude changes with different depth limits.
        """
        if 'settled_amount_mean' not in self.config_stats.columns:
            print("Warning: settled_amount data not available. Skipping this plot.")
            return

        # Prepare the data for plotting
        depths = self.config_stats.index.tolist()
        values = self.config_stats['settled_amount_mean']
        errors = self.config_stats['settled_amount_std']

        # Create the figure
        plt.figure(figsize=(12, 8))

        # Create bar chart with error bars
        bars = plt.bar(depths, values, yerr=errors, capsize=5,
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
        plt.xlabel('Maximum Child Depth', fontsize=14)
        plt.ylabel('Average Settled Amount', fontsize=14)
        plt.title('Settlement Amount vs Maximum Child Depth', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(depths)

        # Add a line showing the trend
        plt.plot(depths, values, 'ro-', linewidth=2, alpha=0.7)

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
        plt.savefig(os.path.join(self.output_dir, 'settled_amount_vs_depth.png'), dpi=300)
        plt.close()

        print(f"Created settled amount vs depth plot in {self.output_dir}")

    def plot_partial_settlements_vs_depth(self):
        """
        Plot the number of partial settlements against maximum child depth.
        Shows how the frequency of partial settlements changes with different depth limits.
        """
        if 'partial_settlements_mean' not in self.config_stats.columns:
            print("Warning: partial_settlements data not available. Skipping this plot.")
            return

        # Prepare the data for plotting
        depths = self.config_stats.index.tolist()
        values = self.config_stats['partial_settlements_mean']
        errors = self.config_stats['partial_settlements_std']

        # Create the figure
        plt.figure(figsize=(12, 8))

        # Create bar chart with error bars
        bars = plt.bar(depths, values, yerr=errors, capsize=5,
                       color='lightgreen', edgecolor='darkgreen', alpha=0.7)

        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.0f}',
                     ha='center', va='bottom', fontsize=12)

        # Customize plot
        plt.xlabel('Maximum Child Depth', fontsize=14)
        plt.ylabel('Average Number of Partial Settlements', fontsize=14)
        plt.title('Partial Settlements vs Maximum Child Depth', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(depths)

        # Add a line showing the trend
        plt.plot(depths, values, 'ro-', linewidth=2, alpha=0.7)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'partial_settlements_vs_depth.png'), dpi=300)
        plt.close()

        print(f"Created partial settlements vs depth plot in {self.output_dir}")

    def plot_avg_tree_depth_vs_max_depth(self):
        """
        Plot the average tree depth against maximum configured child depth.
        Shows if the actual tree depth approaches the maximum limit.
        """
        if 'avg_tree_depth_mean' not in self.config_stats.columns:
            print("Warning: avg_tree_depth data not available. Skipping this plot.")
            return

        # Prepare the data for plotting
        depths = self.config_stats.index.tolist()
        values = self.config_stats['avg_tree_depth_mean']
        errors = self.config_stats['avg_tree_depth_std']

        # Create the figure
        plt.figure(figsize=(12, 8))

        # Create scatter plot with error bars
        plt.errorbar(depths, values, yerr=errors, fmt='o-',
                     linewidth=2, markersize=10, capsize=5,
                     color='purple', label='Average Tree Depth')

        # Add reference line (y = x)
        max_depth = max(depths)
        plt.plot([0, max_depth], [0, max_depth], 'k--',
                 alpha=0.7, label='Maximum Possible (y = x)')

        # Add data labels
        for i, depth in enumerate(depths):
            plt.annotate(f"{values.iloc[i]:.2f}",
                         (depth, values.iloc[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=12)

        # Customize plot
        plt.xlabel('Maximum Child Depth Configuration', fontsize=14)
        plt.ylabel('Average Tree Depth Achieved', fontsize=14)
        plt.title('Average Tree Depth vs Maximum Configured Depth', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Set axis limits
        plt.xlim(-0.5, max_depth + 0.5)
        plt.ylim(-0.5, max(max_depth + 0.5, max(values) * 1.1))

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'avg_tree_depth_vs_max_depth.png'), dpi=300)
        plt.close()

        print(f"Created average tree depth vs max depth plot in {self.output_dir}")

    def plot_memory_usage_vs_depth(self):
        """
        Plot memory usage against maximum child depth.
        Shows how computational resources scale with depth configuration.
        """
        if 'memory_usage_mb_mean' not in self.config_stats.columns:
            print("Warning: memory_usage_mb data not available. Skipping this plot.")
            return

        # Prepare the data for plotting
        depths = self.config_stats.index.tolist()
        values = self.config_stats['memory_usage_mb_mean']
        errors = self.config_stats['memory_usage_mb_std']

        # Create the figure
        plt.figure(figsize=(12, 8))

        # Create scatter plot with error bars
        plt.errorbar(depths, values, yerr=errors, fmt='o-',
                     linewidth=2, markersize=10, capsize=5,
                     color='orange', label='Memory Usage')

        # Add data labels
        for i, depth in enumerate(depths):
            plt.annotate(f"{values.iloc[i]:.1f} MB",
                         (depth, values.iloc[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=12)

        # Customize plot
        plt.xlabel('Maximum Child Depth', fontsize=14)
        plt.ylabel('Average Memory Usage (MB)', fontsize=14)
        plt.title('Memory Usage vs Maximum Child Depth', fontsize=16)
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'memory_usage_vs_depth.png'), dpi=300)
        plt.close()

        print(f"Created memory usage vs depth plot in {self.output_dir}")

    def plot_runtime_scaling(self):
        """
        Plot the relationship between max child depth and runtime on a log scale.
        Helps identify potential exponential growth in computation time.
        """
        # Prepare the data for plotting
        depths = self.config_stats.index.tolist()
        runtimes = self.config_stats['runtime_seconds_mean']

        # Create the figure
        plt.figure(figsize=(12, 8))

        # Create log scale scatter plot
        plt.plot(depths, runtimes, 'o-', linewidth=2, markersize=10, color='red')

        # Add data labels
        for i, depth in enumerate(depths):
            plt.annotate(f"{runtimes.iloc[i]:.2f}s",
                         (depth, runtimes.iloc[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=12)

        # Set log scale for y-axis
        plt.yscale('log')

        # Customize plot
        plt.xlabel('Maximum Child Depth', fontsize=14)
        plt.ylabel('Runtime (seconds, log scale)', fontsize=14)
        plt.title('Runtime Scaling (Log Scale) vs Maximum Child Depth', fontsize=16)
        plt.grid(True, which="both", alpha=0.3)

        # Add exponential and polynomial fit lines
        try:
            # Exponential fit
            from scipy.optimize import curve_fit

            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c

            def power_func(x, a, b, c):
                return a * (x ** b) + c

            x_data = np.array(depths)
            y_data = np.array(runtimes)

            # Try exponential fit
            try:
                exp_params, _ = curve_fit(exp_func, x_data, y_data)
                x_fit = np.linspace(min(x_data), max(x_data), 100)
                y_fit = exp_func(x_fit, *exp_params)
                plt.plot(x_fit, y_fit, 'b--',
                         label=f'Exponential: a*e^(b*x)+c\na={exp_params[0]:.2e}, b={exp_params[1]:.4f}, c={exp_params[2]:.2f}')
            except:
                print("Warning: Could not fit exponential curve to runtime data")

            # Try power law fit
            try:
                power_params, _ = curve_fit(power_func, x_data, y_data)
                y_fit = power_func(x_fit, *power_params)
                plt.plot(x_fit, y_fit, 'g--',
                         label=f'Power: a*x^b+c\na={power_params[0]:.2f}, b={power_params[1]:.2f}, c={power_params[2]:.2f}')
            except:
                print("Warning: Could not fit power curve to runtime data")

            plt.legend(fontsize=12)

        except ImportError:
            print("Warning: scipy not available. Skipping curve fitting.")

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'runtime_scaling_log.png'), dpi=300)

        # Create linear scale version as well
        plt.yscale('linear')
        plt.title('Runtime Scaling (Linear Scale) vs Maximum Child Depth', fontsize=16)
        plt.ylabel('Runtime (seconds)', fontsize=14)
        plt.savefig(os.path.join(self.output_dir, 'runtime_scaling_linear.png'), dpi=300)
        plt.close()

        print(f"Created runtime scaling plots in {self.output_dir}")

    def plot_elbow_analysis(self):
        """
        Create elbow plots to find the optimal maximum child depth.
        Balances efficiency gains against computational costs.
        """
        # Prepare the data
        depths = self.config_stats.index.tolist()

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Value Efficiency vs Depth
        ax = axes[0, 0]
        y = self.config_stats['value_efficiency_mean']
        ax.plot(depths, y, 'o-', linewidth=2, markersize=10, color='green')

        # Add data labels
        for i, depth in enumerate(depths):
            ax.annotate(f"{y.iloc[i]:.2f}%",
                        (depth, y.iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10)

        ax.set_xlabel('Maximum Child Depth')
        ax.set_ylabel('Value Efficiency (%)')
        ax.set_title('Value Efficiency vs Depth')
        ax.grid(True, alpha=0.3)

        # 2. Runtime vs Depth
        ax = axes[0, 1]
        y = self.config_stats['runtime_seconds_mean']
        ax.plot(depths, y, 'o-', linewidth=2, markersize=10, color='red')

        # Add data labels
        for i, depth in enumerate(depths):
            ax.annotate(f"{y.iloc[i]:.2f}s",
                        (depth, y.iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10)

        ax.set_xlabel('Maximum Child Depth')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Runtime vs Depth')
        ax.grid(True, alpha=0.3)

        # 3. Efficiency per Second vs Depth
        ax = axes[1, 0]

        # Calculate efficiency per second
        y = self.config_stats['value_efficiency_mean'] / self.config_stats['runtime_seconds_mean']
        ax.plot(depths, y, 'o-', linewidth=2, markersize=10, color='purple')

        # Add data labels
        for i, depth in enumerate(depths):
            ax.annotate(f"{y.iloc[i]:.4f}",
                        (depth, y.iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10)

        ax.set_xlabel('Maximum Child Depth')
        ax.set_ylabel('Value Efficiency per Second')
        ax.set_title('Efficiency/Runtime Ratio vs Depth')
        ax.grid(True, alpha=0.3)

        # Identify the "elbow point" (maximum)
        best_depth = depths[y.argmax()]
        ax.axvline(x=best_depth, color='k', linestyle='--', alpha=0.5,
                   label=f'Optimal Depth: {best_depth}')
        ax.legend()

        # 4. Normalized metrics to show tradeoffs
        ax = axes[1, 1]

        # Normalize metrics to 0-1 scale for comparison
        runtime_norm = (self.config_stats['runtime_seconds_mean'] - self.config_stats['runtime_seconds_mean'].min()) / \
                       (self.config_stats['runtime_seconds_mean'].max() - self.config_stats[
                           'runtime_seconds_mean'].min())

        efficiency_norm = (self.config_stats['value_efficiency_mean'] - self.config_stats[
            'value_efficiency_mean'].min()) / \
                          (self.config_stats['value_efficiency_mean'].max() - self.config_stats[
                              'value_efficiency_mean'].min())

        # Plot normalized metrics
        ax.plot(depths, runtime_norm, 'o-', linewidth=2, markersize=8, color='red', label='Runtime (normalized)')
        ax.plot(depths, efficiency_norm, 'o-', linewidth=2, markersize=8, color='green',
                label='Efficiency (normalized)')

        # Calculate and plot the difference (efficiency gain - runtime cost)
        tradeoff = efficiency_norm - runtime_norm
        ax.plot(depths, tradeoff, 'o-', linewidth=2, markersize=8, color='blue', label='Efficiency - Runtime')

        # Identify optimal depth based on tradeoff
        optimal_depth = depths[tradeoff.argmax()]
        ax.axvline(x=optimal_depth, color='k', linestyle='--', alpha=0.5,
                   label=f'Optimal Depth: {optimal_depth}')

        ax.set_xlabel('Maximum Child Depth')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Efficiency vs Runtime Tradeoff')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add a note with recommendations
        plt.figtext(0.5, 0.01,
                    f"Recommended max_child_depth based on efficiency/runtime ratio: {best_depth}\n"
                    f"Recommended max_child_depth based on efficiency-runtime tradeoff: {optimal_depth}",
                    ha="center", fontsize=14,
                    bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle('Elbow Analysis for Optimal Maximum Child Depth', fontsize=18)

        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'elbow_analysis.png'), dpi=300)
        plt.close()

        print(f"Created elbow analysis plot in {self.output_dir}")

    def plot_3d_surface(self):
        """
        Create a 3D surface plot showing the relationship between
        max child depth, runtime, and efficiency.
        """
        try:
            # Prepare data for 3D plotting
            depths = self.config_stats.index.tolist()
            runtimes = self.config_stats['runtime_seconds_mean'].tolist()
            efficiencies = self.config_stats['value_efficiency_mean'].tolist()

            # Create figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Create scatter plot
            scatter = ax.scatter(depths, runtimes, efficiencies,
                                 c=efficiencies, s=100, cmap='viridis')

            # Add color bar
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Value Efficiency (%)', fontsize=12)

            # Connect points with lines
            ax.plot(depths, runtimes, efficiencies, '-', color='gray', alpha=0.5)

            # Add text labels for each point
            for i, (d, r, e) in enumerate(zip(depths, runtimes, efficiencies)):
                ax.text(d, r, e, f'Depth: {d}\nEff: {e:.2f}%', fontsize=10)

            # Add a surface fit (if more than 3 data points)
            if len(depths) > 3:
                try:
                    # Create a mesh grid
                    depth_range = np.linspace(min(depths), max(depths), 20)
                    runtime_range = np.linspace(min(runtimes), max(runtimes), 20)
                    X, Y = np.meshgrid(depth_range, runtime_range)

                    # Fit a 2D polynomial
                    from scipy.interpolate import griddata
                    Z = griddata((depths, runtimes), efficiencies, (X, Y), method='cubic')

                    # Plot the surface
                    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6,
                                           linewidth=0, antialiased=True)
                except Exception as e:
                    print(f"Warning: Could not create surface fit: {e}")

            # Set labels and title
            ax.set_xlabel('Maximum Child Depth', fontsize=12)
            ax.set_ylabel('Runtime (seconds)', fontsize=12)
            ax.set_zlabel('Value Efficiency (%)', fontsize=12)
            ax.set_title('3D Relationship: Depth, Runtime, and Efficiency', fontsize=14)

            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, '3d_surface_plot.png'), dpi=300)
            plt.close()

            print(f"Created 3D surface plot in {self.output_dir}")
        except Exception as e:
            print(f"Warning: Could not create 3D surface plot: {e}")
            print("This might be due to missing dependencies or too few data points.")

    def generate_all_visualizations(self):
        """Generate all visualization plots."""
        # Base visualizations from initial request
        print("Generating runtime vs efficiency plots...")
        self.plot_runtime_vs_efficiency()

        # Additional visualizations
        print("Generating efficiency vs depth plot...")
        self.plot_efficiency_vs_depth()

        print("Generating settled amount vs depth plot...")
        self.plot_settled_amount_vs_depth()

        print("Generating partial settlements vs depth plot...")
        self.plot_partial_settlements_vs_depth()

        print("Generating average tree depth vs max depth plot...")
        self.plot_avg_tree_depth_vs_max_depth()

        print("Generating memory usage vs depth plot...")
        self.plot_memory_usage_vs_depth()

        print("Generating runtime scaling plot...")
        self.plot_runtime_scaling()

        print("Generating elbow analysis plot...")
        self.plot_elbow_analysis()

        print("Generating 3D surface plot...")
        self.plot_3d_surface()

        print("\nAll visualizations have been generated successfully.")
        print(f"Results are available in: {os.path.abspath(self.output_dir)}")


if __name__ == "__main__":
    # Create visualizer and generate all plots
    visualizer = MaxDepthVisualizer()
    visualizer.generate_all_visualizations()