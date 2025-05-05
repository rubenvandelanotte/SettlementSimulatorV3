import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import warnings


class MaxDepthVisualizer:
    """
    Visualize max-child-depth analysis directly from per-run JSON stats.

    Improvements:
      - Added plots for settled on-time vs late amounts by depth.
      - Integrated bar charts for threshold vs settled on-time amounts.
      - Refactored common plotting logic into helpers (_plot_errorbar_series).
      - Externalized default figure size, DPI, and styling parameters.
      - Added CLI entry point for command-line usage.
      - Export summary DataFrame to CSV for further analysis.
    """

    DEFAULT_FIGSIZE = (10, 6)
    DEFAULT_DPI = 200

    def __init__(self, results_dir, output_dir="max_depth_visualizations"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._load_data()
        self._compute_statistics()
        summary_path = os.path.join(self.output_dir, "summary_stats.csv")
        self.config_stats.to_csv(summary_path)
        print(f"Summary stats exported to {summary_path}")

    def _load_data(self):
        records = []
        for fname in os.listdir(self.results_dir):
            if not fname.lower().endswith(".json"):
                continue
            data = json.load(open(os.path.join(self.results_dir, fname)))
            depth = data.get("max_child_depth")
            if depth is None:
                m = re.search(r"maxdepth(\d+)", fname)
                depth = int(m.group(1)) if m else None
            if depth is None:
                warnings.warn(f"Cannot find max_child_depth for {fname}, skipping.")
                continue
            settled_on_time = data.get("settled_on_time_amount", 0)
            settled_late = data.get("settled_late_amount", 0)
            count = data['mothers_effectively_settled']
            if count is None:
                warnings.warn(f"No settled count in {fname}, skipping.")
                continue
            records.append({
                "max_child_depth": int(depth),
                "instruction_efficiency": data.get("instruction_efficiency"),
                "value_efficiency": data.get("value_efficiency"),
                "normal_instruction_efficiency": data.get("normal_instruction_efficiency"),
                "normal_value_efficiency": data.get("normal_value_efficiency"),
                "runtime_seconds": data.get("execution_time_seconds"),
                "memory_usage_mb": data.get("memory_usage_mb"),
                "settled_count": count,
                "settled_amount": settled_on_time + settled_late,
                "settled_on_time_amount": settled_on_time,
                "settled_late_amount": settled_late,
                "avg_tree_depth": data.get("avg_tree_depth"),
                "partial_settlements": data.get("partial_settlements"),
            })
        self.df = pd.DataFrame(records)
        if self.df.empty:
            raise RuntimeError(f"No valid JSON stats found in {self.results_dir}")
        print(f"Loaded {len(self.df)} runs.")

    def _compute_statistics(self):
        grp = self.df.groupby("max_child_depth")
        stats = {}
        for col in self.df.columns:
            if col == "max_child_depth":
                continue
            stats[f"{col}_mean"] = grp[col].mean()
            stats[f"{col}_std"] = grp[col].std()
            stats[f"{col}_count"] = grp[col].count()
        self.config_stats = pd.DataFrame(stats)
        print("Computed statistics for each depth.")

    def _plot_errorbar_series(self, x, y, yerr=None, xlabel='', ylabel='', title='',
                              fmt='o-', color='C0', label=None, annotate_fmt=None,
                              filename='plot.png'):
        plt.figure(figsize=self.DEFAULT_FIGSIZE)
        plt.errorbar(x, y, yerr=yerr, fmt=fmt, color=color, label=label,
                     capsize=5, linewidth=1.5, markersize=6)
        if annotate_fmt:
            for xi, yi in zip(x, y):
                plt.annotate(annotate_fmt.format(yi), (xi, yi),
                             textcoords='offset points', xytext=(5,5), fontsize=9)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        if label:
            plt.legend()
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=self.DEFAULT_DPI)
        plt.close()
        print(f"Saved plot: {filename}")

    def plot_settled_on_time_vs_depth(self):
        """
        Plot settled-on-time amount vs max_child_depth both as errorbar series and bar chart.
        """
        depths = self.config_stats.index.values
        means = self.config_stats['settled_on_time_amount_mean']
        stds = self.config_stats['settled_on_time_amount_std']
        # Errorbar series
        self._plot_errorbar_series(
            x=depths, y=means, yerr=stds,
            xlabel='Max Child Depth', ylabel='Settled On-time Amount',
            title='Settled On-time Amount vs Depth', fmt='s-', color='tab:cyan',
            annotate_fmt='{:.0f}', filename='settled_on_time_series.png'
        )
        # Bar chart
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        bars = ax.bar(depths, means, yerr=stds, capsize=5,
                      color='tab:cyan', edgecolor='navy', alpha=0.7)
        ax.set_xlabel('Max Child Depth')
        ax.set_ylabel('Settled On-time Amount')
        ax.set_title('Average Settled On-time Amount by Depth (Bar Chart)')
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1e6:.1f}M' if v>=1e6 else f'{v:.0f}'))
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:,.0f}',
                    ha='center', va='bottom', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        filename = 'settled_on_time_bar.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.DEFAULT_DPI)
        plt.close(fig)
        print(f"Saved plot: {filename}")

    def plot_runtime_vs_efficiency(self):
        x = self.config_stats['runtime_seconds_mean']
        depths = self.config_stats.index.tolist()

        # Instruction efficiency
        plt.figure(figsize=(12, 8))
        y = self.config_stats['instruction_efficiency_mean']
        yerr = self.config_stats['instruction_efficiency_std']
        plt.errorbar(x, y, yerr=yerr, fmt='o-', linewidth=2, markersize=8,
                     capsize=5, color='blue', label='Instruction Efficiency')
        for i, d in enumerate(depths):
            plt.annotate(f"Depth: {d}", (x.iloc[i], y.iloc[i]),
                         xytext=(8, 4), textcoords='offset points', fontsize=10)
        z = np.polyfit(x, y, 1);
        p = np.poly1d(z)
        plt.plot(x, p(x), 'r--', alpha=0.7,
                 label=f"Trend: y={z[0]:.4f}x+{z[1]:.2f}")
        corr = np.corrcoef(x, y)[0, 1]
        plt.title(f'Runtime vs Instruction Efficiency\nCorrelation: {corr:.4f}', fontsize=16)
        plt.xlabel('Average Runtime (s)', fontsize=14)
        plt.ylabel('Instruction Efficiency (%)', fontsize=14)
        plt.grid(alpha=0.3);
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'runtime_vs_instruction_efficiency.png'), dpi=300)
        plt.close()

        # Value efficiency
        plt.figure(figsize=(12, 8))
        y = self.config_stats['value_efficiency_mean']
        yerr = self.config_stats['value_efficiency_std']
        plt.errorbar(x, y, yerr=yerr, fmt='o-', linewidth=2, markersize=8,
                     capsize=5, color='green', label='Value Efficiency')
        for i, d in enumerate(depths):
            plt.annotate(f"Depth: {d}", (x.iloc[i], y.iloc[i]),
                         xytext=(8, 4), textcoords='offset points', fontsize=10)
        z = np.polyfit(x, y, 1);
        p = np.poly1d(z)
        plt.plot(x, p(x), 'r--', alpha=0.7,
                 label=f"Trend: y={z[0]:.4f}x+{z[1]:.2f}")
        corr = np.corrcoef(x, y)[0, 1]
        plt.title(f'Runtime vs Value Efficiency\nCorrelation: {corr:.4f}', fontsize=16)
        plt.xlabel('Average Runtime (s)', fontsize=14)
        plt.ylabel('Value Efficiency (%)', fontsize=14)
        plt.grid(alpha=0.3);
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'runtime_vs_value_efficiency.png'), dpi=300)
        plt.close()

    def plot_efficiency_vs_depth(self):
        depths = self.config_stats.index.tolist()
        instr = self.config_stats['instruction_efficiency_mean']
        instr_err = self.config_stats['instruction_efficiency_std']
        val = self.config_stats['value_efficiency_mean']
        val_err = self.config_stats['value_efficiency_std']

        fig, ax1 = plt.subplots(figsize=(12, 8))
        color = 'blue'
        ax1.errorbar(depths, instr, yerr=instr_err, fmt='o-', capsize=5,
                     color=color, label='Instruction')
        ax1.set_xlabel('Max Child Depth', fontsize=14)
        ax1.set_ylabel('Instruction Efficiency (%)', color=color, fontsize=14)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'green'
        ax2.errorbar(depths, val, yerr=val_err, fmt='s--', capsize=5,
                     color=color, label='Value')
        ax2.set_ylabel('Value Efficiency (%)', color=color, fontsize=14)
        ax2.tick_params(axis='y', labelcolor=color)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)
        plt.title('Efficiency Metrics vs Max Child Depth', fontsize=16)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'efficiency_vs_depth.png'), dpi=300)
        plt.close()

    def plot_settled_amount_vs_depth(self):
        if 'settled_amount_mean' not in self.config_stats:
            return
        depths = self.config_stats.index.tolist()
        vals = self.config_stats['settled_amount_mean']
        errs = self.config_stats['settled_amount_std']
        plt.figure(figsize=(12, 8))
        bars = plt.bar(depths, vals, yerr=errs, capsize=5,
                       color='skyblue', edgecolor='navy', alpha=0.7)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h,
                     f"{h:,.0f}", ha='center', va='bottom', fontsize=12)
        plt.plot(depths, vals, 'ro-', alpha=0.7)
        plt.xlabel('Max Child Depth', fontsize=14)
        plt.ylabel('Average Settled Amount', fontsize=14)
        plt.title('Settlement Amount vs Max Child Depth', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        from matplotlib.ticker import FuncFormatter
        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda val, pos: f'{val / 1e6:.1f}M' if val >= 1e6 else f'{val:.0f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'settled_amount_vs_depth.png'), dpi=300)
        plt.close()

    def plot_partial_settlements_vs_depth(self):
        if 'partial_settlements_mean' not in self.config_stats:
            return
        depths = self.config_stats.index.tolist()
        vals = self.config_stats['partial_settlements_mean']
        errs = self.config_stats['partial_settlements_std']
        plt.figure(figsize=(12, 8))
        bars = plt.bar(depths, vals, yerr=errs, capsize=5,
                       color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=12)
        plt.plot(depths, vals, 'ro-', alpha=0.7)
        plt.xlabel('Max Child Depth', fontsize=14)
        plt.ylabel('Avg Partial Settlements', fontsize=14)
        plt.title('Partial Settlements vs Max Child Depth', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'partial_settlements_vs_depth.png'), dpi=300)
        plt.close()

    def plot_avg_tree_depth_vs_max_depth(self):
        if 'avg_tree_depth_mean' not in self.config_stats:
            return
        depths = self.config_stats.index.tolist()
        vals = self.config_stats['avg_tree_depth_mean']
        errs = self.config_stats['avg_tree_depth_std']
        plt.figure(figsize=(12, 8))
        plt.errorbar(depths, vals, yerr=errs, fmt='o-', linewidth=2, markersize=8,
                     capsize=5, color='purple', label='Avg Tree Depth')
        max_d = max(depths)
        plt.plot([0, max_d], [0, max_d], 'k--', alpha=0.7, label='y=x')
        for i, d in enumerate(depths):
            plt.annotate(f"{vals.iloc[i]:.2f}", (d, vals.iloc[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=10)
        plt.xlabel('Max Child Depth', fontsize=14)
        plt.ylabel('Avg Tree Depth Achieved', fontsize=14)
        plt.title('Avg Tree Depth vs Max Child Depth', fontsize=16)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        plt.xlim(-0.5, max_d + 0.5);
        plt.ylim(-0.5, max(max_d + 0.5, max(vals) * 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'avg_tree_depth_vs_max_depth.png'), dpi=300)
        plt.close()

    def plot_memory_usage_vs_depth(self):
        if 'memory_usage_mb_mean' not in self.config_stats:
            return
        depths = self.config_stats.index.tolist()
        vals = self.config_stats['memory_usage_mb_mean']
        errs = self.config_stats['memory_usage_mb_std']
        plt.figure(figsize=(12, 8))
        plt.errorbar(depths, vals, yerr=errs, fmt='o-', linewidth=2, markersize=8,
                     capsize=5, color='orange', label='Memory Usage')
        for i, d in enumerate(depths):
            plt.annotate(f"{vals.iloc[i]:.1f} MB", (d, vals.iloc[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=10)
        plt.xlabel('Max Child Depth', fontsize=14)
        plt.ylabel('Memory Usage (MB)', fontsize=14)
        plt.title('Memory Usage vs Max Child Depth', fontsize=16)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'memory_usage_vs_depth.png'), dpi=300)
        plt.close()

    def plot_runtime_scaling(self):
        depths = self.config_stats.index.tolist()
        runtimes = self.config_stats['runtime_seconds_mean']

        plt.figure(figsize=(12, 8))
        plt.plot(depths, runtimes, 'o-', linewidth=2, markersize=8, color='red')
        for i, d in enumerate(depths):
            plt.annotate(f"{runtimes.iloc[i]:.2f}s", (d, runtimes.iloc[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=10)
        plt.yscale('log')
        plt.xlabel('Max Child Depth', fontsize=14)
        plt.ylabel('Runtime (s, log scale)', fontsize=14)
        plt.title('Runtime Scaling (Log) vs Depth', fontsize=16)
        plt.grid(which='both', alpha=0.3)
        # Fits
        try:
            from scipy.optimize import curve_fit
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c

            def power_func(x, a, b, c):
                return a * (x ** b) + c

            x, y = np.array(depths), np.array(runtimes)
            x_fit = np.linspace(min(x), max(x), 100)
            try:
                params, _ = curve_fit(exp_func, x, y)
                plt.plot(x_fit, exp_func(x_fit, *params), 'b--',
                         label=f'exp: a={params[0]:.2e}, b={params[1]:.2f}, c={params[2]:.2f}')
            except:
                warnings.warn('Could not fit exponential')
            try:
                params, _ = curve_fit(power_func, x, y)
                plt.plot(x_fit, power_func(x_fit, *params), 'g--',
                         label=f'power: a={params[0]:.2f}, b={params[1]:.2f}, c={params[2]:.2f}')
            except:
                warnings.warn('Could not fit power')
            plt.legend(fontsize=12)
        except ImportError:
            warnings.warn('scipy unavailable, skipping fits')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'runtime_scaling_log.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(depths, runtimes, 'o-', linewidth=2, markersize=8, color='red')
        plt.xlabel('Max Child Depth', fontsize=14)
        plt.ylabel('Runtime (s)', fontsize=14)
        plt.title('Runtime Scaling (Linear) vs Depth', fontsize=16)
        plt.grid(alpha=0.3)
        # linear fit
        m, b = np.polyfit(depths, runtimes, 1)
        plt.plot(depths, m * np.array(depths) + b, 'r--',
                 label=f'y={m:.2f}x+{b:.2f}')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'runtime_scaling_linear.png'), dpi=300)
        plt.close()

    def plot_elbow_analysis(self):
        depths = self.config_stats.index.tolist()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # Value vs depth
        y = self.config_stats['value_efficiency_mean']
        ax = axes[0, 0]
        ax.plot(depths, y, 'o-', color='green', markersize=8, linewidth=2)
        for i, d in enumerate(depths): ax.annotate(f"{y.iloc[i]:.2f}%", (d, y.iloc[i]), xytext=(5, 5),
                                                   textcoords='offset points', fontsize=10)
        ax.set_title('Value Efficiency vs Depth');
        ax.grid(alpha=0.3)

        # Runtime vs depth
        y = self.config_stats['runtime_seconds_mean']
        ax = axes[0, 1]
        ax.plot(depths, y, 'o-', color='red', markersize=8, linewidth=2)
        for i, d in enumerate(depths): ax.annotate(f"{y.iloc[i]:.2f}s", (d, y.iloc[i]), xytext=(5, 5),
                                                   textcoords='offset points', fontsize=10)
        ax.set_title('Runtime vs Depth');
        ax.grid(alpha=0.3)

        # Efficiency per second
        ax = axes[1, 0]
        ratio = self.config_stats['value_efficiency_mean'] / self.config_stats['runtime_seconds_mean']
        ax.plot(depths, ratio, 'o-', color='purple', markersize=8, linewidth=2)
        for i, d in enumerate(depths): ax.annotate(f"{ratio.iloc[i]:.4f}", (d, ratio.iloc[i]), xytext=(5, 5),
                                                   textcoords='offset points', fontsize=10)
        opt = depths[np.argmax(ratio)]
        ax.axvline(opt, linestyle='--', color='black', alpha=0.5)
        ax.set_title('Efficiency/Runtime Ratio vs Depth');
        ax.grid(alpha=0.3)

        # Normalized tradeoff
        ax = axes[1, 1]
        rt = self.config_stats['runtime_seconds_mean'];
        ve = self.config_stats['value_efficiency_mean']
        rt_n = (rt - rt.min()) / (rt.max() - rt.min());
        ve_n = (ve - ve.min()) / (ve.max() - ve.min())
        trade = ve_n - rt_n
        ax.plot(depths, rt_n, 'o-', color='red', label='Runtime norm')
        ax.plot(depths, ve_n, 'o-', color='green', label='Efficiency norm')
        ax.plot(depths, trade, 'o-', color='blue', label='Efficiency - Runtime')
        opt2 = depths[np.argmax(trade)]
        ax.axvline(opt2, linestyle='--', color='black', alpha=0.5)
        ax.set_title('Efficiency vs Runtime Tradeoff');
        ax.grid(alpha=0.3);
        ax.legend(fontsize=12)

        plt.figtext(0.5, 0.01, f"Recommended depths: ratio={opt}, tradeoff={opt2}",
                    ha='center', fontsize=14, bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5})
        plt.suptitle('Elbow Analysis for Optimal Depth', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(self.output_dir, 'elbow_analysis.png'), dpi=300)
        plt.close()

    def plot_3d_surface(self):
        try:
            depths = self.config_stats.index.tolist()
            runtimes = self.config_stats['runtime_seconds_mean'].tolist()
            eff = self.config_stats['value_efficiency_mean'].tolist()
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(depths, runtimes, eff, c=eff, cmap='viridis', s=60)
            fig.colorbar(scatter, ax=ax, pad=0.1, label='Value Efficiency (%)')
            ax.plot(depths, runtimes, eff, '-', color='gray', alpha=0.5)
            for i, d in enumerate(depths):
                ax.text(d, runtimes[i], eff[i], f"D:{d}\n{eff[i]:.2f}%", fontsize=10)
            if len(depths) > 3:
                try:
                    from scipy.interpolate import griddata
                    xi = np.linspace(min(depths), max(depths), 20)
                    yi = np.linspace(min(runtimes), max(runtimes), 20)
                    X, Y = np.meshgrid(xi, yi)
                    Z = griddata((depths, runtimes), eff, (X, Y), method='cubic')
                    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
                except Exception:
                    warnings.warn('Surface fit failed')
            ax.set_xlabel('Max Child Depth', fontsize=12)
            ax.set_ylabel('Runtime (s)', fontsize=12)
            ax.set_zlabel('Value Efficiency (%)', fontsize=12)
            ax.set_title('3D Relationship: Depth, Runtime, Efficiency', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, '3d_surface_plot.png'), dpi=300)
            plt.close()
        except Exception as e:
            warnings.warn(f"3D plot failed: {e}")

    def plot_efficiency_with_without_partials(self):
        """Plot efficiency with and without partial settlements"""
        if not all(c in self.config_stats.columns for c in
                   ['normal_instruction_efficiency_mean', 'instruction_efficiency_mean']):
            print("WARNING: Missing normal efficiency metrics, skipping with/without partials comparison")
            return

        depths = self.config_stats.index.tolist()

        # Create plots for instruction efficiency
        plt.figure(figsize=(12, 8))
        normal_inst = self.config_stats['normal_instruction_efficiency_mean']
        total_inst = self.config_stats['instruction_efficiency_mean']
        normal_inst_err = self.config_stats.get('normal_instruction_efficiency_std', None)
        total_inst_err = self.config_stats.get('instruction_efficiency_std', None)

        width = 0.35
        x = np.arange(len(depths))

        bars1 = plt.bar(x - width / 2, normal_inst, width,
                        yerr=normal_inst_err, capsize=5,
                        label="Without Partial Settlement", color="lightblue")
        bars2 = plt.bar(x + width / 2, total_inst, width,
                        yerr=total_inst_err, capsize=5,
                        label="With Partial Settlement", color="green")

        # Annotate bars
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                     f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

        for i, bar in enumerate(bars2):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                     f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

            # Add annotation for the contribution
            contribution = total_inst.iloc[i] - normal_inst.iloc[i]
            if contribution > 1:
                plt.text(bar.get_x() + bar.get_width() / 2, height - 1.5,
                         f"+{contribution:.1f}%", ha="center", va="top",
                         fontsize=8, color="darkgreen", fontweight="bold")

        plt.xlabel('Max Child Depth', fontsize=14)
        plt.ylabel('Instruction Efficiency (%)', fontsize=14)
        plt.title('Instruction Efficiency Comparison: With vs Without Partial Settlement', fontsize=16)
        plt.xticks(x, depths)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'instruction_efficiency_comparison.png'), dpi=300)
        plt.close()

        # Create plots for value efficiency
        plt.figure(figsize=(12, 8))
        normal_val = self.config_stats['normal_value_efficiency_mean']
        total_val = self.config_stats['value_efficiency_mean']
        normal_val_err = self.config_stats.get('normal_value_efficiency_std', None)
        total_val_err = self.config_stats.get('value_efficiency_std', None)

        bars1 = plt.bar(x - width / 2, normal_val, width,
                        yerr=normal_val_err, capsize=5,
                        label="Without Partial Settlement", color="lightblue")
        bars2 = plt.bar(x + width / 2, total_val, width,
                        yerr=total_val_err, capsize=5,
                        label="With Partial Settlement", color="green")

        # Annotate bars
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                     f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

        for i, bar in enumerate(bars2):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                     f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

            # Add annotation for the contribution
            contribution = total_val.iloc[i] - normal_val.iloc[i]
            if contribution > 1:
                plt.text(bar.get_x() + bar.get_width() / 2, height - 1.5,
                         f"+{contribution:.1f}%", ha="center", va="top",
                         fontsize=8, color="darkgreen", fontweight="bold")

        plt.xlabel('Max Child Depth', fontsize=14)
        plt.ylabel('Value Efficiency (%)', fontsize=14)
        plt.title('Value Efficiency Comparison: With vs Without Partial Settlement', fontsize=16)
        plt.xticks(x, depths)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'value_efficiency_comparison.png'), dpi=300)
        plt.close()

        print("Saved efficiency comparison plots")

    def generate_all_visualizations(self):

        print("Generating settled on-time plots...")
        self.plot_settled_on_time_vs_depth()
        print("Generating runtime vs efficiency...")
        self.plot_runtime_vs_efficiency()
        print("Generating efficiency vs depth...")
        self.plot_efficiency_vs_depth()
        print("Generating settled amount vs depth...")
        self.plot_settled_amount_vs_depth()
        print("Generating partial settlements vs depth...")
        self.plot_partial_settlements_vs_depth()
        print("Generating avg tree depth vs max depth...")
        self.plot_avg_tree_depth_vs_max_depth()
        print("Generating memory usage vs depth...")
        self.plot_memory_usage_vs_depth()
        print("Generating runtime scaling...")
        self.plot_runtime_scaling()
        print("Generating elbow analysis...")
        self.plot_elbow_analysis()
        print("Generating 3D surface...")
        self.plot_3d_surface()
        print("Generating efficiency with/without partials comparison...")
        self.plot_efficiency_with_without_partials()
        print(f"All visualizations saved in {os.path.abspath(self.output_dir)}")
