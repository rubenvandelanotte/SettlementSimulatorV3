import os
import json
import re
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter

class SettlementAmountVisualizer:
    """
    Enhanced visualization suite for settlement simulations.

    - Uniform color palette
    - Axis labels and data point annotations
    - Settled amounts formatted in billions
    """
    DEFAULT_FIGSIZE = (12, 8)
    DEFAULT_DPI = 300
    COLORS = {
        'instruction': '#1f77b4',
        'value':       '#ff7f0e',
        'settled':     '#2ca02c',
        'mothers':     '#d62728',
        'partial':     '#9467bd',
        'memory':      '#8c564b',
        'runtime':     '#e377c2'
    }

    def __init__(self, results_dir, output_dir="settlement_visuals"):
        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = results_dir
        self.output_dir = output_dir
        self._load_data()
        self._compute_statistics()

    def _load_data(self):
        recs = []
        for fname in os.listdir(self.results_dir):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(self.results_dir, fname)
            try:
                data = json.load(open(path))
            except Exception:
                warnings.warn(f"Could not read {fname}")
                continue
            m = re.search(r'minpct(\d+)', fname)
            pct = int(m.group(1)) / 1000.0 if m else data.get('min_settlement_percentage')
            if pct is None:
                warnings.warn(f"Skipping {fname}: missing percentage")
                continue
            on_time = data.get('settled_on_time_amount', 0)
            late    = data.get('settled_late_amount', 0)
            settled_amount = on_time + late
            if 'mothers_effectively_settled' in data:
                mothers = data['mothers_effectively_settled']

            recs.append({
                'min_settlement_percentage': pct,
                'instruction_efficiency':      data.get('instruction_efficiency', np.nan),
                'value_efficiency':            data.get('value_efficiency', np.nan),
                'runtime_seconds':             data.get('execution_time_seconds', np.nan),
                'memory_usage_mb':             data.get('memory_usage_mb', np.nan),
                'mothers_effectively_settled': mothers,
                'partial_settlements':         data.get('partial_settlements', np.nan),
                'settled_amount':              settled_amount
            })
        self.df = pd.DataFrame(recs)
        if self.df.empty:
            raise RuntimeError("No valid JSON stats found.")

    def _compute_statistics(self):
        grp = self.df.groupby('min_settlement_percentage')
        stats = {}
        for col in self.df.columns:
            if col == 'min_settlement_percentage':
                continue
            stats[f"{col}_mean"] = grp[col].mean()
            stats[f"{col}_std"]  = grp[col].std()
        self.stats = pd.DataFrame(stats)

    def _format_billions(self, x, pos):
        return f"{x/1e9:.1f}B"

    def plot_runtime_vs_efficiency(self):
        x = self.stats['runtime_seconds_mean']
        pct = self.stats.index.tolist()
        for key, label, color in [
            ('instruction_efficiency', 'Instruction Efficiency', 'instruction'),
            ('value_efficiency',       'Value Efficiency',       'value')
        ]:
            y = self.stats[f'{key}_mean']
            yerr = self.stats[f'{key}_std']
            plt.figure(figsize=self.DEFAULT_FIGSIZE)
            plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color=self.COLORS[color], label=label)
            coeff = np.polyfit(x, y, 1)
            trend = np.poly1d(coeff)
            plt.plot(x, trend(x), '--', color=self.COLORS[color])
            corr = np.corrcoef(x, y)[0,1]
            for xi, yi in zip(x, y):
                plt.annotate(f"{yi:.1f}%", (xi, yi), xytext=(5,5), textcoords='offset points')
            plt.title(f"Runtime vs {label} (corr={corr:.2f})")
            plt.xlabel('Runtime (s)')
            plt.ylabel(f'{label} (%)')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'runtime_vs_{color}.png'), dpi=self.DEFAULT_DPI)
            plt.close()

    def plot_efficiency_vs_percentage(self):
        p = self.stats.index.tolist()
        labels = [f"{v*100:.1f}%" for v in p]
        fig, ax1 = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        ax2 = ax1.twinx()
        ax1.errorbar(p, self.stats['instruction_efficiency_mean'], yerr=self.stats['instruction_efficiency_std'], fmt='o-', capsize=5, color=self.COLORS['instruction'], label='Instr')
        ax2.errorbar(p, self.stats['value_efficiency_mean'], yerr=self.stats['value_efficiency_std'], fmt='s--', capsize=5, color=self.COLORS['value'], label='Value')
        for x, y in zip(p, self.stats['instruction_efficiency_mean']):
            ax1.annotate(f"{y:.1f}%", (x,y), xytext=(0,5), textcoords='offset points')
        for x, y in zip(p, self.stats['value_efficiency_mean']):
            ax2.annotate(f"{y:.1f}%", (x,y), xytext=(0,-10), textcoords='offset points')
        ax1.set_xticks(p); ax1.set_xticklabels(labels)
        ax1.set_xlabel('Min Settlement %'); ax1.set_ylabel('Instruction Efficiency (%)')
        ax2.set_ylabel('Value Efficiency (%)')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2)
        plt.title('Efficiency vs Settlement %')
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'efficiency_vs_percentage.png'), dpi=self.DEFAULT_DPI)
        plt.close()

    def plot_settled_amount_vs_percentage(self):
        p = self.stats.index.tolist()
        labels = [f"{v*100:.1f}%" for v in p]
        vals = self.stats['settled_amount_mean']
        errs = self.stats['settled_amount_std']
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=self.COLORS['settled'])
        ax.plot(range(len(p)), vals, 'o-', color=self.COLORS['settled'])
        ax.yaxis.set_major_formatter(FuncFormatter(self._format_billions))
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height()/1e9:.1f}B", ha='center', va='bottom')
        ax.set_xlabel('Min Settlement %'); ax.set_ylabel('Settled Amount (B)')
        plt.title('Settled Amount vs Settlement %')
        plt.grid(axis='y', alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'settled_amount_vs_percentage.png'), dpi=self.DEFAULT_DPI)
        plt.close()

    def plot_effective_mothers_vs_percentage(self):
        p = self.stats.index.tolist()
        labels = [f"{v*100:.1f}%" for v in p]
        vals = self.stats['mothers_effectively_settled_mean']
        errs = self.stats['mothers_effectively_settled_std']
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=self.COLORS['mothers'])
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{int(b.get_height())}", ha='center', va='bottom')
        ax.set_xlabel('Min Settlement %'); ax.set_ylabel('Mothers Settled')
        plt.title('Effectively Settled Mothers vs Settlement %')
        plt.grid(axis='y', alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'effective_mothers_vs_percentage.png'), dpi=self.DEFAULT_DPI)
        plt.close()

    def plot_partial_settlements_vs_percentage(self):
        p = self.stats.index.tolist()
        labels = [f"{v*100:.1f}%" for v in p]
        vals = self.stats['partial_settlements_mean']
        errs = self.stats['partial_settlements_std']
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=self.COLORS['partial'])
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.0f}", ha='center', va='bottom')
        ax.set_xlabel('Min Settlement %'); ax.set_ylabel('Partial Settlements')
        plt.title('Partial Settlements vs Settlement %')
        plt.grid(axis='y', alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'partial_settlements_vs_percentage.png'), dpi=self.DEFAULT_DPI)
        plt.close()

    def plot_memory_usage_vs_percentage(self):
        p = self.stats.index.tolist()
        labels = [f"{v*100:.1f}%" for v in p]
        vals = self.stats['memory_usage_mb_mean']
        errs = self.stats['memory_usage_mb_std']
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        ax.errorbar(p, vals, yerr=errs, fmt='o-', capsize=5, color=self.COLORS['memory'])
        for x,y in zip(p, vals): ax.annotate(f"{y:.1f} MB", (x,y), xytext=(5,5), textcoords='offset points')
        ax.set_xlabel('Min Settlement %'); ax.set_ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs Settlement %')
        ax.set_xticks(p); ax.set_xticklabels(labels)
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'memory_usage_vs_percentage.png'), dpi=self.DEFAULT_DPI)
        plt.close()

    def plot_runtime_scaling(self):
        p = self.stats.index.tolist()
        labels = [f"{v*100:.1f}%" for v in p]
        vals = self.stats['runtime_seconds_mean']
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        ax.plot(p, vals, 'o-', color=self.COLORS['runtime'])
        for x,y in zip(p, vals): ax.annotate(f"{y:.2f} s", (x,y), xytext=(5,5), textcoords='offset points')
        ax.set_xlabel('Min Settlement %'); ax.set_ylabel('Runtime (s)')
        plt.title('Runtime Scaling vs Settlement %')
        ax.set_xticks(p); ax.set_xticklabels(labels)
        try:
            from scipy.optimize import curve_fit
            def poly(x,a,b,c): return a*x**2 + b*x + c
            popt,_ = curve_fit(poly, np.array(p), np.array(vals))
            xs = np.linspace(min(p), max(p), 100)
            ax.plot(xs, poly(xs,*popt), '--', color=self.COLORS['runtime'], label='Fit')
            ax.legend()
        except:
            pass
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'runtime_scaling.png'), dpi=self.DEFAULT_DPI)
        plt.close()

    def plot_elbow_analysis(self):
        p = self.stats.index.tolist()
        labels = [f"{v*100:.1f}%" for v in p]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # Value
        y = self.stats['value_efficiency_mean']; ax = axes[0,0]
        ax.plot(p, y, 'o-', color=self.COLORS['value']); ax.set_title('Value Eff vs %'); ax.set_xticks(p); ax.set_xticklabels(labels)
        for x,v in zip(p,y): ax.annotate(f"{v:.1f}%", (x,v), xytext=(5,5), textcoords='offset points')
        ax.grid(alpha=0.3)
        # Runtime
        y = self.stats['runtime_seconds_mean']; ax=axes[0,1]
        ax.plot(p,y,'o-', color=self.COLORS['runtime']); ax.set_title('Runtime vs %'); ax.set_xticks(p); ax.set_xticklabels(labels)
        for x,v in zip(p,y): ax.annotate(f"{v:.2f}s", (x,v), xytext=(5,5), textcoords='offset points')
        ax.grid(alpha=0.3)
        # Eff/Time
        ratio = self.stats['value_efficiency_mean'] / self.stats['runtime_seconds_mean']; ax=axes[1,0]
        ax.plot(p,ratio,'o-', color=self.COLORS['instruction']); ax.set_title('Eff/Time Ratio'); ax.set_xticks(p); ax.set_xticklabels(labels)
        for x,v in zip(p,ratio): ax.annotate(f"{v:.3f}", (x,v), xytext=(5,5), textcoords='offset points')
        best = p[np.argmax(ratio)]; ax.axvline(best, linestyle='--', color='gray'); ax.grid(alpha=0.3)
        # Tradeoff
        rt = self.stats['runtime_seconds_mean']; ve = self.stats['value_efficiency_mean']
        rt_n = (rt - rt.min())/(rt.max()-rt.min()); ve_n = (ve - ve.min())/(ve.max()-ve.min())
        trade = ve_n - rt_n; ax=axes[1,1]
        ax.plot(p, rt_n, 'o-', label='Runtime norm', color=self.COLORS['runtime'])
        ax.plot(p, ve_n, 'o-', label='Value norm', color=self.COLORS['value'])
        ax.plot(p, trade,'o-', label='Tradeoff', color=self.COLORS['partial'])
        opt = p[np.argmax(trade)]; ax.axvline(opt, linestyle='--', color='gray')
        ax.set_title('Tradeoff'); ax.legend(); ax.set_xticks(p); ax.set_xticklabels(labels); ax.grid(alpha=0.3)
        plt.suptitle('Elbow Analysis'); plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(os.path.join(self.output_dir, 'elbow_analysis.png'), dpi=self.DEFAULT_DPI)
        plt.close()

    def plot_3d_surface(self):
        try:
            p = self.stats.index.tolist()
            rt = self.stats['runtime_seconds_mean'].tolist()
            ve = self.stats['value_efficiency_mean'].tolist()
            fig = plt.figure(figsize=self.DEFAULT_FIGSIZE)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(p, rt, ve, c=ve, cmap='viridis', s=60)
            fig.colorbar(sc, pad=0.1, label='Value Eff (%)')
            for x,y,z in zip(p, rt, ve): ax.text(x, y, z, f"{z:.1f}%")
            xi = np.linspace(min(p), max(p), 20)
            yi = np.linspace(min(rt), max(rt), 20)
            X, Y = np.meshgrid(xi, yi)
            from scipy.interpolate import griddata
            Z = griddata((p, rt), ve, (X, Y), method='cubic')
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
            ax.set_xlabel('Min Settlement %'); ax.set_ylabel('Runtime (s)'); ax.set_zlabel('Value Eff (%)')
            plt.tight_layout(); plt.savefig(os.path.join(self.output_dir, '3d_surface.png'), dpi=self.DEFAULT_DPI); plt.close()
        except Exception as e:
            warnings.warn(f"3D surface failed: {e}")

    def plot_comparative_metrics(self):
        p = self.stats.index.tolist()
        labels = [f"{v*100:.1f}%" for v in p]
        metrics = [
            ('instruction_efficiency_mean', 'Instr', 'instruction'),
            ('value_efficiency_mean',       'Value',       'value'),
            ('partial_settlements_mean',    'Partial',     'partial')
        ]
        normalized = {}
        for key, _, _ in metrics:
            series = self.stats[key]
            normalized[key] = (series - series.min()) / (series.max() - series.min() if series.max() != series.min() else 1)
        idx = np.arange(len(p)); width = 0.2
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        bars = []
        for i, (key, label, color_key) in enumerate(metrics):
            bar = ax.bar(idx + i*width, normalized[key], width, label=label, color=self.COLORS[color_key])
            bars.append((bar, key))
        ax.set_xticks(idx + width); ax.set_xticklabels(labels)
        ax.set_xlabel('Min Settlement %'); ax.set_ylabel('Normalized')
        plt.title('Comparative Metrics')
        plt.legend()
        for bar_group, key in bars:
            for rect in bar_group:
                h = rect.get_height()
                ax.annotate(f"{h:.2f}", (rect.get_x()+rect.get_width()/2, h), xytext=(0,5), textcoords='offset points', ha='center')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,'comparative_metrics.png'),dpi=self.DEFAULT_DPI)
        plt.close()


    def plot_efficiency_distribution(self):
        df = self.df.copy()
        df['label'] = df['min_settlement_percentage'].apply(lambda x: f"{x*100:.1f}%")
        labels = df['label'].unique().tolist()
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        # Instruction efficiency distribution
        sns.violinplot(
            x='label',
            y='instruction_efficiency',
            data=df,
            ax=axes[0],
            color=self.COLORS['instruction'],
            inner='points'
        )
        axes[0].set_title('Instruction Efficiency Distribution')
        axes[0].set_xlabel('Min Settlement %')
        axes[0].set_ylabel('Instruction Efficiency (%)')
        axes[0].grid(alpha=0.3)
        # Value efficiency distribution
        sns.violinplot(
            x='label',
            y='value_efficiency',
            data=df,
            ax=axes[1],
            color=self.COLORS['value'],
            inner='points'
        )
        axes[1].set_title('Value Efficiency Distribution')
        axes[1].set_xlabel('Min Settlement %')
        axes[1].set_ylabel('Value Efficiency (%)')
        axes[1].grid(alpha=0.3)
        plt.suptitle('Efficiency Distribution')
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(os.path.join(self.output_dir, 'efficiency_distribution.png'), dpi=self.DEFAULT_DPI)
        plt.close()

    def generate_all_visualizations(self):
        self.plot_runtime_vs_efficiency()
        self.plot_efficiency_vs_percentage(),
        self.plot_settled_amount_vs_percentage(),
        self.plot_effective_mothers_vs_percentage(),
        self.plot_partial_settlements_vs_percentage(),
        self.plot_memory_usage_vs_percentage(),
        self.plot_runtime_scaling(),
        self.plot_elbow_analysis(),
        self.plot_3d_surface(),
        self.plot_comparative_metrics(),
        self.plot_efficiency_distribution()
        print(f"All visualizations saved to {os.path.abspath(self.output_dir)}")
