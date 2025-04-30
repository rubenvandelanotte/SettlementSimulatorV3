import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DepthAnalyzer:
    """
    Analyzes depth statistics from settlement simulation results and produces
    individual and comparative visualizations:
      - Depth distribution per configuration
      - Status distribution by depth per configuration
      - Success rate by depth per configuration
      - Comparative plots across configurations
      - Export summary CSV/JSON
    """

    def __init__(self, input_dir: str, output_dir: str, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "depth_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def _extract_config_name(self, filename: str) -> str:
        match = re.search(r'(?:config|truecount)(\d+)', filename, re.IGNORECASE)
        if match:
            return f"Config {int(match.group(1))}"
        return os.path.splitext(filename)[0]

    def _sort_configs(self, configs):
        def key_fn(c):
            m = re.search(r'(\d+)', c)
            return int(m.group(1)) if m else float('inf')
        return sorted(configs, key=key_fn)

    def _build_dataframe(self):
        depth_records, status_records = [], []
        for filename, data in self.suite.statistics.items():
            cfg = self._extract_config_name(filename)
            for depth_str, cnt in data.get('depth_counts', {}).items():
                try:
                    d = int(depth_str)
                    depth_records.append({'config': cfg, 'depth': d, 'count': int(cnt)})
                except:
                    continue
            for depth_str, smap in data.get('depth_status_counts', {}).items():
                try:
                    d = int(depth_str)
                except:
                    continue
                if isinstance(smap, dict):
                    for status, cnt in smap.items():
                        try:
                            status_records.append({'config': cfg, 'depth': d, 'status': status, 'count': int(cnt)})
                        except:
                            continue
        return pd.DataFrame(depth_records), pd.DataFrame(status_records)

    def run(self):
        df_depth, df_status = self._build_dataframe()
        if df_depth.empty or df_status.empty:
            print("[WARNING] No depth statistics found. Skipping depth analysis.")
            return

        # individual plots
        self.plot_depth_distribution(df_depth)
        self.plot_status_by_depth(df_status)
        self.plot_success_rate_by_depth(df_status)

        # comparative plots
        self.compare_depth_distributions(df_depth)
        self.compare_status_distributions(df_status)
        self.compare_success_rates(df_status)
        self.compare_total_instructions(df_depth)
        self.compare_normalized_completion_rate(df_status)
        self.plot_success_rate_heatmap(df_status, df_depth)

        # summary export
        self.export_summary(df_depth, df_status)

    def plot_depth_distribution(self, df):
        for cfg, grp in df.groupby('config'):
            fig, ax = plt.subplots()
            grp_sorted = grp.sort_values('depth')
            depths = grp_sorted['depth'].tolist()
            counts = grp_sorted['count'].tolist()
            ax.bar(depths, counts, color='skyblue')
            ax.set_title(f'Depth Distribution - {cfg}')
            ax.set_xlabel('Depth')
            ax.set_ylabel('Count')
            ax.set_xticks(depths)
            ax.set_xticklabels(depths)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            od = os.path.join(self.output_dir, cfg)
            os.makedirs(od, exist_ok=True)
            fig.savefig(os.path.join(od, 'depth_distribution.png'), dpi=300)
            plt.close(fig)

    def plot_status_by_depth(self, df):
        for cfg, grp in df.groupby('config'):
            pivot = grp.pivot_table(index='depth', columns='status', values='count', aggfunc='sum', fill_value=0)
            depths = pivot.index.tolist()
            fig, ax = plt.subplots(figsize=(10,6))
            pivot.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'Status by Depth - {cfg}')
            ax.set_xlabel('Depth')
            ax.set_ylabel('Count')
            ax.set_xticks(range(len(depths)))
            ax.set_xticklabels(depths)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            od = os.path.join(self.output_dir, cfg)
            os.makedirs(od, exist_ok=True)
            fig.savefig(os.path.join(od, 'status_by_depth.png'), dpi=300)
            plt.close(fig)

    def plot_success_rate_by_depth(self, df):
        for cfg, grp in df.groupby('config'):
            pivot = grp.pivot_table(index='depth', columns='status', values='count', aggfunc='sum', fill_value=0)
            depths = sorted(pivot.index.tolist())
            rates = [(pivot.loc[d].get('Settled on time',0) + pivot.loc[d].get('Settled late',0)) / pivot.loc[d].sum() * 100 if pivot.loc[d].sum()>0 else 0 for d in depths]
            fig, ax = plt.subplots()
            ax.plot(depths, rates, 'o-', linewidth=2)
            ax.set_title(f'Success Rate by Depth - {cfg}')
            ax.set_xlabel('Depth')
            ax.set_ylabel('Success Rate (%)')
            ax.set_xticks(depths)
            ax.set_xticklabels(depths)
            ax.grid(True, linestyle='--', alpha=0.7)
            od = os.path.join(self.output_dir, cfg)
            os.makedirs(od, exist_ok=True)
            fig.savefig(os.path.join(od, 'success_rate_by_depth.png'), dpi=300)
            plt.close(fig)

    def compare_depth_distributions(self, df):
        configs = self._sort_configs(df['config'].unique())
        fig, ax = plt.subplots(figsize=(12,8))
        for cfg in configs:
            grp = df[df['config']==cfg].sort_values('depth')
            ax.plot(grp['depth'], grp['count'], 'o-', label=cfg)
        ax.set_title('Depth Distributions Across Configurations')
        ax.set_xlabel('Depth')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'compare_depth_distributions.png'), dpi=300)
        plt.close(fig)

    def compare_status_distributions(self, df):
        configs = self._sort_configs(df['config'].unique())
        statuses = sorted(df['status'].unique())
        depths = sorted(df['depth'].unique())
        fig, ax = plt.subplots(figsize=(14,8))
        for cfg in configs:
            sub = df[df['config']==cfg]
            for status in statuses:
                counts = [sub[(sub['depth']==d)&(sub['status']==status)]['count'].sum() for d in depths]
                ax.plot(depths, counts, 'o-', label=f"{cfg}-{status}")
        ax.set_title('Status Distributions Across Configurations')
        ax.set_xlabel('Depth')
        ax.set_ylabel('Count')
        ax.legend(ncol=2)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'compare_status_distributions.png'), dpi=300)
        plt.close(fig)

    def compare_success_rates(self, df):
        configs = self._sort_configs(df['config'].unique())
        fig, ax = plt.subplots(figsize=(12,8))
        depths_all = sorted(df['depth'].unique())
        for cfg in configs:
            sub = df[df['config']==cfg]
            pivot = sub.pivot_table(index='depth', columns='status', values='count', aggfunc='sum', fill_value=0)
            depths = sorted(pivot.index)
            rates = [(pivot.loc[d].get('Settled on time',0) + pivot.loc[d].get('Settled late',0)) / pivot.loc[d].sum()*100 if pivot.loc[d].sum()>0 else 0 for d in depths]
            ax.plot(depths, rates, 'o-', label=cfg)
        ax.set_title('Success Rates by Depth Across Configurations')
        ax.set_xlabel('Depth')
        ax.set_ylabel('Success Rate (%)')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'compare_success_rates.png'), dpi=300)
        plt.close(fig)

    def compare_total_instructions(self, df):
        configs = self._sort_configs(df['config'].unique())
        totals = [df[df['config']==cfg]['count'].sum() for cfg in configs]
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(range(len(configs)), totals, color='skyblue')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_title('Total Instructions per Configuration')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Total Instructions')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'compare_total_instructions.png'), dpi=300)
        plt.close(fig)

    def compare_normalized_completion_rate(self, df):
        configs = self._sort_configs(df['config'].unique())
        rates = []
        for cfg in configs:
            sub = df[df['config']==cfg]
            total = sub['count'].sum()
            success = sub[sub['status'].isin(['Settled on time','Settled late'])]['count'].sum()
            rates.append(success/total*100 if total>0 else 0)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(range(len(configs)), rates, color='lightgreen')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_title('Normalized Completion Rate by Configuration')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Completion Rate (%)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'compare_normalized_completion_rate.png'), dpi=300)
        plt.close(fig)

    def plot_success_rate_heatmap(self, df_status, df_depth):
        configs = self._sort_configs(df_status['config'].unique())
        depths = sorted(df_depth['depth'].unique())
        heat = np.full((len(depths), len(configs)), np.nan)
        for j, cfg in enumerate(configs):
            sub = df_status[df_status['config']==cfg]
            pivot = sub.pivot_table(index='depth', columns='status', values='count', aggfunc='sum', fill_value=0)
            for i, d in enumerate(depths):
                if d in pivot.index:
                    row = pivot.loc[d]
                    tot = row.sum()
                    succ = row.get('Settled on time',0) + row.get('Settled late',0)
                    heat[i,j] = succ/tot*100 if tot>0 else np.nan
        if np.any(~np.isnan(heat)):
            fig, ax = plt.subplots(figsize=(12,10))
            sns.heatmap(heat, xticklabels=configs, yticklabels=depths, cmap='YlGnBu', ax=ax)
            ax.set_title('Heatmap of Success Rate by Depth and Configuration')
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Depth')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'success_rate_heatmap.png'), dpi=300)
            plt.close(fig)

    def export_summary(self, df_depth, df_status):
        rows = []
        for cfg in self._sort_configs(df_depth['config'].unique()):
            tot_inst = int(df_depth[df_depth['config']==cfg]['count'].sum())
            sub = df_status[df_status['config']==cfg]
            total = sub['count'].sum()
            succ = sub[sub['status'].isin(['Settled on time','Settled late'])]['count'].sum()
            rate = succ/total*100 if total>0 else 0
            maxd = int(df_depth[df_depth['config']==cfg]['depth'].max())
            rows.append({'Configuration':cfg,'Total Instructions':tot_inst,'Success Rate (%)':rate,'Max Depth':maxd})
        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(os.path.join(self.output_dir,'depth_summary.csv'), index=False)
        with open(os.path.join(self.output_dir,'depth_summary.json'),'w') as f:
            json.dump(rows, f, indent=2)
