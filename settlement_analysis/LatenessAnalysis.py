import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LatenessAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "lateness_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        df, total_intended_amounts = self._build_lateness_dataframe()
        print(f"[DEBUG] LatenessAnalyzer DataFrame shape: {df.shape}")

        if df.empty:
            print("[WARNING] No lateness data available.")
            return

        self._plot_late_percentage_by_config(df)
        self._plot_lateness_by_depth(df)
        self._plot_lateness_depth_config_heatmap(df)
        self._plot_ontime_vs_late_amounts_fixed(df)
        self._plot_ontime_vs_late_counts(df)
        self._plot_settlement_amount_trends_fixed(df, total_intended_amounts)

    def _build_lateness_dataframe(self):
        records = []
        total_intended_amounts = {}

        for filename, stats in self.suite.statistics.items():
            config = self._extract_config_name(filename)

            ontime_count = stats.get("settled_ontime_rtp", 0) + stats.get("settled_ontime_batch", 0)
            late_count = stats.get("settled_late_rtp", 0) + stats.get("settled_late_batch", 0)

            ontime_amount = stats.get("settled_on_time_amount", 0)
            late_amount = stats.get("settled_late_amount", 0)

            depth_counts = stats.get("depth_counts", {})

            for depth, count in depth_counts.items():
                try:
                    depth_int = int(depth)
                except ValueError:
                    continue

                records.append({
                    "config": int(config) if isinstance(config, str) and str(config).isdigit() else config,
                    "depth": depth_int,
                    "total_count": count,
                    "ontime_count": ontime_count,
                    "late_count": late_count,
                    "ontime_amount": ontime_amount,
                    "late_amount": late_amount,
                })

            intended = stats.get("intended_amount", 0)
            total_intended_amounts[config] = intended

        df = pd.DataFrame(records)

        # New lines for robust typing:
        df["config"] = pd.to_numeric(df["config"], errors="coerce")
        df = df.dropna(subset=["config"])
        df["config"] = df["config"].astype(int)

        return df, total_intended_amounts

    def _extract_config_name(self, filename):
        """
        Extract config from filename.
        Supports both 'config' and 'truecount' formats.
        Returns an integer config number or None if not found.
        """
        import re
        match = re.search(r'(?:config|truecount)(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            print(f"[WARNING] Failed to parse config from {filename}")
            return None

    def _plot_late_percentage_by_config(self, df):
        df_grouped = df.groupby("config")[['ontime_count', 'late_count']].sum().sort_index()
        df_grouped['late_pct'] = df_grouped['late_count'] / (df_grouped['ontime_count'] + df_grouped['late_count']) * 100

        plt.figure(figsize=(14, 8))
        bars = plt.bar(df_grouped.index, df_grouped['late_pct'], color='tomato')
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=8)

        plt.title('Late Settlement Percentage by Configuration')
        plt.xlabel('Configuration')
        plt.ylabel('Late Settlement (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "late_settlement_percentage.png"))
        plt.close()

    def _plot_lateness_by_depth(self, df):
        df_depth = df.groupby('depth')[['ontime_count', 'late_count']].sum()
        df_depth['late_pct'] = df_depth['late_count'] / (df_depth['ontime_count'] + df_depth['late_count']) * 100

        plt.figure(figsize=(14, 8))
        plt.plot(df_depth.index, df_depth['late_pct'], marker='o', color='purple')
        for x, y in zip(df_depth.index, df_depth['late_pct']):
            plt.text(x, y + 1, f"{y:.1f}%", ha='center', va='bottom', fontsize=8)

        plt.title('Late Settlement Percentage by Instruction Depth')
        plt.xlabel('Instruction Depth')
        plt.ylabel('Late Settlement (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_by_depth.png"))
        plt.close()

    def _plot_lateness_depth_config_heatmap(self, df):
        grouped = df.groupby(['depth', 'config'])[['ontime_count', 'late_count']].sum()
        grouped['late_pct'] = grouped['late_count'] / (grouped['ontime_count'] + grouped['late_count']) * 100
        pivot = grouped['late_pct'].unstack(fill_value=0)

        plt.figure(figsize=(16, 10))
        ax = sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=0.5)
        ax.set_title('Late Settlement % by Depth and Configuration')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Instruction Depth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_depth_config_heatmap.png"))
        plt.close()

    def _plot_ontime_vs_late_amounts_fixed(self, df):
        df_grouped = df.groupby('config')[['ontime_amount', 'late_amount']].mean().sort_index()

        plt.figure(figsize=(16, 8))
        bars1 = plt.bar(df_grouped.index - 0.15, df_grouped['ontime_amount'], width=0.3, label='On-Time (€)', color='green')
        bars2 = plt.bar(df_grouped.index + 0.15, df_grouped['late_amount'], width=0.3, label='Late (€)', color='orange')
        for bar in bars1 + bars2:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + max(df_grouped['ontime_amount']+df_grouped['late_amount'])*0.01,
                     f"{h:.0f}", ha='center', va='bottom', fontsize=8)

        plt.title('On-Time vs Late Settlement Amounts (€)')
        plt.xlabel('Configuration')
        plt.ylabel('Amount (€)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ontime_vs_late_amounts_fixed.png"))
        plt.close()

    def _plot_ontime_vs_late_counts(self, df):
        df_grouped = df.groupby('config')[['ontime_count', 'late_count']].mean().sort_index()

        plt.figure(figsize=(16, 8))
        bars1 = plt.bar(df_grouped.index - 0.15, df_grouped['ontime_count'], width=0.3, label='On-Time', color='green')
        bars2 = plt.bar(df_grouped.index + 0.15, df_grouped['late_count'], width=0.3, label='Late', color='orange')
        for bar in bars1 + bars2:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + max(df_grouped['ontime_count']+df_grouped['late_count'])*0.01,
                     f"{h}", ha='center', va='bottom', fontsize=8)

        plt.title('On-Time vs Late Settlement Counts')
        plt.xlabel('Configuration')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ontime_vs_late_counts.png"))
        plt.close()

    def _plot_settlement_amount_trends_fixed(self, df, total_intended_amounts):
        df_grouped = df.groupby('config')[['ontime_amount', 'late_amount']].sum().sort_index()
        configs = df_grouped.index.values
        total_settled = df_grouped['ontime_amount'] + df_grouped['late_amount']
        intended = [total_intended_amounts.get(cfg, 1) for cfg in configs]

        pct_total  = total_settled / intended * 100
        pct_ontime = df_grouped['ontime_amount'] / intended * 100
        pct_late   = df_grouped['late_amount'] / intended * 100

        plt.figure(figsize=(16, 8))
        plt.plot(configs, pct_total, marker='o', label='Total Settled %', color='blue')
        plt.plot(configs, pct_ontime, marker='s', linestyle='--', label='On-Time %', color='green')
        plt.plot(configs, pct_late, marker='^', linestyle='--', label='Late %', color='orange')
        for x, y in zip(configs, pct_total):
            plt.text(x, y + 1, f"{y:.1f}%", ha='center', va='bottom', fontsize=8)
        for x, y in zip(configs, pct_ontime):
            plt.text(x, y - 1, f"{y:.1f}%", ha='center', va='top', fontsize=8)
        for x, y in zip(configs, pct_late):
            plt.text(x, y - 1, f"{y:.1f}%", ha='center', va='top', fontsize=8)

        plt.title('Normalized Settlement Amount Trends (%)')
        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Intended Amount')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "settlement_amount_trends_fixed.png"))
        plt.close()
