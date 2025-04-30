import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

class LatenessAnalyzer:
    def __init__(self, input_dir, output_dir, suite):
        """
        Analyze and visualize on-time vs late settlements by configuration and depth.
        Expects suite.statistics[file] to contain:
          - settled_ontime_rtp, settled_ontime_batch
          - settled_late_rtp, settled_late_batch
          - settled_on_time_amount, settled_late_amount
          - intended_amount
          - optional depth_status_counts: { depth_str: { "Settled on time": int, "Settled late": int, … } }
        """
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "lateness_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        self.suite = suite

    def run(self):
        config_df, depth_df = self._build_dataframes()
        if config_df.empty:
            print("[WARNING] No lateness data to plot.")
            return

        self._plot_late_percentage_by_config(config_df)
        # only plot depth‐based charts if we have per‐depth status counts
        if "late_depth" in depth_df.columns:
            self._plot_lateness_by_depth(depth_df)
            self._plot_lateness_depth_config_heatmap(depth_df)
        self._plot_ontime_vs_late_amounts(config_df)
        self._plot_ontime_vs_late_counts(config_df)
        self._plot_normalized_settlement_amount_trends(config_df)

    def _extract_config(self, filename):
        m = re.search(r'(?:config|truecount)(\d+)', filename)
        return int(m.group(1)) if m else None

    def _build_dataframes(self):
        config_records = []
        depth_records = []

        for fname, stats in self.suite.statistics.items():
            cfg = self._extract_config(fname)
            if cfg is None:
                continue

            # --- per-configuration totals
            ontime_ct  = stats.get("settled_ontime_rtp", 0) + stats.get("settled_ontime_batch", 0)
            late_ct    = stats.get("settled_late_rtp", 0)   + stats.get("settled_late_batch", 0)
            ontime_amt = stats.get("settled_on_time_amount", 0)
            late_amt   = stats.get("settled_late_amount", 0)
            intended   = stats.get("intended_amount", 1)

            config_records.append({
                "config": cfg,
                "ontime_count":  ontime_ct,
                "late_count":    late_ct,
                "ontime_amount": ontime_amt,
                "late_amount":   late_amt,
                "intended_amt":  intended
            })

            # --- per-depth status (if available)
            depth_status = stats.get("depth_status_counts", {})
            if depth_status:
                for dstr, statuses in depth_status.items():
                    try:
                        d = int(dstr)
                    except ValueError:
                        continue
                    depth_records.append({
                        "config":     cfg,
                        "depth":      d,
                        "ontime_depth": statuses.get("Settled on time", 0),
                        "late_depth":   statuses.get("Settled late",   0)
                    })

        config_df = pd.DataFrame(config_records)
        depth_df  = pd.DataFrame(depth_records)

        # ensure types
        if not config_df.empty:
            config_df["config"] = config_df["config"].astype(int)
            config_df.sort_values("config", inplace=True)

        if not depth_df.empty:
            depth_df["config"] = depth_df["config"].astype(int)
            depth_df["depth"]  = depth_df["depth"].astype(int)

        return config_df, depth_df

    def _plot_late_percentage_by_config(self, df):
        # sum across runs to get true totals, then compute percent
        agg = df.groupby("config")[["ontime_count","late_count"]].sum()
        agg["late_pct"] = 100 * agg["late_count"] / (agg["ontime_count"] + agg["late_count"])

        fig, ax = plt.subplots(figsize=(12,6))
        bars = ax.bar(agg.index, agg["late_pct"], color="tomato")
        for cfg, bar in zip(agg.index, bars):
            pct = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, pct+0.5, f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize= 9)

        ax.set_title("Late Settlement Percentage by Configuration")
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Late Settlements (%)")
        ax.set_xticks(agg.index)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "late_settlement_percentage.png"))
        plt.close()

    def _plot_lateness_by_depth(self, df):
        # aggregate all depth‐status across configs
        agg = df.groupby("depth")[["ontime_depth","late_depth"]].sum()
        agg["late_pct"] = 100 * agg["late_depth"] / (agg["ontime_depth"] + agg["late_depth"])

        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(agg.index, agg["late_pct"], "o-", color="purple")
        for d, pct in zip(agg.index, agg["late_pct"]):
            ax.text(d, pct+0.5, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_title("Late Settlement Percentage by Instruction Depth")
        ax.set_xlabel("Instruction Depth")
        ax.set_ylabel("Late Settlements (%)")
        ax.set_xticks(agg.index)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_by_depth.png"))
        plt.close()

    def _plot_lateness_depth_config_heatmap(self, df):
        # pivot depth × config
        agg = df.groupby(["depth","config"])[["ontime_depth","late_depth"]].sum()
        agg["late_pct"] = 100 * agg["late_depth"] / (agg["ontime_depth"] + agg["late_depth"])
        heat = agg["late_pct"].unstack().sort_index(axis=1)

        fig, ax = plt.subplots(figsize=(14,8))
        sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=.5, ax=ax)
        ax.set_title("Late Settlement % by Depth and Configuration")
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Instruction Depth")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lateness_depth_config_heatmap.png"))
        plt.close()

    def _plot_ontime_vs_late_amounts(self, df):
        # ---- aggregate and scale to billions
        agg = df.groupby("config")[["ontime_amount","late_amount"]].mean()
        scale = 1e9
        agg_billions = agg / scale

        fig, ax = plt.subplots(figsize=(12,6))
        x = agg_billions.index.to_numpy()
        w = 0.4

        bars1 = ax.bar(x - w/2, agg_billions["ontime_amount"],
                       width=w, label="On-Time (€ B)", color="green")
        bars2 = ax.bar(x + w/2, agg_billions["late_amount"],
                       width=w, label="Late (€ B)",    color="orange")

        # annotate in billions
        for bar in (*bars1, *bars2):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h + agg_billions.values.max() * 0.01,
                f"{h:.2f} B",
                ha="center", va="bottom", fontsize=8
            )

        # format y-axis ticks as "1.2 B"
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y:.1f} B")
        )

        ax.set_title("On-Time vs Late Settlement Amounts by Configuration")
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Settled Amount (€ billions)")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ontime_vs_late_amounts.png"))
        plt.close()

    def _plot_ontime_vs_late_counts(self, df):
        agg = df.groupby("config")[["ontime_count","late_count"]].mean()

        fig, ax = plt.subplots(figsize=(12,6))
        x = agg.index.to_numpy()
        w = 0.4
        bars1 = ax.bar(x - w/2, agg["ontime_count"], width=w, label="On‐Time", color="green")
        bars2 = ax.bar(x + w/2, agg["late_count"],   width=w, label="Late",    color="orange")

        maxval = (agg["ontime_count"] + agg["late_count"]).max()
        for bar in (*bars1, *bars2):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h + maxval*0.005, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7)

        ax.set_title("Average On-Time vs Late Settlement Counts")
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Settlements (count)")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ontime_vs_late_counts.png"))
        plt.close()

    import os
    import matplotlib.pyplot as plt

    def _plot_normalized_settlement_amount_trends(self, config_df):
        """
        Teken % van intended amount *per run*, en neem dan het gemiddelde per config.
        Daardoor komt de On-Time % mooi overeen met wat je in de Value Efficiency-analyse krijgt.
        """

        # 1) Bereken per run de 3 percentages
        df = config_df.copy()
        df["pct_total"] = (df["ontime_amount"] + df["late_amount"]) / df["intended_amt"] * 100
        df["pct_ontime"] = df["ontime_amount"] / df["intended_amt"] * 100
        df["pct_late"] = df["late_amount"] / df["intended_amt"] * 100

        # 2) Gemiddelde per config
        pct = (
            df
            .groupby("config")[["pct_total", "pct_ontime", "pct_late"]]
            .mean()
            .sort_index()
        )

        # 3) Plotten
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(pct.index, pct["pct_total"], "o-", label="Total Settled (%)", color="blue")
        ax.plot(pct.index, pct["pct_ontime"], "s--", label="On-Time Settled (%)", color="green")
        ax.plot(pct.index, pct["pct_late"], "d--", label="Late Settled (%)", color="orange")

        # 4) Annotaties
        for cfg, val in pct["pct_total"].items():
            ax.text(cfg, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        for cfg, val in pct["pct_ontime"].items():
            ax.text(cfg, val - 0.8, f"{val:.1f}%", ha="center", va="top", fontsize=8)
        for cfg, val in pct["pct_late"].items():
            ax.text(cfg, val - 0.8, f"{val:.1f}%", ha="center", va="top", fontsize=8)

        ax.set_title("Normalized Settlement Amount Trends (%)")
        ax.set_xlabel("Configuration")
        ax.set_ylabel("% of Intended Amount Settled")
        ax.set_xticks(pct.index)
        ax.legend(loc="lower right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        # 5) Opslaan
        plt.savefig(os.path.join(self.output_dir, "normalized_settlement_amount_trends_fixed.png"))
        plt.close()
        print("✔️  Gecorrigeerde normalized trends opgeslagen.")



