import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from collections import defaultdict


class SettlementTypeAnalyzer:
    """
    Analyzes and visualizes the comparison between normal first-try settlements
    and settlements via partial settlement mechanism (child instructions).
    """

    def __init__(self, input_dir, output_dir, suite):
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, "settlement_type_analysis")
        self.suite = suite
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """Execute all analysis and generate visualizations."""
        data = self._build_data()
        if not data:
            print("[WARNING] No settlement type data available to analyze.")
            return

        self._plot_settlement_type_comparison(data)
        self._plot_settlement_type_percentage(data)
        self._plot_settlement_type_ratio_by_config(data)
        self._plot_total_settled_stacked(data)
        self._plot_settlement_efficiency_comparison(data)

    def _extract_config_id(self, filename):
        """Extract configuration ID from filename."""
        match = re.search(r'(?:config|truecount)(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    def _build_data(self):
        """Build dataset from statistics files."""
        config_data = defaultdict(list)

        for filename, stats in self.suite.statistics.items():
            # Try to get config from metadata first
            meta = stats.get("config_metadata", {})
            config_id = meta.get("config_id")

            # If not found, try true_count
            if config_id is None:
                config_id = meta.get("true_count")

                # If still not found, extract from filename
                if config_id is None:
                    config_id = self._extract_config_id(filename)

                    if config_id is None:
                        print(f"[WARNING] Could not determine configuration for {filename}, skipping.")
                        continue

            normal_amount = stats.get("normal_settled_amount", 0)
            partial_amount = stats.get("partial_settled_amount", 0)
            intended_amount = stats.get("intended_amount", 0)

            # Try to get from statistics_tracker if not directly in stats
            if not normal_amount and not partial_amount:
                summary = stats.get("statistics_tracker_summary", {})
                if summary:
                    normal_amount = summary.get("normal_settled_amount", 0)
                    partial_amount = summary.get("partial_settled_amount", 0)

            # Only include if we have the settlement type breakdown
            if normal_amount or partial_amount:
                config_data[config_id].append({
                    "normal_amount": normal_amount,
                    "partial_amount": partial_amount,
                    "total_amount": normal_amount + partial_amount,
                    "intended_amount": intended_amount if intended_amount else normal_amount + partial_amount,
                    "normal_pct": (normal_amount / intended_amount * 100) if intended_amount else 0,
                    "partial_pct": (partial_amount / intended_amount * 100) if intended_amount else 0
                })
            else:
                print(f"[WARNING] No settlement type data found in {filename}")

        # Average values across runs for each config
        result = {}
        for config_id, runs in config_data.items():
            if not runs:
                continue

            avg_data = {}
            for key in runs[0].keys():
                avg_data[key] = sum(run[key] for run in runs) / len(runs)

            result[config_id] = avg_data

        return result

    def _plot_settlement_type_comparison(self, data):
        """Plot absolute amounts settled via normal vs partial settlement."""
        configs = sorted(data.keys())
        normal_amounts = [data[cfg]["normal_amount"] for cfg in configs]
        partial_amounts = [data[cfg]["partial_amount"] for cfg in configs]

        # Convert to billions for better readability
        scale = 1e9
        normal_amounts_b = [amt / scale for amt in normal_amounts]
        partial_amounts_b = [amt / scale for amt in partial_amounts]

        x = np.arange(len(configs))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(x - width / 2, normal_amounts_b, width,
                       label="Normal First-Try Settlement", color="steelblue")
        bars2 = ax.bar(x + width / 2, partial_amounts_b, width,
                       label="Partial Settlement", color="darkorange")

        # Annotate bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                    f"{height:.2f}B", ha="center", va="bottom", fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                    f"{height:.2f}B", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Configuration (Number of Institutions Allowing Partials)")
        ax.set_ylabel("Amount Settled (€ Billions)")
        ax.set_title("Comparison of Settlement Amounts: Normal vs Partial")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Config {cfg}" for cfg in configs])
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "settlement_type_comparison.png"), dpi=300)
        plt.close()
        print(
            f"[INFO] Saved settlement type comparison to {os.path.join(self.output_dir, 'settlement_type_comparison.png')}")

    def _plot_settlement_type_percentage(self, data):
        """Plot percentage breakdown of normal vs partial settlement."""
        configs = sorted(data.keys())

        # Extract percentages
        normal_pcts = []
        partial_pcts = []

        for cfg in configs:
            total = data[cfg]["normal_amount"] + data[cfg]["partial_amount"]
            if total > 0:
                normal_pcts.append(data[cfg]["normal_amount"] / total * 100)
                partial_pcts.append(data[cfg]["partial_amount"] / total * 100)
            else:
                normal_pcts.append(0)
                partial_pcts.append(0)

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(configs))
        ax.bar(x, normal_pcts, label="Normal First-Try Settlement", color="steelblue")
        ax.bar(x, partial_pcts, bottom=normal_pcts, label="Partial Settlement", color="darkorange")

        # Annotate segments
        for i, (normal, partial) in enumerate(zip(normal_pcts, partial_pcts)):
            # Normal settlement percentage
            if normal > 5:  # Only annotate if segment is large enough
                ax.text(i, normal / 2, f"{normal:.1f}%", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=9)

            # Partial settlement percentage
            if partial > 5:  # Only annotate if segment is large enough
                ax.text(i, normal + partial / 2, f"{partial:.1f}%", ha="center",
                        va="center", color="white", fontweight="bold", fontsize=9)

        ax.set_xlabel("Configuration (Number of Institutions Allowing Partials)")
        ax.set_ylabel("Percentage of Total Settled Amount")
        ax.set_title("Percentage of Settlement Amount by Type")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Config {cfg}" for cfg in configs])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "settlement_type_percentage.png"), dpi=300)
        plt.close()
        print(
            f"[INFO] Saved settlement type percentage to {os.path.join(self.output_dir, 'settlement_type_percentage.png')}")

    def _plot_settlement_type_ratio_by_config(self, data):
        """Plot the ratio of partial to normal settlement by configuration."""
        configs = sorted(data.keys())

        # Calculate ratios (partial / normal)
        ratios = []
        for cfg in configs:
            normal = data[cfg]["normal_amount"]
            partial = data[cfg]["partial_amount"]
            if normal > 0:
                ratios.append(partial / normal)
            else:
                ratios.append(0)  # Avoid division by zero

        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(x=np.arange(len(configs)), height=ratios, color="purple")

        # Annotate bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Configuration (Number of Institutions Allowing Partials)")
        ax.set_ylabel("Partial / Normal Settlement Ratio")
        ax.set_title("Ratio of Partial Settlement to Normal Settlement")
        ax.set_xticks(np.arange(len(configs)))
        ax.set_xticklabels([f"Config {cfg}" for cfg in configs])
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "settlement_type_ratio.png"), dpi=300)
        plt.close()
        print(f"[INFO] Saved settlement type ratio to {os.path.join(self.output_dir, 'settlement_type_ratio.png')}")

    def _plot_total_settled_stacked(self, data):
        """Plot total settled amount with stacked breakdown by settlement type."""
        configs = sorted(data.keys())
        normal_amounts = [data[cfg]["normal_amount"] for cfg in configs]
        partial_amounts = [data[cfg]["partial_amount"] for cfg in configs]
        intended_amounts = [data[cfg]["intended_amount"] for cfg in configs]

        # Convert to billions
        scale = 1e9
        normal_amounts_b = [amt / scale for amt in normal_amounts]
        partial_amounts_b = [amt / scale for amt in partial_amounts]
        intended_amounts_b = [amt / scale for amt in intended_amounts]

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(configs))
        width = 0.6

        # Plot bars
        bars1 = ax.bar(x, normal_amounts_b, width, label="Normal First-Try Settlement", color="steelblue")
        bars2 = ax.bar(x, partial_amounts_b, width, bottom=normal_amounts_b, label="Partial Settlement",
                       color="darkorange")

        # Plot intended amount line if different from actual settled
        if not np.array_equal(np.array(intended_amounts_b), np.array(normal_amounts_b) + np.array(partial_amounts_b)):
            ax.plot(x, intended_amounts_b, 'o--', color="red", label="Intended Amount", linewidth=2)

        # Annotate total (normal + partial)
        for i, (normal, partial) in enumerate(zip(normal_amounts_b, partial_amounts_b)):
            total = normal + partial
            ax.text(i, total + 0.1, f"{total:.2f}B", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Configuration (Number of Institutions Allowing Partials)")
        ax.set_ylabel("Amount (€ Billions)")
        ax.set_title("Total Settled Amount by Settlement Type")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Config {cfg}" for cfg in configs])
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "total_settled_stacked.png"), dpi=300)
        plt.close()
        print(f"[INFO] Saved total settled stacked to {os.path.join(self.output_dir, 'total_settled_stacked.png')}")

    def _plot_settlement_efficiency_comparison(self, data):
        """Compare settlement efficiency with and without partial settlement contribution."""
        configs = sorted(data.keys())

        normal_efficiency = []
        combined_efficiency = []
        partial_contribution = []

        for cfg in configs:
            intended = data[cfg]["intended_amount"]
            if intended > 0:
                normal_eff = data[cfg]["normal_amount"] / intended * 100
                combined_eff = (data[cfg]["normal_amount"] + data[cfg]["partial_amount"]) / intended * 100

                normal_efficiency.append(normal_eff)
                combined_efficiency.append(combined_eff)
                partial_contribution.append(combined_eff - normal_eff)
            else:
                normal_efficiency.append(0)
                combined_efficiency.append(0)
                partial_contribution.append(0)

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(configs))
        width = 0.35

        bars1 = ax.bar(x - width / 2, normal_efficiency, width,
                       label="Value Efficiency without Partial Settlement", color="lightblue")
        bars2 = ax.bar(x + width / 2, combined_efficiency, width,
                       label="Value Efficiency with Partial Settlement", color="green")

             # Annotate bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                    f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                    f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

            # Add annotation for the partial contribution if significant
            if partial_contribution[i] > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, height - 1.5,
                        f"+{partial_contribution[i]:.1f}%", ha="center", va="top",
                        fontsize=8, color="darkgreen", fontweight="bold")

        ax.set_xlabel("Configuration (Number of Institutions Allowing Partials)")
        ax.set_ylabel("Settlement Efficiency (%)")
        ax.set_title("Settlement Efficiency: Contribution of Partial Settlement")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Config {cfg}" for cfg in configs])
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "settlement_efficiency_comparison.png"), dpi=300)
        plt.close()
        print(
            f"[INFO] Saved settlement efficiency comparison to {os.path.join(self.output_dir, 'settlement_efficiency_comparison.png')}")