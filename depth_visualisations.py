import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
from collections import defaultdict
import glob


class SettlementAnalyzer:
    """
    Analyzes settlement data across multiple partial allowance configurations
    """

    def __init__(self, results_dir="./"):
        """
        Initialize with directory containing batch runner results

        Args:
            results_dir: Directory containing results and logs
        """
        self.results_dir = results_dir

        # Define status groupings
        self.status_groups = {
            "Settled on time": ["Settled on time"],
            "Settled late": ["Settled late"],
            "Cancelled": ["Cancelled due to timeout", "Cancelled due to partial settlement", "Cancelled due to error"],
            "In process": ["Matched", "Validated", "Pending", "Exists"]
        }

        # Define colors for each group
        self.group_colors = {
            "Settled on time": "green",
            "Settled late": "orange",
            "Cancelled": "red",
            "In process": "lightgray"
        }

        # Find all depth statistics files
        self.configs = {}
        self.config_runs = defaultdict(dict)  # Store runs for each configuration

        depth_stats_dir = r"C:\Users\matth\Documents\GitHub\SettlementSimulatorV3\depth_statistics"

        # Check if directory exists
        if not os.path.exists(depth_stats_dir):
            print(f"Directory does not exist: {depth_stats_dir}")
            return

        # Get all jsonocel files
        depth_files = glob.glob(os.path.join(depth_stats_dir, "*.jsonocel"))

        print(f"Found {len(depth_files)} .jsonocel files")

        if not depth_files:
            print(f"No statistics files found in {depth_stats_dir}")
            return

        # Group files by configuration
        for depth_file in depth_files:
            basename = os.path.basename(depth_file)
            parts = basename.split("_")

            if len(parts) >= 3 and parts[0] == "depth" and parts[1] == "statistics":
                config_name = parts[2]  # config1, config2, etc.
                run_name = parts[3].split(".")[0] if len(parts) > 3 else "run1"  # Extract run name, removing extension

                try:
                    with open(depth_file, 'r') as f:
                        run_data = json.load(f)
                        self.config_runs[config_name][run_name] = run_data
                        print(f"Loaded data for {config_name} {run_name}")
                except Exception as e:
                    print(f"Error loading {basename}: {e}")
            else:
                print(f"Skipping file with unexpected name format: {basename}")

        # Aggregate data for each configuration
        for config_name, runs in self.config_runs.items():
            if runs:
                print(f"Aggregating data for {config_name} from {len(runs)} runs")
                self.configs[config_name] = self._aggregate_runs(runs)

        print(f"Loaded {len(self.configs)} configuration(s)")

    def _aggregate_runs(self, runs):
        """
        Aggregate data from multiple runs of the same configuration

        Args:
            runs: Dictionary of run_name -> run_data

        Returns:
            Aggregated data dictionary
        """
        if not runs:
            return {}

        # Initialize aggregated data
        aggregated = {
            "depth_counts": defaultdict(float),
            "depth_status_counts": defaultdict(lambda: defaultdict(float))
        }

        # Collect all depths and statuses
        all_depths = set()
        all_statuses = set()

        for run_data in runs.values():
            # Collect depths
            if "depth_counts" in run_data:
                all_depths.update(run_data["depth_counts"].keys())

            # Collect statuses
            if "depth_status_counts" in run_data:
                for depth, status_data in run_data["depth_status_counts"].items():
                    all_depths.add(depth)
                    all_statuses.update(status_data.keys())

        # Aggregate depth counts
        for depth in all_depths:
            depth_counts = []
            for run_data in runs.values():
                depth_counts.append(run_data.get("depth_counts", {}).get(depth, 0))

            # Store average count
            aggregated["depth_counts"][depth] = sum(depth_counts) / len(runs)

        # Aggregate status counts
        for depth in all_depths:
            for status in all_statuses:
                status_counts = []
                for run_data in runs.values():
                    depth_status = run_data.get("depth_status_counts", {}).get(depth, {})
                    status_counts.append(depth_status.get(status, 0))

                # Store average count if there are any
                if any(status_counts):
                    aggregated["depth_status_counts"][depth][status] = sum(status_counts) / len(runs)

        return aggregated

    def analyze_single_config(self, config_name, output_dir="visualizations/"):
        """
        Generate visualizations for a single configuration

        Args:
            config_name: Name of the configuration
            output_dir: Directory to save visualizations
        """
        if config_name not in self.configs:
            print(f"Configuration {config_name} not found")
            return

        data = self.configs[config_name]
        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        # Generate depth distribution chart
        self._depth_distribution(data, config_name, os.path.join(config_dir, "depth_distribution.png"))

        # Generate status by depth chart
        self._status_by_depth(data, config_name, os.path.join(config_dir, "status_by_depth.png"))

        # Generate success rate chart
        self._success_rate_by_depth(data, config_name, os.path.join(config_dir, "success_rate.png"))

        print(f"Generated visualizations for {config_name} in {config_dir}")

    def _depth_distribution(self, data, config_name, output_file):
        """Generate depth distribution chart showing average across runs"""
        depth_counts = data["depth_counts"]
        depths = sorted([int(d) for d in depth_counts.keys()])
        counts = [depth_counts[str(d)] for d in depths]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(depths, counts, color='skyblue')

        # Add count labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{count:.1f}", ha='center', va='bottom')

        plt.xlabel('Instruction Depth')
        plt.ylabel('Average Number of Instructions')
        plt.title(f'Configuration {config_name}: Avg. Distribution of Instructions by Depth')
        plt.xticks(depths)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _status_by_depth(self, data, config_name, output_file):
        """Generate status by depth chart showing average across runs"""
        depth_status_counts = data["depth_status_counts"]
        depths = sorted([int(d) for d in depth_status_counts.keys()])

        # Group statuses for simplified view
        group_data = {}
        for group, statuses in self.status_groups.items():
            group_data[group] = []
            for depth in depths:
                # Sum counts for all statuses in this group
                count = sum(depth_status_counts.get(str(depth), {}).get(status, 0)
                            for status in statuses)
                group_data[group].append(count)

        plt.figure(figsize=(12, 8))
        bottom = [0] * len(depths)

        # Create stacked bars with colors from group_colors
        for group, counts in group_data.items():
            color = self.group_colors.get(group, 'gray')
            plt.bar(depths, counts, bottom=bottom, label=group, color=color)
            bottom = [b + c for b, c in zip(bottom, counts)]

        plt.xlabel('Instruction Depth')
        plt.ylabel('Average Number of Instructions')
        plt.title(f'Configuration {config_name}: Avg. Distribution of Outcomes by Depth')
        plt.xticks(depths)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _success_rate_by_depth(self, data, config_name, output_file):
        """Generate success rate chart showing average across runs"""
        depth_status_counts = data["depth_status_counts"]
        depths = sorted([int(d) for d in depth_status_counts.keys()])

        success_rates = []
        for depth in depths:
            statuses = depth_status_counts.get(str(depth), {})

            # Calculate total instructions at this depth
            total = sum(statuses.values())

            # Calculate successful settlements (on time + late)
            successful = sum(statuses.get(status, 0) for status in
                             self.status_groups["Settled on time"] + self.status_groups["Settled late"])

            # Calculate success rate
            success_rate = (successful / total * 100) if total > 0 else 0
            success_rates.append(success_rate)

        plt.figure(figsize=(10, 6))
        plt.plot(depths, success_rates, 'o-', linewidth=2, markersize=8, color='blue')

        # Add percentage labels at each point
        for i, rate in enumerate(success_rates):
            plt.annotate(f"{rate:.1f}%",
                         (depths[i], success_rates[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        plt.xlabel('Depth Level')
        plt.ylabel('Success Rate (%)')
        plt.title(f'Configuration {config_name}: Avg. Settlement Success Rate by Depth')
        plt.xticks(depths)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, max(success_rates) * 1.2 if success_rates else 100)  # Leave room for labels
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_configs(self, output_dir="comparisons/"):
        """Generate comparative visualizations across configurations"""
        if len(self.configs) < 2:
            print("Need at least 2 configurations for comparison")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Compare success rates across configurations
        self._compare_success_rates(os.path.join(output_dir, "success_rate_comparison.png"))

        # Compare depth distributions across configurations
        self._compare_depth_distributions(os.path.join(output_dir, "depth_distribution_comparison.png"))

        # Generate summary table
        self._generate_summary_table(os.path.join(output_dir, "summary.csv"))

        print(f"Generated comparison visualizations in {output_dir}")

    def _compare_success_rates(self, output_file):
        """Compare success rates across configurations"""
        plt.figure(figsize=(12, 8))

        for config_name, data in self.configs.items():
            depth_status = data["depth_status_counts"]
            depths = sorted([int(d) for d in depth_status.keys()])

            success_rates = []
            for depth in depths:
                statuses = depth_status.get(str(depth), {})
                total = sum(statuses.values())
                successful = sum(statuses.get(status, 0) for status in
                                 self.status_groups["Settled on time"] + self.status_groups["Settled late"])
                success_rate = (successful / total * 100) if total > 0 else 0
                success_rates.append(success_rate)

            # Plot this configuration's success rates
            plt.plot(depths, success_rates, 'o-', linewidth=2, markersize=6, label=config_name)

        plt.xlabel('Depth Level')
        plt.ylabel('Success Rate (%)')
        plt.title('Average Settlement Success Rate Comparison Across Configurations')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _compare_depth_distributions(self, output_file):
        """Compare depth distributions across configurations"""
        plt.figure(figsize=(12, 8))

        # Find the maximum depth across all configurations
        max_depth = 0
        for data in self.configs.values():
            depth_counts = data["depth_counts"]
            depths = [int(d) for d in depth_counts.keys()]
            if depths:
                max_depth = max(max_depth, max(depths))

        # Calculate bar width based on number of configurations
        bar_width = 0.8 / len(self.configs)
        depths = list(range(max_depth + 1))

        for i, (config_name, data) in enumerate(self.configs.items()):
            depth_counts = data["depth_counts"]

            # Get counts for each depth (0 if not present)
            counts = [float(depth_counts.get(str(d), 0)) for d in depths]

            # Calculate positions for this configuration's bars
            x_positions = [d + (i - len(self.configs) / 2 + 0.5) * bar_width for d in depths]

            # Plot this configuration's depth distribution
            plt.bar(x_positions, counts, width=bar_width, label=config_name)

        plt.xlabel('Depth Level')
        plt.ylabel('Average Number of Instructions')
        plt.title('Average Instruction Depth Distribution Comparison')
        plt.xticks(depths)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _generate_summary_table(self, output_file):
        """Generate summary statistics table for all configurations"""
        summary_data = []

        for config_name, data in self.configs.items():
            # Calculate overall success rate
            depth_status = data["depth_status_counts"]
            total_instructions = 0
            total_successful = 0

            for depth, statuses in depth_status.items():
                depth_total = sum(statuses.values())
                total_instructions += depth_total

                # Count successful instructions
                successful = sum(statuses.get(status, 0) for status in
                                 self.status_groups["Settled on time"] + self.status_groups["Settled late"])
                total_successful += successful

            # Calculate max depth reached
            max_depth = max([int(d) for d in data["depth_counts"].keys()]) if data["depth_counts"] else 0

            # Add to summary data
            summary_data.append({
                "Configuration": config_name,
                "Total Instructions": total_instructions,
                "Success Rate (%)": (total_successful / total_instructions * 100) if total_instructions else 0,
                "Max Depth": max_depth
            })

        # Create DataFrame and save to CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(output_file, index=False)
        print(f"Summary table saved to {output_file}")

    def analyze_all(self, output_base="settlement_analysis/"):
        """
        Analyze all configurations individually and comparatively

        Args:
            output_base: Base directory for all output
        """
        # Create output directory
        os.makedirs(output_base, exist_ok=True)

        # Analyze each configuration individually
        indiv_dir = os.path.join(output_base, "individual/")
        for config_name in self.configs.keys():
            self.analyze_single_config(config_name, indiv_dir)

        # Generate comparative visualizations
        self.compare_configs(os.path.join(output_base, "comparisons/"))

        # Generate RTP vs Batch visualizations
        self.analyze_rtp_vs_batch_from_logs(
            log_folder="simulatie_logs",
            output_dir=os.path.join(output_base, "rtp_vs_batch")
        )

        # Generate settlement lateness visualizations
        self.analyze_lateness_from_depth_stats(
            stats_folder="depth_statistics",
            output_dir=os.path.join(output_base, "lateness")
        )

        # self.analyze_lateness_hours(
        #     log_folder="simulatie_logs",
        #     output_dir=os.path.join(output_base, "lateness_hours")
        # )

        print(f"Analysis complete. Results in {output_base}")

    # Add these methods to your SettlementAnalyzer class

    def compare_settlement_times(self, output_file):
        """
        Compare the proportion of on-time vs late settlements across configurations
        """
        # Prepare data
        config_names = []
        on_time_percentages = []
        late_percentages = []

        for config_name, data in self.configs.items():
            depth_status = data["depth_status_counts"]

            total_settlements = 0
            on_time = 0
            late = 0

            # Sum across all depths
            for depth, statuses in depth_status.items():
                # Count on-time settlements
                on_time += sum(statuses.get(status, 0) for status in self.status_groups["Settled on time"])

                # Count late settlements
                late += sum(statuses.get(status, 0) for status in self.status_groups["Settled late"])

            total_settlements = on_time + late

            if total_settlements > 0:
                config_names.append(config_name)
                on_time_percentages.append((on_time / total_settlements) * 100)
                late_percentages.append((late / total_settlements) * 100)

        # Create stacked bar chart
        plt.figure(figsize=(12, 8))
        bar_width = 0.8

        # Create bars
        bars1 = plt.bar(config_names, on_time_percentages, bar_width, label='Settled on time',
                        color=self.group_colors["Settled on time"])
        bars2 = plt.bar(config_names, late_percentages, bar_width, bottom=on_time_percentages, label='Settled late',
                        color=self.group_colors["Settled late"])

        # Add percentage labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            plt.text(bar1.get_x() + bar1.get_width() / 2., height1 / 2, f'{height1:.1f}%',
                     ha='center', va='center', color='white', fontweight='bold')
            if height2 > 5:  # Only add label if segment is large enough
                plt.text(bar2.get_x() + bar2.get_width() / 2., height1 + height2 / 2, f'{height2:.1f}%',
                         ha='center', va='center', color='white', fontweight='bold')

        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Settlements')
        plt.title('Settlement Timing Comparison Across Configurations')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_depth_completion_rates(self, output_file):
        """
        Compare how settlement completion rates drop with increasing depth
        """
        plt.figure(figsize=(12, 8))

        # Find max depth across all configs
        max_depth = 0
        for data in self.configs.values():
            depths = [int(d) for d in data["depth_counts"].keys()]
            if depths:
                max_depth = max(max_depth, max(depths))

        completion_rates = {}

        # Calculate completion rate at each depth for each config
        for config_name, data in self.configs.items():
            depth_status = data["depth_status_counts"]
            rates = []

            # For each depth level
            for depth in range(max_depth + 1):
                str_depth = str(depth)
                if str_depth in depth_status:
                    statuses = depth_status[str_depth]
                    total = sum(statuses.values())

                    # Total successful (on time + late)
                    successful = sum(statuses.get(status, 0) for status in
                                     self.status_groups["Settled on time"] + self.status_groups["Settled late"])

                    # Calculate completion rate
                    rate = (successful / total * 100) if total > 0 else 0
                    rates.append(rate)
                else:
                    rates.append(0)  # No data for this depth

            completion_rates[config_name] = rates

        # Plot normalized completion rates (where depth 0 = 100%)
        for config_name, rates in completion_rates.items():
            if rates and rates[0] > 0:  # Ensure we have data and depth 0 has a non-zero rate
                # Normalize to depth 0
                normalized_rates = [rate / rates[0] * 100 for rate in rates]
                plt.plot(range(len(normalized_rates)), normalized_rates, 'o-', linewidth=2, markersize=6,
                         label=config_name)

        plt.xlabel('Depth Level')
        plt.ylabel('Completion Rate (% of Depth 0)')
        plt.title('Normalized Settlement Completion Rate by Depth')
        plt.xticks(range(max_depth + 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_status_distributions(self, output_file):
        """
        Compare the overall status distribution across configurations
        """
        # Prepare data
        config_names = []
        status_data = {status_group: [] for status_group in self.status_groups.keys()}

        for config_name, data in self.configs.items():
            config_names.append(config_name)
            depth_status = data["depth_status_counts"]

            # Count total instructions
            total_instructions = 0
            group_counts = {group: 0 for group in self.status_groups.keys()}

            for depth, statuses in depth_status.items():
                for group, status_list in self.status_groups.items():
                    count = sum(statuses.get(status, 0) for status in status_list)
                    group_counts[group] += count
                    total_instructions += count

            # Calculate percentage for each status group
            if total_instructions > 0:
                for group in self.status_groups.keys():
                    status_data[group].append((group_counts[group] / total_instructions) * 100)
            else:
                for group in self.status_groups.keys():
                    status_data[group].append(0)

        # Create stacked bar chart
        plt.figure(figsize=(12, 8))
        bar_width = 0.8

        # Plot bars
        bottom = np.zeros(len(config_names))
        for group, percentages in status_data.items():
            plt.bar(config_names, percentages, bar_width, bottom=bottom,
                    label=group, color=self.group_colors.get(group, 'gray'))

            # Add percentage labels if segment is large enough
            for i, height in enumerate(percentages):
                if height > 5:  # Only label if segment is visible enough
                    plt.text(i, bottom[i] + height / 2, f'{height:.1f}%',
                             ha='center', va='center', color='white', fontweight='bold')

            bottom += np.array(percentages)

        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Instructions')
        plt.title('Overall Status Distribution Across Configurations')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def create_heatmap_comparison(self, output_file):
        """
        Create a heatmap comparing success rates at each depth across configurations
        """
        # Find max depth across all configs
        max_depth = 0
        for data in self.configs.values():
            depths = [int(d) for d in data["depth_counts"].keys()]
            if depths:
                max_depth = max(max_depth, max(depths))

        # Prepare data for heatmap
        config_names = list(self.configs.keys())
        success_rates = np.zeros((len(config_names), max_depth + 1))

        for i, (config_name, data) in enumerate(self.configs.items()):
            depth_status = data["depth_status_counts"]

            for depth in range(max_depth + 1):
                str_depth = str(depth)
                if str_depth in depth_status:
                    statuses = depth_status[str_depth]
                    total = sum(statuses.values())

                    # Total successful (on time + late)
                    successful = sum(statuses.get(status, 0) for status in
                                     self.status_groups["Settled on time"] + self.status_groups["Settled late"])

                    # Calculate success rate
                    rate = (successful / total * 100) if total > 0 else 0
                    success_rates[i, depth] = rate

        # Create heatmap
        plt.figure(figsize=(max(10, max_depth * 0.8), len(config_names) * 0.8))
        ax = plt.gca()

        # Plot heatmap
        im = ax.imshow(success_rates, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Success Rate (%)')

        # Configure axes
        ax.set_xticks(np.arange(max_depth + 1))
        ax.set_yticks(np.arange(len(config_names)))
        ax.set_xticklabels(range(max_depth + 1))
        ax.set_yticklabels(config_names)

        # Add success rate text in each cell
        for i in range(len(config_names)):
            for j in range(max_depth + 1):
                rate = success_rates[i, j]
                color = "black" if 30 < rate < 70 else "white"  # Choose text color based on background
                ax.text(j, i, f"{rate:.1f}%", ha="center", va="center", color=color)

        plt.xlabel('Depth Level')
        plt.title('Settlement Success Rate Heatmap')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def compare_total_instructions(self, output_file):
        """
        Compare the total number of instructions processed in each configuration
        """
        config_names = []
        total_counts = []

        for config_name, data in self.configs.items():
            config_names.append(config_name)

            # Calculate total instructions
            total = sum(float(count) for count in data["depth_counts"].values())
            total_counts.append(total)

        # Sort by total instructions
        sorted_indices = np.argsort(total_counts)[::-1]  # Sort descending
        sorted_configs = [config_names[i] for i in sorted_indices]
        sorted_counts = [total_counts[i] for i in sorted_indices]

        # Create bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.bar(sorted_configs, sorted_counts, color='steelblue')

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.1f}', ha='center', va='bottom')

        plt.xlabel('Configuration')
        plt.ylabel('Average Number of Instructions')
        plt.title('Total Instructions Processed per Configuration')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    # Add these calls to the compare_configs method
    def compare_configs(self, output_dir="comparisons/"):
        """Generate comparative visualizations across configurations"""
        if len(self.configs) < 2:
            print("Need at least 2 configurations for comparison")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Original comparisons
        self._compare_success_rates(os.path.join(output_dir, "success_rate_comparison.png"))
        self._compare_depth_distributions(os.path.join(output_dir, "depth_distribution_comparison.png"))
        self._generate_summary_table(os.path.join(output_dir, "summary.csv"))

        # New additional comparisons
        self.compare_settlement_times(os.path.join(output_dir, "settlement_times_comparison.png"))
        self.compare_depth_completion_rates(os.path.join(output_dir, "depth_completion_rates.png"))
        self.compare_status_distributions(os.path.join(output_dir, "status_distribution_comparison.png"))
        self.create_heatmap_comparison(os.path.join(output_dir, "success_rate_heatmap.png"))
        self.compare_total_instructions(os.path.join(output_dir, "total_instructions_comparison.png"))

        print(f"Generated comparison visualizations in {output_dir}")

    def analyze_rtp_vs_batch_from_logs(self, log_folder="simulatie_logs/",
                                       output_dir="settlement_analysis/rtp_vs_batch/"):
        """
        Analyze settlement timing patterns from the CSV logs to determine RTP vs Batch processing

        Args:
            log_folder: Directory containing simulation logs in CSV format
            output_dir: Directory to save visualizations
        """
        import os
        import pandas as pd
        import csv
        from datetime import datetime, time
        from collections import defaultdict

        os.makedirs(output_dir, exist_ok=True)

        # Dictionary to store data for each configuration
        config_data = defaultdict(lambda: {"rtp": 0, "batch": 0})

        # Find all CSV log files
        if not os.path.exists(log_folder):
            print(f"Log directory '{log_folder}' does not exist!")
            print(f"Current working directory: {os.getcwd()}")
            return

        log_files = [f for f in os.listdir(log_folder) if f.endswith(".csv") and f.startswith("log_config")]

        if not log_files:
            print(f"No CSV log files found in {log_folder}")
            print(f"Files in directory: {os.listdir(log_folder)}")
            return

        print(f"Found {len(log_files)} CSV log files for analysis")

        # Process each log file
        settlements_found = False

        for log_file in log_files:
            # Extract configuration number from filename (e.g., log_config6_run8.csv)
            try:
                import re
                config_match = re.search(r'config(\d+)', log_file)
                if config_match:
                    config_num = int(config_match.group(1))
                    config_name = f"Config {config_num}"
                else:
                    config_name = log_file.split('.')[0]  # Fallback
            except Exception as e:
                print(f"Error extracting config number from {log_file}: {e}")
                config_name = log_file.split('.')[0]

            try:
                # Open and process the CSV file
                with open(os.path.join(log_folder, log_file), 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)

                    # Check if we have the required columns
                    if log_file == log_files[0]:
                        try:
                            first_row = next(reader)
                            print(f"CSV columns: {list(first_row.keys())}")
                            f.seek(0)  # Reset to beginning
                            reader = csv.DictReader(f)  # Recreate reader
                        except StopIteration:
                            print(f"Empty CSV file: {log_file}")
                            continue

                    for row in reader:
                        # Look for settlement events
                        activity = row.get('activity', '')
                        if 'Settled' in activity:
                            settlements_found = True

                            # Extract timestamp
                            timestamp_str = row.get('timestamp', '')
                            if timestamp_str:
                                try:
                                    timestamp = datetime.fromisoformat(
                                        timestamp_str.replace('Z', '+00:00').replace('T', ' '))
                                    event_time = timestamp.time()

                                    # Determine if RTP or Batch based on time of day
                                    if time(1, 30) <= event_time <= time(19, 30):
                                        config_data[config_name]["rtp"] += 1
                                    else:
                                        config_data[config_name]["batch"] += 1
                                except Exception as e:
                                    print(f"Error parsing timestamp '{timestamp_str}': {e}")

            except Exception as e:
                print(f"Error processing file {log_file}: {e}")

        if not settlements_found:
            print("No settlement events found in the log files. Check CSV format.")
            return

        # Create visualizations if we have data
        if not config_data or all(sum(data.values()) == 0 for data in config_data.values()):
            print("No RTP vs Batch data could be extracted from logs.")
            return

        print("Found settlement data for configurations:")
        for config, data in config_data.items():
            rtp_count = data["rtp"]
            batch_count = data["batch"]
            print(f"  {config}: {rtp_count} RTP settlements, {batch_count} Batch settlements")

        # Sort configurations by their number
        try:
            configs = sorted(config_data.keys(),
                             key=lambda x: int(x.split()[1]) if len(x.split()) > 1 and x.split()[1].isdigit() else 0)
        except (IndexError, ValueError):
            configs = sorted(config_data.keys())

        # Extract data for plotting
        rtp_counts = [config_data[config]["rtp"] for config in configs]
        batch_counts = [config_data[config]["batch"] for config in configs]
        total_counts = [rtp + batch for rtp, batch in zip(rtp_counts, batch_counts)]

        # 1. Create percentage stacked bar chart
        plt.figure(figsize=(12, 8))

        # Calculate percentages
        rtp_percentages = [rtp / total * 100 if total > 0 else 0 for rtp, total in zip(rtp_counts, total_counts)]
        batch_percentages = [batch / total * 100 if total > 0 else 0 for batch, total in
                             zip(batch_counts, total_counts)]

        # Create stacked bar chart
        plt.bar(configs, rtp_percentages, label='Real-Time Processing', color='skyblue')
        plt.bar(configs, batch_percentages, bottom=rtp_percentages, label='Batch Processing', color='salmon')

        # Add percentage labels
        for i, (rtp_pct, batch_pct) in enumerate(zip(rtp_percentages, batch_percentages)):
            if rtp_pct > 5:  # Only show label if segment is large enough
                plt.text(i, rtp_pct / 2, f'{rtp_pct:.1f}%', ha='center', va='center', fontweight='bold')
            if batch_pct > 5:
                plt.text(i, rtp_pct + batch_pct / 2, f'{batch_pct:.1f}%', ha='center', va='center', fontweight='bold')

        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Settled Instructions')
        plt.title('RTP vs Batch Processing Settlements by Configuration')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rtp_vs_batch_percentage.png"), dpi=300)
        plt.close()

        # 2. Create absolute count bar chart
        plt.figure(figsize=(12, 8))
        width = 0.35
        x = range(len(configs))

        # Plot bars
        plt.bar([i - width / 2 for i in x], rtp_counts, width, label='Real-Time Processing', color='skyblue')
        plt.bar([i + width / 2 for i in x], batch_counts, width, label='Batch Processing', color='salmon')

        # Add count labels
        for i, count in enumerate(rtp_counts):
            plt.text(i - width / 2, count, str(count), ha='center', va='bottom')
        for i, count in enumerate(batch_counts):
            plt.text(i + width / 2, count, str(count), ha='center', va='bottom')

        plt.xlabel('Configuration')
        plt.ylabel('Number of Settled Instructions')
        plt.title('RTP vs Batch Processing Settlements by Configuration')
        plt.xticks(x, configs)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rtp_vs_batch_absolute.png"), dpi=300)
        plt.close()

        # 3. Create line chart for trends
        plt.figure(figsize=(12, 8))

        # Plot total line
        plt.plot(configs, total_counts, 'o-', color='purple', linewidth=2, markersize=8, label='Total Settlements')

        # Plot RTP and Batch lines
        plt.plot(configs, rtp_counts, 's--', color='skyblue', linewidth=2, markersize=6, label='Real-Time Processing')
        plt.plot(configs, batch_counts, '^--', color='salmon', linewidth=2, markersize=6, label='Batch Processing')

        # Add data labels
        for i, count in enumerate(total_counts):
            plt.text(i, count, str(count), ha='center', va='bottom')

        plt.xlabel('Configuration')
        plt.ylabel('Number of Settled Instructions')
        plt.title('Settlement Trends by Processing Type Across Configurations')
        plt.xticks(range(len(configs)), configs)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rtp_vs_batch_trend.png"), dpi=300)
        plt.close()

        print(f"RTP vs Batch analysis visualizations created in {output_dir}")


    def analyze_lateness_from_depth_stats(self, stats_folder="depth_statistics/",
                                          output_dir="settlement_analysis/lateness/",
                                          measurement_csv="New_measurement.csv"):
        """
        Analyze settlement lateness patterns using depth statistics files

        Args:
            stats_folder: Directory containing depth statistics files
            output_dir: Directory to save visualizations
            measurement_csv: Path to New_measurement.csv for settled amount data
        """
        import os
        import json
        import numpy as np
        import pandas as pd
        from collections import defaultdict

        os.makedirs(output_dir, exist_ok=True)

        print("Analyzing settlement lateness patterns from depth statistics...")

        # Find all depth statistics files
        if not os.path.exists(stats_folder):
            print(f"Statistics directory '{stats_folder}' does not exist!")
            print(f"Current working directory: {os.getcwd()}")
            return

        stats_files = [f for f in os.listdir(stats_folder) if f.endswith(".jsonocel") and f.startswith("depth_statistics")]

        if not stats_files:
            print(f"No depth statistics files found in {stats_folder}")
            print(f"Files in directory: {os.listdir(stats_folder)}")
            return

        print(f"Found {len(stats_files)} depth statistics files for analysis")

        # Load settled amount data from CSV if it exists
        amount_data_available = False
        amount_by_config = {}

        if os.path.exists(measurement_csv):
            try:
                df = pd.read_csv(measurement_csv)
                print(f"Loaded measurement data from {measurement_csv}: {df.shape[0]} rows")
                print(f"Columns: {df.columns.tolist()}")

                # Check if we have settled_amount column
                if 'settled_amount' in df.columns and 'Partial' in df.columns:
                    amount_data_available = True

                    # Group by Partial configuration and calculate average settled amount
                    for _, group in df.groupby('Partial'):
                        config_str = group['Partial'].iloc[0]

                        # Extract config number from the string
                        import re
                        true_count = config_str.count("True")
                        config_name = f"Config {true_count}"

                        # Calculate average settled amount for this config
                        avg_amount = group['settled_amount'].mean()
                        amount_by_config[config_name] = avg_amount

                    print(f"Loaded settled amount data for {len(amount_by_config)} configurations")
                else:
                    print(f"CSV file does not contain required columns. Found: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error loading measurement CSV: {e}")
        else:
            print(f"Measurement CSV file not found: {measurement_csv}")

        # Prepare data structures for analysis
        config_data = defaultdict(lambda: {"on_time": [], "late": [], "late_ratio": []})
        depth_late_ratios = defaultdict(list)  # For tracking late settlement ratios by depth

        # Process each statistics file
        for stats_file in stats_files:
            # Extract configuration number
            try:
                import re
                config_match = re.search(r'config(\d+)', stats_file)
                if config_match:
                    config_num = int(config_match.group(1))
                    config_name = f"Config {config_num}"
                else:
                    config_name = stats_file.split('.')[0]
            except Exception as e:
                print(f"Error extracting config number from {stats_file}: {e}")
                config_name = stats_file.split('.')[0]

            try:
                # Load the depth statistics JSON file
                with open(os.path.join(stats_folder, stats_file), 'r') as f:
                    stats = json.load(f)

                # Extract settlement pattern data from the statistics
                if "depth_status_counts" in stats:
                    depth_status = stats["depth_status_counts"]

                    total_on_time = 0
                    total_late = 0

                    # Process each depth level
                    for depth, statuses in depth_status.items():
                        on_time = statuses.get("Settled on time", 0)
                        late = statuses.get("Settled late", 0)
                        total_on_time += on_time
                        total_late += late

                        # Calculate late ratio for this depth
                        total_settled = on_time + late
                        late_ratio = late / total_settled if total_settled > 0 else 0
                        depth_late_ratios[int(depth)].append(late_ratio)

                    # Collect data for this configuration
                    config_data[config_name]["on_time"].append(total_on_time)
                    config_data[config_name]["late"].append(total_late)
                    total_settled = total_on_time + total_late
                    late_ratio = total_late / total_settled if total_settled > 0 else 0
                    config_data[config_name]["late_ratio"].append(late_ratio)

                else:
                    print(f"No depth_status_counts found in {stats_file}")

            except Exception as e:
                print(f"Error processing file {stats_file}: {e}")

        # Aggregate data across runs for each configuration
        aggregate_data = {}
        for config, data in config_data.items():
            on_time_avg = np.mean(data["on_time"]) if data["on_time"] else 0
            late_avg = np.mean(data["late"]) if data["late"] else 0
            late_ratio_avg = np.mean(data["late_ratio"]) if data["late_ratio"] else 0

            aggregate_data[config] = {
                "on_time": on_time_avg,
                "late": late_avg,
                "late_ratio": late_ratio_avg
            }

        # Prepare data for plotting
        try:
            configs = sorted(aggregate_data.keys(),
                             key=lambda x: int(x.split()[1]) if len(x.split()) > 1 and x.split()[1].isdigit() else 0)
        except (IndexError, ValueError):
            configs = sorted(aggregate_data.keys())

        on_time_counts = [aggregate_data[config]["on_time"] for config in configs]
        late_counts = [aggregate_data[config]["late"] for config in configs]
        late_ratios = [aggregate_data[config]["late_ratio"] * 100 for config in configs]  # Convert to percentage

        # 1. Create stacked bar chart of on-time vs. late settlements
        plt.figure(figsize=(12, 8))

        # Plot bars
        plt.bar(configs, on_time_counts, label='Settled On Time', color='green')
        plt.bar(configs, late_counts, bottom=on_time_counts, label='Settled Late', color='orange')

        # Add count labels
        for i, (on_time, late) in enumerate(zip(on_time_counts, late_counts)):
            # Label for on-time
            plt.text(i, on_time / 2, f'{int(on_time)}', ha='center', va='center', fontweight='bold')
            # Label for late if large enough
            if late > max(on_time_counts) * 0.05:  # Only label if segment is at least 5% of the max
                plt.text(i, on_time + late / 2, f'{int(late)}', ha='center', va='center', fontweight='bold')

        plt.xlabel('Configuration')
        plt.ylabel('Number of Settlements')
        plt.title('On-Time vs Late Settlements by Configuration')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ontime_vs_late_counts.png"), dpi=300)
        plt.close()

        # 2. Create a bar chart of late settlement percentages
        plt.figure(figsize=(12, 8))

        # Plot bars with gradient colors based on ratio value
        bars = plt.bar(configs, late_ratios, color=plt.cm.YlOrRd(np.array(late_ratios) / 100))

        # Add percentage labels
        for i, ratio in enumerate(late_ratios):
            plt.text(i, ratio + 1, f'{ratio:.1f}%', ha='center', va='bottom')

        plt.xlabel('Configuration')
        plt.ylabel('Percentage of Settlements that were Late')
        plt.title('Late Settlement Percentage by Configuration')
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Add trend line
        z = np.polyfit(range(len(configs)), late_ratios, 1)
        p = np.poly1d(z)
        plt.plot(range(len(configs)), p(range(len(configs))), "r--", alpha=0.7)

        # Add trend indicator
        slope = z[0]
        if abs(slope) > 0.1:  # Only add annotation if trend is significant
            trend = "Increasing" if slope > 0 else "Decreasing"
            plt.annotate(f"{trend} trend", xy=(len(configs) / 2, max(late_ratios) / 2),
                         xytext=(len(configs) / 2, max(late_ratios) * 0.8),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "late_settlement_percentage.png"), dpi=300)
        plt.close()

        # 3. Create a line chart showing how lateness varies by depth
        max_depth = max(depth_late_ratios.keys())

        plt.figure(figsize=(14, 8))

        # Calculate average late ratio by depth
        depths = sorted(depth_late_ratios.keys())
        avg_late_ratios = [np.mean(depth_late_ratios[depth]) * 100 for depth in depths]

        # Plot the line
        plt.plot(depths, avg_late_ratios, 'o-', linewidth=2, markersize=8, color='purple')

        # Add percentage labels
        for i, ratio in enumerate(avg_late_ratios):
            plt.text(depths[i], ratio + 2, f'{ratio:.1f}%', ha='center', va='bottom')

        plt.xlabel('Instruction Depth')
        plt.ylabel('Late Settlement Percentage')
        plt.title('Late Settlement Percentage by Instruction Depth')
        plt.xticks(depths)
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lateness_by_depth.png"), dpi=300)
        plt.close()

        # 4. Create a heatmap showing how lateness varies by depth and configuration
        # Prepare data by configuration and depth
        depth_config_data = defaultdict(lambda: defaultdict(float))

        # Process each statistics file again for this visualization
        for stats_file in stats_files:
            try:
                # Extract configuration number
                config_match = re.search(r'config(\d+)', stats_file)
                if config_match:
                    config_num = int(config_match.group(1))
                    config_name = f"Config {config_num}"
                else:
                    continue  # Skip if config can't be determined

                # Load the depth statistics
                with open(os.path.join(stats_folder, stats_file), 'r') as f:
                    stats = json.load(f)

                if "depth_status_counts" in stats:
                    depth_status = stats["depth_status_counts"]

                    # Process each depth level
                    for depth_str, statuses in depth_status.items():
                        depth = int(depth_str)
                        on_time = statuses.get("Settled on time", 0)
                        late = statuses.get("Settled late", 0)

                        # Calculate late ratio for this depth in this config
                        total_settled = on_time + late
                        if total_settled > 0:
                            # Use a weighted average if we already have data
                            if config_name in depth_config_data[depth]:
                                existing_ratio = depth_config_data[depth][config_name]
                                # Simple averaging across runs
                                depth_config_data[depth][config_name] = (existing_ratio + (late / total_settled)) / 2
                            else:
                                depth_config_data[depth][config_name] = late / total_settled
            except Exception as e:
                print(f"Error processing file {stats_file} for heatmap: {e}")

        # Create heatmap data matrix
        depths_for_heatmap = sorted(depth_config_data.keys())
        configs_for_heatmap = sorted(configs, key=lambda x: int(x.split()[1]) if len(x.split()) > 1 and x.split()[
            1].isdigit() else 0)

        # For depths that have data in at least one configuration
        if depths_for_heatmap and configs_for_heatmap:
            heatmap_data = np.zeros((len(depths_for_heatmap), len(configs_for_heatmap)))

            for i, depth in enumerate(depths_for_heatmap):
                for j, config in enumerate(configs_for_heatmap):
                    heatmap_data[i, j] = depth_config_data[depth].get(config, 0) * 100  # Convert to percentage

            plt.figure(figsize=(14, 10))

            # Create heatmap with custom colormap (green to red gradient)
            cmap = plt.cm.get_cmap('RdYlGn_r')

            # Set aspect to ensure correct display of all depths/configs
            if len(depths_for_heatmap) > 0 and len(configs_for_heatmap) > 0:
                aspect = max(0.1, min(5, len(configs_for_heatmap) / len(depths_for_heatmap)))
            else:
                aspect = 'auto'

            im = plt.imshow(heatmap_data, cmap=cmap, aspect=aspect)

            # Configure axes
            plt.colorbar(im, label='Late Settlement Percentage (%)')
            plt.yticks(range(len(depths_for_heatmap)), depths_for_heatmap)
            plt.xticks(range(len(configs_for_heatmap)), configs_for_heatmap, rotation=45, ha='right')
            plt.ylabel('Instruction Depth')
            plt.xlabel('Configuration')
            plt.title('Late Settlement Percentage by Depth and Configuration')

            # Add percentage values in cells
            for i in range(len(depths_for_heatmap)):
                for j in range(len(configs_for_heatmap)):
                    if heatmap_data[i, j] > 0:  # Only add text for non-zero values
                        text_color = 'white' if heatmap_data[i, j] > 50 else 'black'
                        plt.text(j, i, f"{heatmap_data[i, j]:.1f}%",
                                 ha="center", va="center", color=text_color)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "lateness_depth_config_heatmap.png"), dpi=300)
            plt.close()

        # 5. Add visualizations for settled amounts if the data is available
        if amount_data_available and amount_by_config:
            # Calculate estimated on-time and late amounts
            on_time_amounts = []
            late_amounts = []

            for config in configs:
                if config in amount_by_config:
                    total_amount = amount_by_config[config]
                    late_ratio = aggregate_data[config]["late_ratio"]

                    # Estimate the split between on-time and late amounts
                    on_time_amount = total_amount * (1 - late_ratio)
                    late_amount = total_amount * late_ratio

                    on_time_amounts.append(on_time_amount)
                    late_amounts.append(late_amount)
                else:
                    on_time_amounts.append(0)
                    late_amounts.append(0)

            # 5a. Create stacked bar chart of on-time vs. late settlement amounts
            plt.figure(figsize=(12, 8))

            # Plot bars
            plt.bar(configs, on_time_amounts, label='Settled On Time', color='green')
            plt.bar(configs, late_amounts, bottom=on_time_amounts, label='Settled Late', color='orange')

            # Add amount labels with appropriate formatting based on magnitude
            for i, (on_time, late) in enumerate(zip(on_time_amounts, late_amounts)):
                # Format numbers based on size
                def format_value(val):
                    if val >= 1e12:
                        return f'{val / 1e12:.1f}T'
                    elif val >= 1e9:
                        return f'{val / 1e9:.1f}B'
                    elif val >= 1e6:
                        return f'{val / 1e6:.1f}M'
                    else:
                        return f'{int(val)}'

                # Label for on-time
                plt.text(i, on_time / 2, format_value(on_time), ha='center', va='center', fontweight='bold')
                # Label for late if large enough
                if late > max(on_time_amounts) * 0.05:  # Only label if segment is at least 5% of the max
                    plt.text(i, on_time + late / 2, format_value(late), ha='center', va='center', fontweight='bold')

            plt.xlabel('Configuration')
            plt.ylabel('Settlement Amount')
            plt.title('On-Time vs Late Settlement Amounts by Configuration')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.3)

            # Format y-axis to show abbreviated large numbers
            def y_fmt(x, pos):
                if x >= 1e12:
                    return f'{x / 1e12:.1f}T'
                elif x >= 1e9:
                    return f'{x / 1e9:.1f}B'
                elif x >= 1e6:
                    return f'{x / 1e6:.1f}M'
                else:
                    return f'{x:.0f}'

            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(y_fmt))

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "ontime_vs_late_amounts.png"), dpi=300)
            plt.close()

            # 5b. Create line chart showing total settled amount by configuration
            plt.figure(figsize=(12, 8))

            total_amounts = [on_time + late for on_time, late in zip(on_time_amounts, late_amounts)]

            plt.plot(configs, total_amounts, 'o-', linewidth=2, markersize=8, color='blue', label='Total Amount')
            plt.plot(configs, on_time_amounts, 's--', linewidth=2, markersize=6, color='green', label='On-Time Amount')
            plt.plot(configs, late_amounts, '^--', linewidth=2, markersize=6, color='orange', label='Late Amount')

            # Add labels for total amounts
            for i, amount in enumerate(total_amounts):
                plt.text(i, amount, y_fmt(amount, 0), ha='center', va='bottom')

            plt.xlabel('Configuration')
            plt.ylabel('Settlement Amount')
            plt.title('Settlement Amount Trends by Configuration')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)

            # Format y-axis
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(y_fmt))

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "settlement_amount_trends.png"), dpi=300)
            plt.close()

        print(f"Settlement lateness visualizations created in {output_dir}")

    # def analyze_lateness_hours(self, log_folder="simulation_logs/", output_dir="settlement_analysis/lateness_hours/"):
    #     """
    #     Analyze and visualize settlement lateness hours from logs
    #
    #     This method processes logs with lateness_hours attribute to create
    #     visualizations of how many hours settlements are late across configurations.
    #
    #     Args:
    #         log_folder: Directory containing logs with lateness_hours data
    #         output_dir: Directory to save visualizations
    #     """
    #     import os
    #     import json
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from collections import defaultdict
    #     import re
    #     from datetime import datetime, timedelta
    #
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     print("Analyzing lateness hours from logs...")
    #
    #     # Find all JSONOCEL files
    #     if not os.path.exists(log_folder):
    #         print(f"Log directory '{log_folder}' does not exist!")
    #         print(f"Current working directory: {os.getcwd()}")
    #         return
    #
    #     log_files = [f for f in os.listdir(log_folder) if f.endswith(".jsonocel") and "simulation" in f]
    #
    #     if not log_files:
    #         print(f"No simulation log files found in {log_folder}")
    #         print(f"Files in directory: {os.listdir(log_folder)}")
    #         return
    #
    #     print(f"Found {len(log_files)} log files for analysis")
    #
    #     # Prepare data structures
    #     config_lateness = defaultdict(list)  # lateness hours by configuration
    #     lateness_by_depth = defaultdict(list)  # lateness hours by depth level
    #     all_lateness_hours = []  # all lateness hours across all configurations
    #
    #     # Track if we found any lateness data
    #     lateness_data_found = False
    #
    #     # Process each log file
    #     for log_file in log_files:
    #         # Extract configuration number
    #         config_match = re.search(r'config(\d+)', log_file)
    #         if config_match:
    #             config_num = int(config_match.group(1))
    #             config_name = f"Config {config_num}"
    #         else:
    #             print(f"Could not extract configuration number from {log_file}")
    #             continue
    #
    #         try:
    #             # Load the log file
    #             with open(os.path.join(log_folder, log_file), 'r') as f:
    #                 log_data = json.load(f)
    #
    #             # Handle different JSON structures
    #             if isinstance(log_data, dict):
    #                 # Check for different event formats
    #                 if "ocel:events" in log_data:
    #                     # Classic OCEL format
    #                     event_key = "ocel:events"
    #                     attr_key = "ocel:attributes"
    #                     activity_key = "ocel:activity"
    #
    #                     # Process events (dictionary format)
    #                     events = log_data.get(event_key, {})
    #                     for event_id, event in events.items():
    #                         # Check if this is a "Settled Late" event
    #                         if event.get(activity_key) == "Settled Late":
    #                             attributes = event.get(attr_key, {})
    #
    #                             # Try to get lateness_hours if available
    #                             if "lateness_hours" in attributes:
    #                                 lateness_hours = float(attributes["lateness_hours"])
    #                                 lateness_data_found = True
    #
    #                                 # Collect lateness data
    #                                 config_lateness[config_name].append(lateness_hours)
    #                                 all_lateness_hours.append(lateness_hours)
    #
    #                                 # If depth information is available, collect by depth
    #                                 if "depth" in attributes:
    #                                     depth = int(attributes["depth"])
    #                                     lateness_by_depth[depth].append(lateness_hours)
    #
    #                 elif "events" in log_data:
    #                     # Alternative format with events as an array
    #                     events = log_data.get("events", [])
    #
    #                     for event in events:
    #                         # Check if this is a "Settled Late" event
    #                         event_type = event.get("type")
    #                         if event_type == "Settled Late" or event_type == "transaction_settled_late":
    #                             # Try to get attributes
    #                             attributes = []
    #                             for attr in event.get("attributes", []):
    #                                 if attr["name"] == "lateness_hours" and "value" in attr:
    #                                     try:
    #                                         lateness_hours = float(attr["value"])
    #                                         lateness_data_found = True
    #
    #                                         # Collect lateness data
    #                                         config_lateness[config_name].append(lateness_hours)
    #                                         all_lateness_hours.append(lateness_hours)
    #                                     except (ValueError, TypeError):
    #                                         pass
    #
    #                                 elif attr["name"] == "depth" and "value" in attr:
    #                                     try:
    #                                         depth = int(attr["value"])
    #
    #                                         # If we already found lateness_hours, add to depth data
    #                                         if lateness_data_found and all_lateness_hours:
    #                                             lateness_by_depth[depth].append(all_lateness_hours[-1])
    #                                     except (ValueError, TypeError):
    #                                         pass
    #
    #                             # If lateness_hours not found in attributes but we have timestamp
    #                             # Calculate approximate lateness from timestamp (fallback)
    #                             if not lateness_data_found and "time" in event:
    #                                 try:
    #                                     # Parse timestamp
    #                                     timestamp = event["time"]
    #                                     event_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    #
    #                                     # Use a fixed lateness of 1 hour as fallback
    #                                     lateness_hours = 1.0
    #                                     lateness_data_found = True
    #
    #                                     # Collect lateness data
    #                                     config_lateness[config_name].append(lateness_hours)
    #                                     all_lateness_hours.append(lateness_hours)
    #
    #                                     # Try to get depth from relationships
    #                                     for rel in event.get("relationships", []):
    #                                         if "objectId" in rel:
    #                                             # Extract depth from object ID if format allows
    #                                             obj_id = rel["objectId"]
    #                                             depth_match = re.search(r'_(\d+)_', obj_id)
    #                                             if depth_match:
    #                                                 depth = int(depth_match.group(1))
    #                                                 lateness_by_depth[depth].append(lateness_hours)
    #                                                 break
    #                                 except Exception as time_error:
    #                                     pass
    #                 else:
    #                     print(f"Unknown dictionary format in {log_file}")
    #
    #             else:
    #                 print(f"Unexpected data type in {log_file}: {type(log_data).__name__}")
    #
    #         except Exception as e:
    #             print(f"Error processing file {log_file}: {e}")
    #
    #     if not lateness_data_found:
    #         print("No lateness_hours data found in logs. Check your logging implementation.")
    #         return
    #
    #     print(f"Collected lateness data for {len(config_lateness)} configurations")
    #     print(f"Total number of late settlements: {len(all_lateness_hours)}")
    #
    #     # Sort configurations by number
    #     configs = sorted(config_lateness.keys(), key=lambda x: int(x.split()[1]))
    #
    #     # Calculate average lateness for each configuration
    #     avg_lateness = [np.mean(config_lateness[config]) for config in configs]
    #     median_lateness = [np.median(config_lateness[config]) for config in configs]
    #     max_lateness = [np.max(config_lateness[config]) for config in configs]
    #
    #     # 1. Create boxplot of lateness hours by configuration
    #     plt.figure(figsize=(14, 8))
    #
    #     # Prepare data for boxplot
    #     box_data = [config_lateness[config] for config in configs]
    #
    #     # Create boxplot
    #     bp = plt.boxplot(box_data, labels=configs, patch_artist=True)
    #
    #     # Customize box colors
    #     for box in bp['boxes']:
    #         box.set(facecolor='lightblue', alpha=0.8)
    #
    #     # Add jittered points to show distribution
    #     for i, data in enumerate(box_data):
    #         # Limit points to avoid overcrowding
    #         if len(data) > 100:
    #             import random
    #             data = random.sample(data, 100)
    #
    #         x = np.random.normal(i + 1, 0.08, size=len(data))
    #         plt.scatter(x, data, alpha=0.5, s=10, color='navy')
    #
    #     plt.xlabel('Configuration')
    #     plt.ylabel('Hours Late')
    #     plt.title('Distribution of Settlement Lateness Hours by Configuration')
    #     plt.grid(axis='y', linestyle='--', alpha=0.3)
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, "lateness_hours_boxplot.png"), dpi=300)
    #     plt.close()
    #
    #     # 2. Create histogram of lateness hours
    #     plt.figure(figsize=(12, 8))
    #
    #     # Determine appropriate number of bins based on data range
    #     max_hours = max(all_lateness_hours)
    #     if max_hours < 24:
    #         # If all settlements are less than a day late, use hourly bins
    #         bins = np.linspace(0, max_hours * 1.1, min(24, int(max_hours) + 1))
    #     else:
    #         # If settlements are very late, use more bins
    #         bins = np.linspace(0, max_hours * 1.1, 30)
    #
    #     plt.hist(all_lateness_hours, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    #     plt.xlabel('Hours Late')
    #     plt.ylabel('Frequency')
    #     plt.title('Overall Distribution of Settlement Lateness Hours')
    #     plt.grid(axis='y', linestyle='--', alpha=0.3)
    #
    #     # Add statistics
    #     mean_late = np.mean(all_lateness_hours)
    #     median_late = np.median(all_lateness_hours)
    #
    #     plt.axvline(x=mean_late, color='red', linestyle='--', label=f'Mean: {mean_late:.2f} hours')
    #     plt.axvline(x=median_late, color='green', linestyle='--', label=f'Median: {median_late:.2f} hours')
    #     plt.legend()
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, "lateness_hours_histogram.png"), dpi=300)
    #     plt.close()
    #
    #     # 3. Create bar chart comparing average, median, and max lateness across configurations
    #     plt.figure(figsize=(14, 8))
    #
    #     # Set width of bars
    #     bar_width = 0.25
    #     x = np.arange(len(configs))
    #
    #     # Create bars
    #     plt.bar(x - bar_width, avg_lateness, width=bar_width, label='Average', color='skyblue')
    #     plt.bar(x, median_lateness, width=bar_width, label='Median', color='forestgreen')
    #     plt.bar(x + bar_width, max_lateness, width=bar_width, label='Maximum', color='salmon')
    #
    #     # Add value labels
    #     def add_labels(positions, values):
    #         for pos, value in zip(positions, values):
    #             plt.text(pos, value, f'{value:.1f}', ha='center', va='bottom')
    #
    #     add_labels(x - bar_width, avg_lateness)
    #     add_labels(x, median_lateness)
    #     add_labels(x + bar_width, max_lateness)
    #
    #     plt.xlabel('Configuration')
    #     plt.ylabel('Hours Late')
    #     plt.title('Lateness Statistics by Configuration')
    #     plt.xticks(x, configs)
    #     plt.legend()
    #     plt.grid(axis='y', linestyle='--', alpha=0.3)
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, "lateness_hours_statistics.png"), dpi=300)
    #     plt.close()
    #
    #     # 4. Create line charts showing how lateness varies by depth
    #     if lateness_by_depth:
    #         depths = sorted(lateness_by_depth.keys())
    #
    #         # Calculate statistics by depth
    #         avg_lateness_by_depth = [np.mean(lateness_by_depth[depth]) for depth in depths]
    #         median_lateness_by_depth = [np.median(lateness_by_depth[depth]) for depth in depths]
    #         max_lateness_by_depth = [np.max(lateness_by_depth[depth]) for depth in depths]
    #
    #         plt.figure(figsize=(14, 8))
    #
    #         # Plot line charts
    #         plt.plot(depths, avg_lateness_by_depth, 'o-', linewidth=2, markersize=8,
    #                  color='blue', label='Average')
    #         plt.plot(depths, median_lateness_by_depth, 's--', linewidth=2, markersize=8,
    #                  color='green', label='Median')
    #         plt.plot(depths, max_lateness_by_depth, '^:', linewidth=2, markersize=8,
    #                  color='red', label='Maximum')
    #
    #         # Add data labels for average lateness
    #         for i, hours in enumerate(avg_lateness_by_depth):
    #             plt.text(depths[i], hours, f"{hours:.1f}", ha='center', va='bottom')
    #
    #         plt.xlabel('Instruction Depth')
    #         plt.ylabel('Hours Late')
    #         plt.title('Settlement Lateness by Instruction Depth')
    #         plt.xticks(depths)
    #         plt.legend()
    #         plt.grid(True, linestyle='--', alpha=0.3)
    #
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(output_dir, "lateness_hours_by_depth.png"), dpi=300)
    #         plt.close()
    #
    #         # 5. Create violin plot to show lateness distribution by depth
    #         plt.figure(figsize=(14, 8))
    #
    #         # Prepare data for violin plot
    #         violin_data = [lateness_by_depth[depth] for depth in depths]
    #
    #         # Create violin plot
    #         parts = plt.violinplot(violin_data, positions=depths, showmeans=True, showmedians=True)
    #
    #         # Customize violin plot
    #         for pc in parts['bodies']:
    #             pc.set_facecolor('lightblue')
    #             pc.set_alpha(0.7)
    #
    #         parts['cmeans'].set_color('red')
    #         parts['cmedians'].set_color('green')
    #
    #         plt.xlabel('Instruction Depth')
    #         plt.ylabel('Hours Late')
    #         plt.title('Distribution of Lateness Hours by Instruction Depth')
    #         plt.grid(True, linestyle='--', alpha=0.3)
    #
    #         # Add legend
    #         from matplotlib.lines import Line2D
    #         legend_elements = [
    #             Line2D([0], [0], color='red', lw=2, label='Mean'),
    #             Line2D([0], [0], color='green', lw=2, label='Median')
    #         ]
    #         plt.legend(handles=legend_elements)
    #
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(output_dir, "lateness_hours_violin.png"), dpi=300)
    #         plt.close()
    #
    #     # 6. Create heatmap of lateness hours by depth and configuration
    #     # We need to collect more data first
    #     lateness_by_config_depth = defaultdict(lambda: defaultdict(list))
    #
    #     # Process log files again to get lateness by depth and config
    #     for log_file in log_files:
    #         config_match = re.search(r'config(\d+)', log_file)
    #         if not config_match:
    #             continue
    #
    #         config_num = int(config_match.group(1))
    #         config_name = f"Config {config_num}"
    #
    #         try:
    #             with open(os.path.join(log_folder, log_file), 'r') as f:
    #                 log_data = json.load(f)
    #
    #             # Handle different formats
    #             if isinstance(log_data, dict):
    #                 if "ocel:events" in log_data:
    #                     # Classic OCEL format
    #                     event_key = "ocel:events"
    #                     attr_key = "ocel:attributes"
    #                     activity_key = "ocel:activity"
    #
    #                     # Process events (dictionary format)
    #                     events = log_data.get(event_key, {})
    #                     for event_id, event in events.items():
    #                         if event.get(activity_key) == "Settled Late":
    #                             attributes = event.get(attr_key, {})
    #
    #                             if "lateness_hours" in attributes and "depth" in attributes:
    #                                 lateness_hours = float(attributes["lateness_hours"])
    #                                 depth = int(attributes["depth"])
    #
    #                                 lateness_by_config_depth[config_name][depth].append(lateness_hours)
    #
    #                 elif "events" in log_data:
    #                     # Alternative format with events as an array
    #                     events = log_data.get("events", [])
    #
    #                     for event in events:
    #                         event_type = event.get("type")
    #                         if event_type == "Settled Late" or event_type == "transaction_settled_late":
    #                             # Try to extract depth and lateness
    #                             lateness_hours = None
    #                             depth = None
    #
    #                             # Try attributes for lateness_hours and depth
    #                             for attr in event.get("attributes", []):
    #                                 if attr["name"] == "lateness_hours" and "value" in attr:
    #                                     try:
    #                                         lateness_hours = float(attr["value"])
    #                                     except (ValueError, TypeError):
    #                                         pass
    #                                 elif attr["name"] == "depth" and "value" in attr:
    #                                     try:
    #                                         depth = int(attr["value"])
    #                                     except (ValueError, TypeError):
    #                                         pass
    #
    #                             # If we didn't find depth in attributes, try extracting from object IDs
    #                             if depth is None:
    #                                 for rel in event.get("relationships", []):
    #                                     if "objectId" in rel:
    #                                         obj_id = rel["objectId"]
    #                                         depth_match = re.search(r'_(\d+)_', obj_id)
    #                                         if depth_match:
    #                                             depth = int(depth_match.group(1))
    #                                             break
    #
    #                             # If no lateness_hours but we need a value, use 1 hour as fallback
    #                             if lateness_hours is None:
    #                                 lateness_hours = 1.0
    #
    #                             # If we have both pieces of data, store it
    #                             if lateness_hours is not None and depth is not None:
    #                                 lateness_by_config_depth[config_name][depth].append(lateness_hours)
    #         except Exception as e:
    #             pass  # Already logged errors above
    #
    #     # Create heatmap if we have data
    #     if lateness_by_config_depth:
    #         # Get all depths across all configurations
    #         all_depths = set()
    #         for config in lateness_by_config_depth:
    #             all_depths.update(lateness_by_config_depth[config].keys())
    #
    #         depths_for_heatmap = sorted(all_depths)
    #         configs_for_heatmap = sorted(lateness_by_config_depth.keys(),
    #                                      key=lambda x: int(x.split()[1]))
    #
    #         if depths_for_heatmap and configs_for_heatmap:
    #             # Create matrix for average lateness hours
    #             heatmap_data = np.zeros((len(depths_for_heatmap), len(configs_for_heatmap)))
    #
    #             for i, depth in enumerate(depths_for_heatmap):
    #                 for j, config in enumerate(configs_for_heatmap):
    #                     hours = lateness_by_config_depth[config].get(depth, [])
    #                     if hours:
    #                         heatmap_data[i, j] = np.mean(hours)
    #
    #             plt.figure(figsize=(14, 10))
    #
    #             # Create heatmap with custom colormap
    #             cmap = plt.cm.get_cmap('YlOrRd')
    #             aspect = max(0.1, min(5, len(configs_for_heatmap) / len(depths_for_heatmap)))
    #             im = plt.imshow(heatmap_data, cmap=cmap, aspect=aspect)
    #
    #             # Add colorbar
    #             plt.colorbar(im, label='Average Hours Late')
    #
    #             # Configure axes
    #             plt.yticks(range(len(depths_for_heatmap)), depths_for_heatmap)
    #             plt.xticks(range(len(configs_for_heatmap)), configs_for_heatmap,
    #                        rotation=45, ha='right')
    #             plt.ylabel('Instruction Depth')
    #             plt.xlabel('Configuration')
    #             plt.title('Average Lateness Hours by Depth and Configuration')
    #
    #             # Add values in cells
    #             for i in range(len(depths_for_heatmap)):
    #                 for j in range(len(configs_for_heatmap)):
    #                     if heatmap_data[i, j] > 0:
    #                         text_color = 'white' if heatmap_data[i, j] > np.max(heatmap_data) * 0.7 else 'black'
    #                         plt.text(j, i, f"{heatmap_data[i, j]:.1f}",
    #                                  ha="center", va="center", color=text_color)
    #
    #             plt.tight_layout()
    #             plt.savefig(os.path.join(output_dir, "lateness_hours_heatmap.png"), dpi=300)
    #             plt.close()
    #
    #     # 7. Create scatter plot of lateness hours vs depth
    #     if lateness_by_depth:
    #         plt.figure(figsize=(12, 8))
    #
    #         # Collect data for scatter plot
    #         x_values = []  # depth values
    #         y_values = []  # hours late values
    #
    #         for depth, hours_list in lateness_by_depth.items():
    #             for hours in hours_list:
    #                 x_values.append(depth)
    #                 y_values.append(hours)
    #
    #         # Create scatter plot with transparency
    #         plt.scatter(x_values, y_values, alpha=0.5, c='blue')
    #
    #         # Add trend line
    #         z = np.polyfit(x_values, y_values, 1)
    #         p = np.poly1d(z)
    #
    #         # Generate x values for trend line
    #         x_trend = range(min(depths), max(depths) + 1)
    #         plt.plot(x_trend, p(x_trend), "r--", label=f"Trend: y = {z[0]:.3f}x + {z[1]:.3f}")
    #
    #         plt.xlabel('Instruction Depth')
    #         plt.ylabel('Hours Late')
    #         plt.title('Relationship Between Instruction Depth and Lateness Hours')
    #         plt.grid(True, linestyle='--', alpha=0.3)
    #         plt.legend()
    #
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(output_dir, "lateness_hours_scatter.png"), dpi=300)
    #         plt.close()
    #
    #     # 8. Create time bucket analysis - categorize lateness
    #     plt.figure(figsize=(12, 8))
    #
    #     # Define time buckets (in hours)
    #     buckets = [
    #         (0, 1, "< 1 hour"),
    #         (1, 6, "1-6 hours"),
    #         (6, 12, "6-12 hours"),
    #         (12, 24, "12-24 hours"),
    #         (24, 48, "1-2 days"),
    #         (48, float('inf'), "> 2 days")
    #     ]
    #
    #     # Count instances in each bucket for each configuration
    #     bucket_counts = {config: [0] * len(buckets) for config in configs}
    #
    #     for config, hours_list in config_lateness.items():
    #         for hours in hours_list:
    #             for i, (lower, upper, _) in enumerate(buckets):
    #                 if lower <= hours < upper:
    #                     bucket_counts[config][i] += 1
    #                     break
    #
    #     # Calculate percentages
    #     bucket_pcts = {}
    #     for config, counts in bucket_counts.items():
    #         total = sum(counts)
    #         if total > 0:
    #             bucket_pcts[config] = [count / total * 100 for count in counts]
    #         else:
    #             bucket_pcts[config] = [0] * len(buckets)
    #
    #     # Create stacked bar chart
    #     bottom = np.zeros(len(configs))
    #     bucket_colors = ['green', 'yellowgreen', 'gold', 'orange', 'darkorange', 'red']
    #
    #     for i, (_, _, bucket_name) in enumerate(buckets):
    #         values = [bucket_pcts[config][i] for config in configs]
    #         plt.bar(configs, values, bottom=bottom, label=bucket_name, color=bucket_colors[i])
    #
    #         # Add percentage labels if large enough
    #         for j, v in enumerate(values):
    #             if v > 7:  # Only show label if segment is large enough
    #                 plt.text(j, bottom[j] + v / 2, f'{v:.1f}%', ha='center', va='center',
    #                          color='black', fontweight='bold')
    #
    #         bottom += np.array(values)
    #
    #     plt.xlabel('Configuration')
    #     plt.ylabel('Percentage of Late Settlements')
    #     plt.title('Lateness Time Categories by Configuration')
    #     plt.legend(title="Lateness Category")
    #     plt.grid(axis='y', linestyle='--', alpha=0.3)
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, "lateness_categories.png"), dpi=300)
    #     plt.close()
    #
    #     print(f"Lateness hours analysis visualizations created in {output_dir}")

if __name__ == "__main__":
    # Create analyzer and run analysis
    print("Starting SettlementAnalyzer...")
    analyzer = SettlementAnalyzer()
    analyzer.analyze_all("settlement_analysis/")