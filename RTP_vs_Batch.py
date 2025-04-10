import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, time
from collections import defaultdict


def add_rtp_batch_analysis_to_settlement_analyzer():
    """
    Method to extend the SettlementAnalyzer class with RTP vs Batch analysis capability
    """

    def analyze_rtp_vs_batch(self, log_folder="simulatie_logs/", output_dir="settlement_analysis/rtp_vs_batch/"):
        """
        Analyze JSONOCEL logs to compare settlements in Real-Time Processing vs Batch processing

        Args:
            log_folder: Directory containing simulation logs in JSONOCEL format
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)

        # Dictionary to store data for each configuration
        config_data = defaultdict(lambda: {"rtp": 0, "batch": 0})

        # Find all JSONOCEL files
        log_files = [f for f in os.listdir(log_folder) if f.endswith(".jsonocel")]

        if not log_files:
            print(f"No JSONOCEL files found in {log_folder}")
            return

        print(f"Found {len(log_files)} JSONOCEL files for analysis")

        # Process each log file
        for log_file in log_files:
            # Extract configuration number from filename
            parts = log_file.split("_")
            if len(parts) >= 3 and parts[0] == "simulation" and parts[1].startswith("config"):
                config_num = int(parts[1].replace("config", ""))
                config_name = f"Config {config_num}"

                try:
                    # Load JSONOCEL data
                    with open(os.path.join(log_folder, log_file), 'r') as f:
                        log_data = json.load(f)

                    # Extract events
                    events = log_data.get("ocel:events", {})

                    # Process each event
                    for event_id, event in events.items():
                        activity = event.get("ocel:activity")
                        timestamp_str = event.get("ocel:timestamp")

                        # Check if this is a settlement event
                        if activity in ["Settled On Time", "Settled Late"]:
                            # Parse timestamp to determine if RTP or Batch
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                event_time = timestamp.time()

                                # Check if it's RTP (trading hours) or Batch time
                                # Using the model's trading hours (1:30-19:30) and batch time (22:00+)
                                if time(1, 30) <= event_time <= time(19, 30):
                                    config_data[config_name]["rtp"] += 1
                                elif event_time >= time(22, 0):
                                    config_data[config_name]["batch"] += 1
                            except Exception as e:
                                print(f"Error parsing timestamp {timestamp_str}: {e}")

                except Exception as e:
                    print(f"Error processing file {log_file}: {e}")

        # Create visualizations if we have data
        if not config_data:
            print("No settlement data found for RTP vs Batch analysis")
            return

        # Sort configurations by their number
        configs = sorted(config_data.keys(), key=lambda x: int(x.split()[1]))

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

        # 2. Create absolute count bar chart with side-by-side bars
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

        # 3. Create line chart showing total settlements by config with RTP/Batch breakdown
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

    # Return the method to be added to the SettlementAnalyzer class
    return analyze_rtp_vs_batch

# When importing this code, add the method to the SettlementAnalyzer class:
# SettlementAnalyzer.analyze_rtp_vs_batch = add_rtp_batch_analysis_to_settlement_analyzer()