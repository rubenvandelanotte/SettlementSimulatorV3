import os
import json
from settlement_analysis.DepthAnalysis import DepthAnalyzer
from settlement_analysis.LatenessAnalysis import LatenessAnalyzer
from settlement_analysis.LatenessHoursAnalysis import LatenessHoursAnalyzer
from settlement_analysis.RuntimeAnalysis import RuntimeAnalyzer
from settlement_analysis.ConfidenceIntervalAnalysis import ConfidenceIntervalAnalyzer
from settlement_analysis.RTPvsBatchAnalysis import RTPvsBatchAnalyzer

class SettlementAnalysisSuite:
    def __init__(self, input_dir="./", output_dir="./"):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.statistics = {}
        self.logs = []
        self.runtime_data = []

        # Initialize analyzers
        self.depth_analyzer = DepthAnalyzer(self.input_dir, self.output_dir, self)
        self.lateness_analyzer = LatenessAnalyzer(self.input_dir, self.output_dir, self)
        self.lateness_hours_analyzer = LatenessHoursAnalyzer(self.input_dir, self.output_dir, self)
        self.runtime_analyzer = RuntimeAnalyzer(self.input_dir, self.output_dir, self)
        self.ci_analyzer = ConfidenceIntervalAnalyzer(self.input_dir, self.output_dir, self)
        self.rtp_vs_batch_analyzer = RTPvsBatchAnalyzer(self.input_dir, self.output_dir, self)

    def analyze_all(self, analysis_types=None):
        if analysis_types is None:
            analysis_types = [
                "depth", "lateness", "lateness_hours", "runtime", "confidence_intervals", "rtp_vs_batch"
            ]

        # Load all inputs
        self._load_statistics()
        self._load_logs()
        self._load_runtime_results()

        # Dispatch analyzers
        if "depth" in analysis_types:
            self.depth_analyzer.run()
        if "lateness" in analysis_types:
            self.lateness_analyzer.run()
        if "lateness_hours" in analysis_types:
            self.lateness_hours_analyzer.run()
        if "runtime" in analysis_types:
            self.runtime_analyzer.run()
        if "confidence_intervals" in analysis_types:
            self.ci_analyzer.run()
        if "rtp_vs_batch" in analysis_types:
            self.rtp_vs_batch_analyzer.run()

    def _load_statistics(self):
        stats_dir = os.path.join(self.input_dir, "results_all_analysis")
        if os.path.exists(stats_dir):
            for file in os.listdir(stats_dir):
                if file.endswith(".json") and "_CRASH" not in file:
                    with open(os.path.join(stats_dir, file), "r") as f:
                        self.statistics[file] = json.load(f)
            print(f"[INFO] Loaded {len(self.statistics)} statistics files from results_all_analysis/")
        else:
            print(f"[WARNING] Statistics directory not found at {stats_dir}")

    def _load_logs(self):
        logs_dir = os.path.join(self.input_dir, "logs")
        if os.path.exists(logs_dir):
            for file in os.listdir(logs_dir):
                if file.endswith(".jsonocel"):
                    with open(os.path.join(logs_dir, file), "r") as f:
                        self.logs.append(json.load(f))
            print(f"[INFO] Loaded {len(self.logs)} logs from logs/")
        else:
            print(f"[WARNING] Logs directory not found at {logs_dir}")

    def _load_runtime_results(self):
        runtime_file = os.path.join(self.input_dir, "runtime_partial.json")
        if os.path.exists(runtime_file):
            with open(runtime_file, "r") as f:
                self.runtime_data = json.load(f)
            print(f"[INFO] Loaded {len(self.runtime_data)} runtime entries from runtime_results.json")
        else:
            self.runtime_data = []
            print(f"[WARNING] No runtime_results.json found at {runtime_file}")



