import os
import json

from settlement_analysis.DepthAnalysis import DepthAnalyzer
from settlement_analysis.LatenessAnalysis import LatenessAnalyzer
from settlement_analysis.LatenessHoursAnalysis import LatenessHoursAnalyzer
from settlement_analysis.RuntimeAnalysis import RuntimeAnalyzer
from settlement_analysis.ConfidenceIntervalAnalysis import ConfidenceIntervalAnalyzer
from settlement_analysis.RTPvsBatchAnalysis import RTPvsBatchAnalyzer
from settlement_analysis.EfficiencyPerDay import EfficiencyPerDayAnalyzer
from settlement_analysis.EfficiencyPerParticipant import EfficiencyPerParticipantAnalyzer
from settlement_analysis.SettlementTypeAnalysis import SettlementTypeAnalyzer



class SettlementAnalysisSuite:
    def __init__(self, input_dir="./", output_dir="./"):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.statistics = {}
        self.runtime_data = []

        # Initialize analyzers
        self.depth_analyzer = DepthAnalyzer(self.input_dir, self.output_dir, self)
        self.lateness_analyzer = LatenessAnalyzer(self.input_dir, self.output_dir, self)
        self.lateness_hours_analyzer = LatenessHoursAnalyzer(self.input_dir, self.output_dir, self)
        self.runtime_analyzer = RuntimeAnalyzer(self.input_dir, self.output_dir, self)
        self.ci_analyzer = ConfidenceIntervalAnalyzer(self.input_dir, self.output_dir, self)
        self.rtp_vs_batch_analyzer = RTPvsBatchAnalyzer(self.input_dir, self.output_dir, self)
        self.eff_per_day_analyzer = EfficiencyPerDayAnalyzer(self.input_dir, self.output_dir, self)
        self.eff_per_part_analyzer = EfficiencyPerParticipantAnalyzer(self.input_dir,self.output_dir,self)
        self.settlement_type_analyzer = SettlementTypeAnalyzer(self.input_dir, self.output_dir, self)  # Add the new analyzer


    def analyze_all(self, analysis_types=None):
        if analysis_types is None:
            analysis_types = [
                "depth", "lateness", "lateness_hours", "runtime", "confidence_intervals", "rtp_vs_batch", "effi_per_day", "effi_per_part","settlement_type"
            ]

        # Load all inputs
        self._load_statistics()
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
        #if "effi_per_day" in analysis_types:
        #    self.eff_per_day_analyzer.run()
        #if "effi_per_part" in analysis_types:
        #    self.eff_per_part_analyzer.run()
        if "settlement_type" in analysis_types:
            self.settlement_type_analyzer.run()


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

    def _load_runtime_results(self):
        runtime_file = None

        # Try to guess the correct runtime file dynamically
        for filename in os.listdir(self.input_dir):
            if filename.startswith("runtime_") and filename.endswith(".json"):
                runtime_file = os.path.join(self.input_dir, filename)
                break

        if runtime_file and os.path.exists(runtime_file):
            with open(runtime_file, "r") as f:
                self.runtime_data = json.load(f)
            print(f"[INFO] Loaded {len(self.runtime_data)} runtime entries from {runtime_file}")
        else:
            self.runtime_data = []
            print(f"[WARNING] No runtime file found in {self.input_dir}")

if __name__ == "__main__":
        suite = SettlementAnalysisSuite(
            input_dir= R"C:\Users\matth\Documents\GitHub\SettlementSimulatorV3\partial_allowance_files",
            output_dir= os.path.join( R"C:\Users\matth\Documents\GitHub\SettlementSimulatorV3\partial_allowance_files", "visualizations", "partial"))
        suite.analyze_all()