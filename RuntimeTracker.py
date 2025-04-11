import time
import json
import os
from datetime import datetime


class RuntimeTracker:
    """
    A simple class to track the runtime of different simulation configurations
    and save the results to a JSON file.
    """

    def __init__(self, output_file="runtime_results.json"):
        """
        Initialize the runtime tracker.

        Args:
            output_file: File where runtime results will be saved
        """
        self.results = []
        self.output_file = output_file

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def track_runtime(self, run_function, config, run_label=None):
        """
        Run a simulation while tracking its runtime and record the results.

        Args:
            run_function: The function to run the simulation
            config: Configuration parameters for the simulation
            run_label: Optional label for this run

        Returns:
            The result of the simulation along with runtime information
        """
        # Generate run label if not provided
        if run_label is None:
            run_label = f"Run-{len(self.results) + 1}"

        print(f"Starting run '{run_label}'...")

        # Record start time
        start_time = time.time()

        # Run the simulation
        simulation_result = run_function(config)

        # Record end time
        end_time = time.time()
        execution_time = end_time - start_time

        # Create result record
        result_record = {
            "timestamp": datetime.now().isoformat(),
            "run_label": run_label,
            "config": config,
            "execution_time_seconds": execution_time
        }

        # Add to results list
        self.results.append(result_record)

        print(f"Run '{run_label}' completed in {execution_time:.2f} seconds")

        return {
            "execution_info": result_record,
            "simulation_result": simulation_result
        }

    def save_results(self):
        """
        Save the runtime results to a JSON file.
        """
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"Runtime results saved to {self.output_file}")

    def load_results(self):
        """
        Load runtime results from the JSON file if it exists.
        """
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded {len(self.results)} previous runtime results")
        else:
            print(f"No previous runtime results found at {self.output_file}")

