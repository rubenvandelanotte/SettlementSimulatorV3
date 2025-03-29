#!/usr/bin/env python3
"""
Object‑Centric Process Mining with ocpa (latest version) => WIP

This script demonstrates:
    • Loading an OCEL from a CSV file using ocpa’s CSV importer.
    • Discovering an Object‑Centric Petri Net (OCPN) using ocpa’s discovery algorithm.
    • Visualizing the discovered OCPN.
    • Placeholder conformance checking.
    • Performance analysis (computing average and median time between events).

Before running, install ocpa from the latest version:
    pip install git+https://github.com/ocpm/ocpa.git
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# --- Monkey-patch pd.read_csv to accept a second positional argument for the separator ---
_original_read_csv = pd.read_csv
def _patched_read_csv(filepath_or_buffer, sep, **kwargs):
    return _original_read_csv(filepath_or_buffer, sep=sep, **kwargs)
pd.read_csv = _patched_read_csv

# --- Import ocpa modules with distinct names ---
from ocpa.objects.log.importer.csv import factory as csv_importer_factory
from ocpa.algo.discovery.ocpn import algorithm as ocpn_algorithm
from ocpa.visualization.oc_petri_net import factory as visualizer_factory

# ----- Step 1: Load the OCEL from CSV -----
def load_ocel_csv(file_path):
    """
    Loads an OCEL from a CSV file using ocpa’s CSV importer.

    The CSV should have the required columns (e.g., event_id, timestamp, activity, object_refs).
    You must specify a parameters dictionary to map CSV column names to the keys expected by ocpa.
    The importer returns an OCEL object with attributes 'events' and 'objects'.
    """
    parameters = {
        "evt_id": "event_id",         # Maps the CSV column "event_id" to the expected "evt_id" key
        "act_name": "activity",         # Maps the CSV column "activity" to the expected "act_name" key
        "time_name": "timestamp",            # Maps the CSV column "timestamp" to the expected "time" key
        "obj_refs": "object_refs",      # Maps the CSV column "object_refs" to the expected "obj_refs" key
        "sep": ",",
        "obj_names": []                # No additional object attribute table provided
    }
    ocel = csv_importer_factory.apply(file_path, parameters=parameters)
    return ocel

# ----- Step 2: Discover an Object‑Centric Petri Net (OCPN) -----
def discover_ocpn(ocel):
    """
    Discovers an Object‑Centric Petri Net (OCPN) from the OCEL using ocpa’s discovery algorithm.

    Returns:
        - ocpn_model: The discovered OCPN model.
        - initial_marking: The initial marking for the Petri net.
        - final_marking: The final marking for the Petri net.
    """
    ocpn_model, initial_marking, final_marking = ocpn_algorithm.apply(ocel)
    return ocpn_model, initial_marking, final_marking

# ----- Step 3: Visualize the Discovered OCPN -----
def visualize_ocpn(ocpn_model, initial_marking, final_marking):
    """
    Visualizes the discovered Object‑Centric Petri Net using ocpa’s visualization functionality.
    """
    parameters = {"format": "png"}
    gviz = visualizer_factory.apply(ocpn_model, initial_marking, final_marking, parameters=parameters)
    visualizer_factory.view(gviz)

# ----- Step 4: Conformance Checking (Placeholder) -----
def conformance_check_ocel(ocel, ocpn_model, initial_marking, final_marking):
    """
    Placeholder for conformance checking between the OCEL and the discovered OCPN.
    (In a full implementation you might use token‑based replay or alignment‑based conformance checking.)

    Returns:
        - fitness: A dummy fitness score.
        - missing_transitions: A list of missing transitions (empty in this placeholder).
    """
    fitness = 0.95  # Dummy value
    missing_transitions = []
    return fitness, missing_transitions

# ----- Step 5: Performance Analysis -----
def performance_analysis_ocel(ocel):
    """
    Computes performance metrics (throughput times) from the OCEL.

    Assumes that the OCEL object has an 'events' attribute which is a list of event objects
    each containing a 'timestamp' (as a string). This function converts the events into a DataFrame,
    sorts them by timestamp, and computes the average and median time differences (in seconds)
    between consecutive events.
    """
    events = ocel.events
    # Convert events to dictionaries if needed
    events_list = [e.__dict__ if hasattr(e, '__dict__') else e for e in events]
    events_df = pd.DataFrame(events_list)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    events_df = events_df.sort_values("timestamp")
    events_df["time_diff"] = events_df["timestamp"].diff().dt.total_seconds()
    avg_time = events_df["time_diff"].mean()
    median_time = events_df["time_diff"].median()
    return avg_time, median_time

# ----- Main Function -----
def main():
    # Path to your CSV file containing the OCEL (update as needed)
    csv_file = "Logs/event_log.csv"

    # Load the OCEL using ocpa's CSV importer
    ocel = load_ocel_csv(csv_file)
    print("Loaded OCEL from CSV:")
    print(f"Number of events: {len(ocel.events)}")
    print(f"Number of objects: {len(ocel.objects)}")

    # Discover the Object‑Centric Petri Net (OCPN)
    ocpn_model, initial_marking, final_marking = discover_ocpn(ocel)
    print("Discovered OCPN.")

    # Visualize the discovered OCPN
    visualize_ocpn(ocpn_model, initial_marking, final_marking)

    # Conformance checking (placeholder)
    fitness, missing = conformance_check_ocel(ocel, ocpn_model, initial_marking, final_marking)
    print(f"Conformance Fitness: {fitness:.2f}")
    if missing:
        print("Missing transitions (placeholder):", missing)

    # Performance analysis: Compute average and median time differences between events
    avg_time, median_time = performance_analysis_ocel(ocel)
    print(f"Average time between events: {avg_time:.2f} seconds")
    print(f"Median time between events: {median_time:.2f} seconds")

if __name__ == "__main__":
    main()
