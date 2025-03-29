import pandas as pd
import json

#module to split the logs:
def split_ocel_event_log_to_traditional(filename="ocel_event_log.csv"):
    """
    Reads an OCEL event log from a CSV file and creates traditional event logs for each object type.
    Each traditional event log contains:
      - event_id: the original event id from the OCEL event.
      - case_id: the object id from the OCEL event's object_refs.
      - timestamp: the event timestamp.
      - activity: the event activity.
    The function saves each traditional event log as a separate CSV file.
    """
    # Load the OCEL event log CSV file
    ocel_df = pd.read_csv(filename)

    # Dictionary to collect traditional log events for each object type
    logs_by_object = {}

    # Process each event in the OCEL log
    for _, row in ocel_df.iterrows():
        # Parse the object_refs field (assuming it's stored as a JSON-like string)
        try:
            object_refs = json.loads(row["object_refs"].replace("'", "\""))
        except Exception as e:
            print(f"Error parsing object_refs: {e}")
            continue

        # For each referenced object, create a traditional event record
        for ref in object_refs:
            obj_type = ref.get("object_type")
            if obj_type:
                new_event = {
                    "event_id": row.get("event_id"),
                    "case_id": int(ref.get("object_id")),
                    "timestamp": row.get("timestamp"),
                    "activity": row.get("activity")
                }
                if obj_type not in logs_by_object:
                    logs_by_object[obj_type] = []
                logs_by_object[obj_type].append(new_event)

    # Save each traditional event log as its own CSV file
    for obj_type, events in logs_by_object.items():
        df = pd.DataFrame(events)
        out_filename = f"{obj_type}_traditional_event_log.csv"
        df.to_csv(out_filename, index=False)
        print(f"Saved traditional event log for {obj_type} to {out_filename}")


split_ocel_event_log_to_traditional("Logs/event_log.csv")

