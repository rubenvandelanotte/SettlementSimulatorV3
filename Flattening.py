import json
import os
from datetime import datetime

# Load OCEL file
input_file_path = r"C:\Users\matth\Documents\GitHub\SettlementSimulatorV3\partial_allowance_files\logs\log_partial_truecount10_run1.jsonocel"
with open(input_file_path) as f:
    ocel_data = json.load(f)

# Extract original filename without extension
file_name = os.path.basename(input_file_path)
file_name_without_ext = os.path.splitext(file_name)[0]

# Choose the object type to flatten on
TARGET_OBJECT_TYPE = "DeliveryInstruction"  # or "ReceiptInstruction"

# Create a new JSONOCEL structure
flattened_ocel = {
    "ocel:global-log": {
        "ocel:attribute-names": [],
        "ocel:object-types": [TARGET_OBJECT_TYPE],
        "ocel:version": "1.0",
        "ocel:ordering": "timestamp"
    },
    "ocel:objects": {},
    "ocel:events": {}
}

# Collect all unique attribute names
attribute_names = set()

# Create mapping from object ID to events
object_event_map = {}

# Examine events structure
for event in ocel_data.get("events", []):
    # Collect attribute names
    for attr in event.get("attributes", []):
        attribute_names.add(attr["name"])

    # Look for relationships that might indicate objects
    for relationship in event.get("relationships", []):
        object_id = relationship.get("objectId")

        # Try to determine if this is our target object type
        is_target = False
        if isinstance(object_id, int) and object_id % 2 == 1 and TARGET_OBJECT_TYPE == "DeliveryInstruction":
            is_target = True
        elif isinstance(object_id, int) and object_id % 2 == 0 and TARGET_OBJECT_TYPE == "ReceiptInstruction":
            is_target = True

        if is_target:
            object_event_map.setdefault(object_id, []).append(event)

# Update the attribute names in the global log
flattened_ocel["ocel:global-log"]["ocel:attribute-names"] = list(attribute_names)

# Create objects in the JSONOCEL
for obj_id in object_event_map:
    flattened_ocel["ocel:objects"][str(obj_id)] = {
        "ocel:type": TARGET_OBJECT_TYPE,
        "ocel:ovmap": {}  # Object attributes can be added here if available
    }

# Create events in the JSONOCEL
for obj_id, events in object_event_map.items():
    for event in events:
        event_id = event["id"]

        # Skip if this event is already processed
        if event_id in flattened_ocel["ocel:events"]:
            # Just add this object to the existing event's omap
            flattened_ocel["ocel:events"][event_id]["ocel:omap"].append(str(obj_id))
            continue

        # Create a new event entry
        flattened_ocel["ocel:events"][event_id] = {
            "ocel:activity": event["type"],
            "ocel:timestamp": event["time"],
            "ocel:omap": [str(obj_id)],  # List of objects involved in this event
            "ocel:vmap": {}  # Event attributes
        }

        # Add attributes to vmap
        for attr in event.get("attributes", []):
            flattened_ocel["ocel:events"][event_id]["ocel:vmap"][attr["name"]] = attr["value"]

# Create a new directory for flattened logs if it doesn't exist
output_dir = r"C:\Users\matth\Documents\GitHub\SettlementSimulatorV3\partial_allowance_files\flattened_logs"
os.makedirs(output_dir, exist_ok=True)

# Create the output filename
output_filename = f"{file_name_without_ext}_{TARGET_OBJECT_TYPE}_flattened.jsonocel"
output_path = os.path.join(output_dir, output_filename)

# Export to JSONOCEL format
with open(output_path, "w") as f:
    json.dump(flattened_ocel, f, indent=2)

print(f"Successfully created JSONOCEL file: {output_path}")