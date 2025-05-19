import json
import uuid
from datetime import datetime


class JSONOCELLogger:
    def __init__(self, buffer_size=10000, temp_file_prefix="sim_log_temp"):
        self.eventTypes = [
            # Event types definitions remain the same
            {"name": "delivery_instruction_created", "attributes": [
                {"name": "securityType", "type": "string"},
                {"name": "amount", "type": "number"},
                {"name": "status", "type": "string"},
                {"name": "linkcode", "type": "string"}
            ]},
            # ... other event types ...
        ]

        self.objectTypes = [
            # Object types definitions remain the same
            {"name": "DeliveryInstruction", "attributes": []},
            # ... other object types ...
        ]

        self.events = []
        self.objects = []
        self.buffer_size = buffer_size
        self.temp_file_prefix = temp_file_prefix
        self.temp_files = []
        self.event_count = 0
        self.total_event_count = 0

    def log_object(self, oid, otype, attributes=None):
        attributes = attributes if attributes else []
        self.objects.append({
            "id": oid,
            "type": otype,
            "attributes": attributes
        })

    def log_event(self, event_type, object_ids, timestamp, event_attributes=None, relationships=None)



        event_attributes = event_attributes if event_attributes else {}
        relationships = relationships if relationships else []
        timestamp = timestamp

        attributes_list = [{"name": key, "value": value} for key, value in event_attributes.items()]

        for oid in object_ids:
            relationships.append({"objectId": oid, "qualifier": "involved"})

        event = {
            "id": str(self.total_event_count),
            "type": event_type,
            "time": timestamp,
            "attributes": attributes_list,
            "relationships": relationships
        }
        self.events.append(event)
        self.event_count += 1
        self.total_event_count += 1

        # When buffer is full, flush to disk
        if self.event_count >= self.buffer_size:
            self._flush_events_to_disk()

    def _flush_events_to_disk(self):
        import json
        import tempfile
        import os

        if not self.events:
            return

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(
            prefix=self.temp_file_prefix,
            suffix=".json",
            delete=False,
            mode="w"
        )

        # Write events to the file
        json.dump(self.events, temp_file)
        temp_file.close()

        # Keep track of the temporary file
        self.temp_files.append(temp_file.name)

        # Clear the events buffer
        self.events = []
        self.event_count = 0
        print(f"Flushed {self.buffer_size} events to disk, total files: {len(self.temp_files)}")

    def export_log(self, filename):
        import json

        # Flush any remaining events
        if self.events:
            self._flush_events_to_disk()

        # Combine all events from temporary files
        all_events = []
        for temp_file in self.temp_files:
            try:
                with open(temp_file, "r") as f:
                    events_chunk = json.load(f)
                    all_events.extend(events_chunk)
            except Exception as e:
                print(f"Error reading temp file {temp_file}: {str(e)}")

        # Write the complete log
        with open(filename, "w") as f:
            json.dump({
                "eventTypes": self.eventTypes,
                "objectTypes": self.objectTypes,
                "events": all_events,
                "objects": self.objects
            }, f, indent=4)

        # Clean up temporary files
        self._cleanup_temp_files()
        print(f"Exported JSONOCEL log to {filename} with {len(all_events)} events")

    def _cleanup_temp_files(self):
        import os

        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Error removing temp file {temp_file}: {str(e)}")

        self.temp_files = []

    def reset(self):
        """Reset the logger to its initial state"""
        # Flush any events in memory
        if self.events:
            self._flush_events_to_disk()

        # The temp_files list keeps track of files to be combined during export
        # We don't clear this as it would cause data loss

        # Clear objects list (these have already been written to files)
        print(f"Clearing {len(self.objects)} objects from logger")
        self.objects = []

        # Reset counters
        self.event_count = 0

        print("Logger reset complete")