import json
import uuid
from datetime import datetime


class JSONOCELLogger:
    def __init__(self):
        # Predefine event types and object types (can be extended dynamically if needed)
        self.eventTypes = [
            {
                "name": "delivery_instruction_created",
                "attributes": [
                    {"name": "securityType", "type": "string"},
                    {"name": "amount", "type": "number"},
                    {"name": "status", "type": "string"},
                    {"name": "linkcode", "type": "string"}
                ]
            },
            {
                "name": "receipt_instruction_created",
                "attributes": [
                    {"name": "securityType", "type": "string"},
                    {"name": "amount", "type": "number"},
                    {"name": "status", "type": "string"},
                    {"name": "linkcode", "type": "string"}
                ]
            }
        ]

        self.objectTypes = [
            {"name": "DeliveryInstruction", "attributes": []},
            {"name": "ReceiptInstruction", "attributes": []},
            {"name": "Institution", "attributes": []},
            {"name": "Account", "attributes": []}
        ]

        self.events = []  # List of event dictionaries
        self.objects = []  # List of object dictionaries

    def log_object(self, oid, otype, attributes=None):
        """
        Log an object.
        :param oid: Unique object ID.
        :param otype: Object type (should match one of the defined types, e.g. "DeliveryInstruction").
        :param attributes: Optional list of attribute dicts: [{"name": ..., "time": ..., "value": ...}, ...]
        """
        if attributes is None:
            attributes = []
        obj = {
            "id": oid,
            "type": otype,
            "attributes": attributes
        }
        self.objects.append(obj)
        print(f"Logged object: {oid} of type {otype}")

    def log_event(self, event_type, object_ids, event_attributes=None, relationships=None, timestamp=None):
        """
        Log an event.
        :param event_type: The type of the event (e.g. "delivery_instruction_created").
        :param object_ids: A list of object IDs involved in the event.
        :param event_attributes: Optional dict of attributes (will be converted to a list of {name, value}).
        :param relationships: Optional list of relationships, each as a dict: {"objectId": ..., "qualifier": ...}
        :param timestamp: ISO formatted timestamp. If None, current time is used.
        """
        if event_attributes is None:
            event_attributes = {}
        if relationships is None:
            relationships = []
        if timestamp is None:
            timestamp = datetime.now().isoformat() + "Z"  # Append "Z" for UTC

        # Convert event_attributes dict into a list of {name, value} objects
        attributes_list = [{"name": key, "value": value} for key, value in event_attributes.items()]

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "type": event_type,
            "time": timestamp,
            "attributes": attributes_list,
            "relationships": relationships
        }
        self.events.append(event)
        print(f"Logged event: {event_id} of type {event_type}")

    def export_log(self, filename):
        """
        Export the complete log as a JSONOCEL file.
        :param filename: Output filename (suggest using .jsonocel extension).
        """
        log_data = {
            "eventTypes": self.eventTypes,
            "objectTypes": self.objectTypes,
            "events": self.events,
            "objects": self.objects
        }
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=4)
        print(f"Exported JSONOCEL log to {filename}")

