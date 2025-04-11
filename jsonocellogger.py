import json
import uuid
from datetime import datetime

class JSONOCELLogger:
    def __init__(self):
        self.eventTypes = [
            {"name": "delivery_instruction_created", "attributes": [
                {"name": "securityType", "type": "string"},
                {"name": "amount", "type": "number"},
                {"name": "status", "type": "string"},
                {"name": "linkcode", "type": "string"}
            ]},
            {"name": "receipt_instruction_created", "attributes": [
                {"name": "securityType", "type": "string"},
                {"name": "amount", "type": "number"},
                {"name": "status", "type": "string"},
                {"name": "linkcode", "type": "string"}
            ]},
            {"name": "transaction_created", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "instruction_inserted", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "instruction_validated", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "instruction_cancelled_timeout", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "instruction_attempting_to_match", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "instruction_matched_failed_due_to_incorrect_state", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "instruction_matched_failed: no counter instruction found", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "instruction_matched", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "transaction_attempting_to_settle", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "transaction_cancelled_error", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "transaction_settled_late", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "transaction_settled_on_time", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "transaction_settlement_failed_due_to_insufficient_funds", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "transaction_partially_settled", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "transaction_settlement_failed_incorrect_status", "attributes": [
                {"name": "status", "type": "string"}
            ]},
            {"name": "Partial_settlement_aborted", "attributes": [
                {"name": "status", "type": "string"}
            ]}
        ]

        self.objectTypes = [
            {"name": "DeliveryInstruction", "attributes": []},
            {"name": "ReceiptInstruction", "attributes": []},
            {"name": "Transaction", "attributes": []},
            {"name": "Institution", "attributes": []},
            {"name": "Account", "attributes": []}
        ]

        self.events = []
        self.objects = []

    def log_object(self, oid, otype, attributes=None):
        attributes = attributes if attributes else []
        self.objects.append({
            "id": oid,
            "type": otype,
            "attributes": attributes
        })
        #print(f"Logged object: {oid} of type {otype}")

    def log_event(self, event_type, object_ids, event_attributes=None, relationships=None, timestamp=None):
        event_attributes = event_attributes if event_attributes else {}
        relationships = relationships if relationships else []

        timestamp = timestamp or datetime.now().isoformat() + "Z"

        attributes_list = [{"name": key, "value": value} for key, value in event_attributes.items()]

        for oid in object_ids:
            relationships.append({"objectId": oid, "qualifier": "involved"})

        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "time": timestamp,
            "attributes": attributes_list,
            "relationships": relationships
        }
        self.events.append(event)
        #print(f"Logged event: {event['id']} of type {event_type}")

    def export_log(self, filename):
        with open(filename, "w") as f:
            json.dump({
                "eventTypes": self.eventTypes,
                "objectTypes": self.objectTypes,
                "events": self.events,
                "objects": self.objects
            }, f, indent=4)
        print(f"Exported JSONOCEL log to {filename}")