from datetime import datetime, timedelta
import random

from mesa import Agent
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from TransactionAgent import TransactionAgent
    from SettlementModel import SettlementModel
    from InstitutionAgent import InstitutionAgent
    from Account import Account

class InstructionAgent (Agent):
    def __init__(self, model: "SettlementModel", uniqueID: str, motherID: str, institution: "InstitutionAgent", securitiesAccount: "Account", cashAccount: "Account", securityType: str, amount: float, isChild: bool, status: str, linkcode: str, creation_time: datetime, original_creation_time: datetime, linkedTransaction: Optional["TransactionAgent"] = None, depth: int=0, original_mother_amount: int = None):
        super().__init__(model)
        self.uniqueID = uniqueID
        self.motherID = motherID
        self.institution = institution
        self.securitiesAccount = securitiesAccount
        self.cashAccount = cashAccount
        self.securityType = securityType
        self.amount = amount
        self.isChild = isChild
        self.status = status
        self.linkcode = linkcode
        self.creation_time = creation_time # track creation time for timeout
        self.original_creation_time = original_creation_time
        self.linkedTransaction = linkedTransaction
        self.last_matched = creation_time
        self.intended_settlement_date = None
        self.depth = depth
        self.last_modified_time = creation_time
        self.priority = None
        # Set original_mother_amount
        if original_mother_amount is None and not isChild:
            # If this is a mother instruction, set to its own amount
            self.original_mother_amount = amount
        else:
            # If this is a child, use the provided value
            self.original_mother_amount = original_mother_amount

    def get_uniqueID(self):
        return self.uniqueID

    def get_motherID(self):
        return self.motherID

    def get_institution(self):
        return self.institution

    def get_securitiesAccount(self):
        return self.securitiesAccount

    def get_securityType(self):
        return self.securityType

    def get_amount(self):
        return self.amount

    def set_amount(self, new_amount:str):
        self.amount = new_amount

    def get_status(self):
        return self.status

    def get_linkcode(self):
        return self.linkcode

    def get_creation_time(self):
        return self.creation_time

    def get_depth(self):
        return self.depth

    def get_intended_settlement_date(self):
        return self.intended_settlement_date

    def set_status(self, new_status: str):
        # Sets status to new_status
        old_status = self.status
        self.status = new_status
        self.last_modified_time = self.model.get_simulated_time()

        # Clean up from indices when status changes from "Validated"
        if old_status == "Validated" and new_status != "Validated":
            import DeliveryInstructionAgent
            import ReceiptInstructionAgent
            if isinstance(self, DeliveryInstructionAgent.DeliveryInstructionAgent):
                if self.linkcode in self.model.validated_delivery_instructions:
                    if self in self.model.validated_delivery_instructions[self.linkcode]:
                        self.model.validated_delivery_instructions[self.linkcode].remove(self)
            elif isinstance(self, ReceiptInstructionAgent.ReceiptInstructionAgent):
                if self.linkcode in self.model.validated_receipt_instructions:
                    if self in self.model.validated_receipt_instructions[self.linkcode]:
                        self.model.validated_receipt_instructions[self.linkcode].remove(self)


    def get_intended_settlement_date(self):
        return self.intended_settlement_date

    def set_intended_settlement_date(self, ts):
        self.intended_settlement_date = ts

    def set_linkedTransaction(self, lt):
        self.linkedTransaction = lt

    def assign_priority(self):
        amount = self.amount
        isd = self.get_intended_settlement_date()
        creation_time = self.creation_time
        age_in_days = (isd - creation_time).days

        if age_in_days < 0:
            self.priority = 3
        elif age_in_days == 0 and amount >= 10000000:
            self.priority = 3
        elif amount >= 100000000 or self.get_securitiesAccount().get_original_balance() * 0.2 >= amount:
            self.priority = 3
        elif (
                10000000 <= amount < 10000_000 or amount >= self.get_securitiesAccount().get_original_balance() * 0.07) and age_in_days <= 2:
            self.priority = 2
        else:
            self.priority = 1

    def get_priority(self):
        return self.priority

    def set_priority(self, new_priority):
        self.priority = new_priority

    def cancel_timeout(self):
        return

    def insert(self):
        # Insert Instruction
        if self.creation_time < self.model.get_simulated_time():
            if self.status == 'Exists':
                self.status = 'Pending'

                self.model.log_event(
                    event_type="Instruction Inserted",
                    object_ids=[self.uniqueID],
                    attributes={"status": self.status}
                )


    def validate(self):
        #Validate Instruction
        if self.status == 'Pending':
            self.set_status('Validated')

            # Add code to allow for faster matching lookup
            import DeliveryInstructionAgent
            import ReceiptInstructionAgent

            if isinstance(self, DeliveryInstructionAgent.DeliveryInstructionAgent):
                if self.linkcode not in self.model.validated_delivery_instructions:
                    self.model.validated_delivery_instructions[self.linkcode] = []
                self.model.validated_delivery_instructions[self.linkcode].append(self)
            elif isinstance(self, ReceiptInstructionAgent.ReceiptInstructionAgent):
                if self.linkcode not in self.model.validated_receipt_instructions:
                    self.model.validated_receipt_instructions[self.linkcode] = []
                self.model.validated_receipt_instructions[self.linkcode].append(self)

            self.model.log_event(
                event_type="Instruction Validated",
                object_ids=[self.uniqueID],
                attributes={"status": self.status}
            )

    def is_instruction_time_out(self):
        return self.intended_settlement_date + timedelta(days = 14) <= self.model.get_simulated_time()

    def step(self):
       # Step method for Instructions
       if self.is_instruction_time_out():
           self.cancel_timeout()
       else:
           if self.status == 'Exists':
               self.insert()
           elif self.status == 'Pending':
               self.validate()
           elif self.status == "Validated":
               if self.last_matched+ timedelta(hours=1) <= self.model.get_simulated_time():
                self.match()
                self.last_matched = self.model.get_simulated_time()


