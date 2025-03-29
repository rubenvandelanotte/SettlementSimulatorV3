from datetime import datetime, timedelta
from mesa import Agent
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from TransactionAgent import TransactionAgent
    from SettlementModel import SettlementModel
    from InstitutionAgent import InstitutionAgent
    from Account import Account

class InstructionAgent (Agent):
    def __init__(self, model: "SettlementModel", uniqueID: str, motherID: str, institution: "InstitutionAgent", securitiesAccount: "Account", cashAccount: "Account", securityType: str, amount: float, isChild: bool, status: str, linkcode: str, creation_time: datetime, linkedTransaction: Optional["TransactionAgent"] = None):
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
        self.creation_time = creation_time# track creation time for timeout
        self.linkedTransaction = linkedTransaction
        self.last_matched = creation_time


#getter methods
    def get_model(self):
        """Returns the model associated with the agent."""
        return self.model

    def get_linkedTransaction(self):
        """Returns the linked transaction agent."""
        return self.linkedTransaction

    def get_uniqueID(self):
        """Returns the unique ID of the instruction."""
        return self.uniqueID

    def get_motherID(self):
        """Returns the mother ID of the instruction (if it's a child instruction)."""
        return self.motherID

    def get_institution(self):
        """Returns the institution associated with the instruction."""
        return self.institution

    def get_securitiesAccount(self):
        """Returns the securities account linked to the instruction."""
        return self.securitiesAccount

    def get_cashAccount(self):
        """Returns the cash account linked to the instruction."""
        return self.cashAccount

    def get_securityType(self):
        """Returns the type of security involved in the instruction."""
        return self.securityType

    def get_amount(self):
        """Returns the amount of securities or cash involved in the instruction."""
        return self.amount

    def get_isChild(self):
        """Returns whether the instruction is a child instruction."""
        return self.isChild

    def get_status(self):
        """Returns the current status of the instruction."""
        return self.status

    def get_linkcode(self):
        """Returns the link code associated with the instruction."""
        return self.linkcode

    def get_creation_time(self):
        """Returns the creation time of the instruction."""
        return self.creation_time

    def set_status(self, new_status: str):
        self.status = new_status

    def insert(self):
        if self.creation_time < self.model.simulated_time:
            if self.status == 'Exists':
                self.status = 'Pending'
                # logging
                #self.model.log_event(f"Instruction {self.uniqueID} inserted.", self.uniqueID, is_transaction=True)
                self.model.log_ocel_event(
                    activity="Instruction Inserted",
                    object_refs=[{"object_id": self.uniqueID, "object_type": "Instruction"}]
                )
    def validate(self):
        if self.status == 'Pending':
            self.set_status('Validated')
            #logging
            #self.model.log_event(f"Instruction {self.uniqueID} validated.", self.uniqueID, is_transaction = True)
            self.model.log_ocel_event(
                activity="Instruction Validated",
                object_refs=[{"object_id": self.uniqueID, "object_type": "Instruction"}]
            )
    def is_instruction_time_out(self):
        return self.creation_time + timedelta(days = 14) <= self.model.simulated_time

    def step(self):

       if self.is_instruction_time_out():
           self.cancel_timout()
       else:
           if self.status == 'Exists':
               self.insert()
           elif self.status == 'Pending':
               self.validate()
           elif self.status == "Validated":
               if self.last_matched+ timedelta(seconds=3) <= self.model.simulated_time:
                    self.match()
                    self.last_matched = self.model.simulated_time
               else:
                   return


       self.model.simulated_time = self.model.simulated_time +timedelta(seconds=1)


#