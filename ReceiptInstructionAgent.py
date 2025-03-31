from datetime import datetime
from typing import TYPE_CHECKING, Optional
import InstructionAgent

if TYPE_CHECKING:
    from SettlementModel import SettlementModel
    from InstitutionAgent import InstitutionAgent
    from Account import Account
    #from TransactionAgent import TransactionAgent

import TransactionAgent
import DeliveryInstructionAgent



class ReceiptInstructionAgent(InstructionAgent.InstructionAgent):
    def __init__(self, model: "SettlementModel", uniqueID: str, motherID: str, institution: "InstitutionAgent",
                 securitiesAccount: "Account", cashAccount: "Account", securityType: str, amount: float, isChild: bool,
                 status: str, linkcode: str, creation_time: datetime, linkedTransaction: Optional["TransactionAgent"] = None):
        super().__init__(
            model=model,
            linkedTransaction=linkedTransaction,
            uniqueID=uniqueID,
            motherID=motherID,
            institution=institution,
            securitiesAccount=securitiesAccount,
            cashAccount=cashAccount,
            securityType=securityType,
            amount=amount,
            isChild=isChild,
            status=status,
            linkcode=linkcode,
            creation_time=creation_time
        )

        self.model.log_object(
            object_id=self.uniqueID,
            object_type="ReceiptInstruction",
            attributes={
                "securityType": self.securityType,
                "amount": self.amount,
                "status": self.status,
                "linkcode": self.linkcode,
                "institutionID": self.institution.institutionID
            }
        )
        self.model.log_event(
            event_type="ReceiptInstruction Created",
            object_ids=[self.uniqueID, self.institution.institutionID, self.securitiesAccount.getAccountID()],
            attributes={
                "securityType": self.securityType,
                "amount": self.amount,
                "status": self.status,
                "linkcode": self.linkcode
            }
        )

        # logging ( don't know why is_transaction = True)
        #self.model.log_event(
        #    f"Receipt instruction with ID {uniqueID} created by institution {institution.institutionID} for {securityType} for amount {amount}",
        #    self.uniqueID, is_transaction=True)
        #old logger
        self.model.log_ocel_event(
            activity="Created",
            object_refs=[
                {"object_id": self.uniqueID, "object_type": "ReceiptInstruction"},
                {"object_id": self.institution.institutionID, "object_type": "Institution"},
                {"object_id": self.securitiesAccount.getAccountID(), "object_type": "Account"}
            ]
        )

    def get_creation_time(self):
        return self.creation_time

    def createReceiptChildren(self):

        MIN_SETTLEMENT_AMOUNT = self.model.min_settlement_amount
        # Calculate the actual available amounts using getBalance(), ensuring correct account types.
        if self.cashAccount.getAccountType() != "Cash":
            available_cash = 0
        else:
            available_cash = self.cashAccount.getBalance()

        deliverer = self.linkedTransaction.deliverer
        if deliverer.securitiesAccount.getAccountType() != self.securityType:
            available_securities = 0
        else:
            available_securities = deliverer.securitiesAccount.getBalance()

        # Compute the amount that can actually be settled.
        available_to_settle = min(self.amount, available_cash, available_securities)

        if available_to_settle > MIN_SETTLEMENT_AMOUNT:
            #  Create receipt child instructions with the computed amounts.
            receipt_child_1 = ReceiptInstructionAgent(
                self.model,
                f"{self.uniqueID}_1",
                self.uniqueID,
                self.institution,
                self.securitiesAccount,
                self.cashAccount,
                self.securityType,
                available_to_settle,
                True,
                "Validated",
                f"{self.linkcode}_1",
                creation_time=self.model.simulated_time,
                linkedTransaction=None
            )
            receipt_child_2 = ReceiptInstructionAgent(
                self.model,
                f"{self.uniqueID}_2",
                self.uniqueID,
                self.institution,
                self.securitiesAccount,
                self.cashAccount,
                self.securityType,
                self.amount - available_to_settle,
                True,
                "Validated",
                f"{self.linkcode}_2",
                creation_time=self.model.simulated_time,
                linkedTransaction=None
            )
            # Add the new child instructions to the agents scheduler.
            self.model.agents.add(receipt_child_1)
            self.model.agents.add(receipt_child_2)
            self.model.log_event(
                event_type="Receipt Children Created",
                object_ids=[receipt_child_1.uniqueID, receipt_child_2.uniqueID, self.uniqueID],
                attributes={"parentInstructionID": self.uniqueID}
            )
            return receipt_child_1, receipt_child_2
        else:
            # Log insufficient funds and return a tuple of Nones.
            #self.model.log_event(
            #    f"ReceiptInstruction {self.uniqueID}: insufficient funds for partial settlement.",
            #    self.uniqueID,
            #    is_transaction=True
            #)

            #new logging
            self.model.log_event(
                event_type="Partial Settlement Failed: Insufficient Funds",
                object_ids=[self.uniqueID],
                attributes={"status": self.status}
            )


            #old logging
            self.model.log_ocel_event(
                activity="Partial Settlement Failed: Insufficient Funds",
                object_refs=[{"object_id": self.uniqueID, "object_type": "ReceiptInstruction"}]
            )
            return (None, None)

    def match(self):
        """Matches this ReceiptInstructionAgent with a DeliveryInstructionAgent
        that has the same link code and creates a TransactionAgent."""

        #self.model.log_event(
        #    f"Instruction {self.uniqueID} attempting to match",
        #    self.uniqueID,
        #    is_transaction=True
        #)
        #new logging
        self.model.log_event(
            event_type="Attempting to Match",
            object_ids=[self.uniqueID],
            attributes={"status": self.status}
        )

        #old logging
        self.model.log_ocel_event(
            activity="Attempting to Match",
            object_refs=[{"object_id": self.uniqueID, "object_type": "ReceiptInstruction"}]
        )

        if self.status != "Validated":
            #self.model.log_event(
            #    f"Error: Instruction {self.uniqueID} in wrong state, cannot match",
            #    self.uniqueID,
            #    is_transaction=True,
            #)
            #new logging
            self.model.log_event(
                event_type="Matching Failed: Incorrect State",
                object_ids=[self.uniqueID],
                attributes={"status": self.status}
            )
            #old logging
            self.model.log_ocel_event(
                activity="Matching Failed: Incorrect State",
                object_refs=[{"object_id": self.uniqueID, "object_type": "ReceiptInstruction"}]
            )
            return None

        # Find a matching DeliveryInstructionAgent
        other_instruction = None
        for agent in self.model.agents:
            if (
                    isinstance(agent, DeliveryInstructionAgent.DeliveryInstructionAgent)  # Ensure it's a DeliveryInstructionAgent
                    and agent.linkcode == self.linkcode  # Check if linkcodes match
                    and agent.status == "Validated"  # Ensure the status is correct
            ):
                other_instruction = agent
                break
        else:
            #self.model.log_event(
            #    f"ERROR: ReceiptInstruction {self.uniqueID} failed to match, DeliveryInstruction not yet validated",
            #    self.uniqueID,
            #    is_transaction=True,
            #)
            #new logging
            self.model.log_event(
                event_type="Matching Failed: No Counter Instruction Found",
                object_ids=[self.uniqueID],
                attributes={"status": self.status}
            )

            #old logging
            self.model.log_ocel_event(
                activity="Matching Failed: No Counter Instruction Found",
                object_refs=[{"object_id": self.uniqueID, "object_type": "ReceiptInstruction"}]
            )
            return None

        # Create a transaction
        transaction = TransactionAgent.TransactionAgent(
            model=self.model,
            transactionID=f"trans{self.uniqueID}_{other_instruction.uniqueID}",
            deliverer=other_instruction,
            receiver=self,
            status="Matched",
        )
        self.model.register_transaction(transaction)

        # Link transaction to both instructions
        self.linkedTransaction = transaction
        other_instruction.linkedTransaction = transaction

        # Update status
        self.set_status("Matched")
        other_instruction.set_status("Matched")

        #self.model.log_event(
        #    f"ReceiptInstruction {self.uniqueID} matched with DeliveryInstruction {other_instruction.uniqueID}",
        #    self.uniqueID,
        #    is_transaction=True,
        #)
        #new logging
        self.model.log_event(
            event_type="Matched",
            object_ids=[self.uniqueID, other_instruction.uniqueID, transaction.transactionID],
            attributes={"status": "Matched"}
        )
        #old logging
        self.model.log_ocel_event(
            activity="Matched",
            object_refs=[
                {"object_id": other_instruction.uniqueID, "object_type": "DeliveryInstruction"},
                {"object_id": self.uniqueID, "object_type": "ReceiptInstruction"},
                {"object_id": transaction.transactionID, "object_type": "Transaction"}
            ]
        )
        return transaction

    def cancel_timout(self):
        if self.status == "Exists" or self.status == "Pending" or self.status == "Validated":
            self.status = "Cancelled due to timeout"
            self.model.agents.remove(self)
            # logging
            #self.model.log_event(f"Instruction {self.uniqueID} cancelled due to timeout.", self.uniqueID,
            #
            #
            #                     is_transaction=True)
            #new logging
            self.model.log_event(
                event_type="Cancelled due to timeout",
                object_ids=[self.uniqueID],
                attributes={"status": "Cancelled due to timeout"}
            )
            #old logging
            self.model.log_ocel_event(
                activity="Cancelled due to Timeout",
                object_refs=[{"object_id": self.uniqueID, "object_type": "ReceiptInstruction"}]
            )
        if self.status == "Matched":
            self.status = "Cancelled due to timeout"
            self.linkedTransaction.deliverer.set_status("Cancelled due to timeout")
            self.linkedTransaction.set_status("Cancelled due to timeout")
            #new logging
            self.model.log_event(
                event_type="Cancelled due to timeout",
                object_ids=[self.uniqueID,self.linkedTransaction.deliverer.uniqueID, self.linkedTransaction.transactionID],
                attributes={"status": "Cancelled due to timeout"}
            )
            #old logging
            self.model.log_ocel_event(
                activity="Cancelled due to Timeout",
                object_refs=[
                    {"object_id": self.uniqueID, "object_type": "ReceiptInstruction"},
                    {"object_id": self.linkedTransaction.transactionID, "object_type": "Transaction"}
                ]
            )

            self.model.remove_transaction(self.linkedTransaction)
            self.model.agents.remove(self.linkedTransaction.deliverer)
            self.model.agents.remove(self.linkedTransaction)
            self.model.agents.remove(self)

            # logging
           # self.model.log_event(f"ReceiptInstruction {self.uniqueID} cancelled due to timeout.", self.uniqueID, is_transaction=True)
           # self.model.log_event(f"DeliveryInstruction {self.linkedTransaction.deliverer.get_uniqueID()} cancelled due to timeout.", self.linkedTransaction.deliverer.get_uniqueID(), is_transaction=True)
           # self.model.log_event(f"Transaction {self.linkedTransaction.get_transactionID()} cancelled due to timeout.", self.linkedTransaction.get_transactionID(), is_transaction=True)


#