from datetime import datetime
from typing import TYPE_CHECKING, Optional
import InstructionAgent

if TYPE_CHECKING:
    from SettlementModel import SettlementModel
    from InstitutionAgent import InstitutionAgent
    from Account import Account
    from TransactionAgent import TransactionAgent

import TransactionAgent

class ReceiptInstructionAgent(InstructionAgent.InstructionAgent):
    def __init__(self, model: "SettlementModel", uniqueID: str, motherID: str, institution: "InstitutionAgent",
                 securitiesAccount: "Account", cashAccount: "Account", securityType: str, amount: float, isChild: bool,
                 status: str, linkcode: str, creation_time: datetime, original_creation_time: datetime, linkedTransaction: Optional["TransactionAgent"] = None, depth: int=0, original_mother_amount: int=None):
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
            creation_time=creation_time,
            original_creation_time=original_creation_time,
            depth=depth,
            original_mother_amount=original_mother_amount
        )

        self.model.log_object(
            object_id=self.uniqueID,
            object_type="ReceiptInstruction",
            attributes={
                "securityType": self.securityType,
                "amount": self.amount,
                "status": self.status,
                "linkcode": self.linkcode,
                "institutionID": self.institution.institutionID,
                "depth": self.depth,
                "motherID": self.motherID
            }
        )
        self.model.log_event(
            event_type="ReceiptInstruction Initialised",
            object_ids=[self.uniqueID, self.institution.institutionID, self.securitiesAccount.getAccountID()],
            attributes={
                "securityType": self.securityType,
                "amount": self.amount,
                "status": self.status,
                "linkcode": self.linkcode,
                "depth": self.depth,
                "motherID": self.motherID
            }
        )

    def get_creation_time(self):
        return self.creation_time

    def createReceiptChildren(self):
        # Creates the delivery children when partial settlement is triggered
        if self.depth >= self.model.max_child_depth:
            return (None, None)

        min_settlement_amount = self.original_mother_amount * self.model.min_settlement_percentage

        if self.cashAccount.getAccountType() != "Cash":
            available_cash = 0
        else:
            available_cash = self.cashAccount.getEffectiveAvailableCash()

        deliverer = self.linkedTransaction.deliverer
        if deliverer.securitiesAccount.getAccountType() != self.securityType:
            available_securities = 0
        else:
            available_securities = deliverer.securitiesAccount.getBalance()

        # Compute the amount that can actually be settled.
        available_to_settle = min(self.amount, available_cash, available_securities)

        if available_to_settle > min_settlement_amount:
            #  Create receipt child instructions with the computed amounts.
            receipt_child_1 = ReceiptInstructionAgent(
                            self.model,f"{self.uniqueID}_1", self.uniqueID, self.institution,
                            self.securitiesAccount, self.cashAccount, self.securityType, available_to_settle,
                            True,"Validated",f"{self.linkcode}_1", self.original_creation_time, self.original_creation_time,
                            linkedTransaction=None, depth = self.depth + 1, original_mother_amount=self.original_mother_amount
                                )

            receipt_child_2 = ReceiptInstructionAgent(
                            self.model,f"{self.uniqueID}_2", self.uniqueID, self.institution,
                             self.securitiesAccount, self.cashAccount, self.securityType, self.amount - available_to_settle,
                                 True,"Validated",f"{self.linkcode}_2", self.original_creation_time, self.original_creation_time,
                               linkedTransaction=None, depth = self.depth + 1, original_mother_amount=self.original_mother_amount
                                )

            # add children to fast lookup list
            if receipt_child_1.get_linkcode() not in self.model.validated_receipt_instructions:
                self.model.validated_receipt_instructions[receipt_child_1.get_linkcode()] = []
            self.model.validated_receipt_instructions[receipt_child_1.get_linkcode()].append(receipt_child_1)

            if receipt_child_2.get_linkcode() not in self.model.validated_receipt_instructions:
                self.model.validated_receipt_instructions[receipt_child_2.get_linkcode()] = []
            self.model.validated_receipt_instructions[receipt_child_2.get_linkcode()].append(receipt_child_2)

            #pass intended settlement time of mother to the children
            receipt_child_1.set_intended_settlement_date(self.get_intended_settlement_date())
            receipt_child_2.set_intended_settlement_date(self.get_intended_settlement_date())
            receipt_child_1.set_priority(self.get_priority())
            receipt_child_2.set_priority(self.get_priority())


            # Add the new child instructions to the agents scheduler.
            self.model.agents.add(receipt_child_1)
            self.model.agents.add(receipt_child_2)

            # Add agents to the instruction list
            self.model.instructions.append(receipt_child_1)
            self.model.instructions.append(receipt_child_2)
            self.model.log_event(
                event_type="Receipt Children Created",
                object_ids=[self.uniqueID, receipt_child_1.uniqueID, receipt_child_2.uniqueID],
                attributes={"parentInstructionID": self.uniqueID,
                            "parent_depth": self.depth,
                            "child1_depth": self.depth + 1,
                            "child2_depth": self.depth + 1,
                            "parent_status": self.status,
                            "child1_status": receipt_child_1.get_status(),
                            "child2_status": receipt_child_2.get_status()
                            }
            )
            return (receipt_child_1, receipt_child_2)
        else:
            return (None, None)

    def match(self):
        # Matches this ReceiptInstructionAgent with a DeliveryInstructionAgent that has the same link code and creates a TransactionAgent
        self.model.log_event(
            event_type="Attempting to Match",
            object_ids=[self.uniqueID],
            attributes={"status": self.status}
        )

        if self.status != "Validated":
            self.model.log_event(
                event_type="Matching Failed: Incorrect State",
                object_ids=[self.uniqueID],
                attributes={"status": self.status}
            )
            return None

        # Find a matching DeliveryInstructionAgent
        other_instruction = None
        if self.linkcode in self.model.validated_delivery_instructions:
            delivery_candidates = self.model.validated_delivery_instructions[self.linkcode]
            for candidate in delivery_candidates:
                if candidate.status == "Validated":
                    other_instruction = candidate
                    break

        if not other_instruction:
            # Logging for failed match
            self.model.log_event(
                event_type="Matching Failed: No Counter Instruction Found",
                object_ids=[self.uniqueID],
                attributes={"status": self.status}
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

        self.model.log_event(
            event_type="Matched",
            object_ids=[self.uniqueID, other_instruction.uniqueID, transaction.transactionID],
            attributes={"status": "Matched"}
        )
        return transaction

    def cancel_timeout(self):
        # Implementation for when a DeliveryInstruction times out
        if self.status == "Exists" or self.status == "Pending" or self.status == "Validated":
            self.set_status("Cancelled due to timeout")
            self.model.agents.remove(self)

            self.model.log_event(
                event_type="Cancelled due to timeout",
                object_ids=[self.uniqueID],
                attributes={"status": "Cancelled due to timeout"}
            )
        if self.status == "Matched":
            self.set_status("Cancelled due to timeout")
            self.linkedTransaction.deliverer.set_status("Cancelled due to timeout")
            self.linkedTransaction.set_status("Cancelled due to timeout")
            self.model.log_event(
                event_type="Cancelled due to timeout",
                object_ids=[self.uniqueID,self.linkedTransaction.deliverer.uniqueID, self.linkedTransaction.transactionID],
                attributes={"status": "Cancelled due to timeout"}
            )
            self.model.remove_transaction(self.linkedTransaction)
            self.model.agents.remove(self.linkedTransaction.deliverer)
            self.model.agents.remove(self.linkedTransaction)
            self.model.agents.remove(self)


