from datetime import datetime
from typing import TYPE_CHECKING, Optional
import InstructionAgent

if TYPE_CHECKING:
    from SettlementModel import SettlementModel
    from InstitutionAgent import InstitutionAgent
    from Account import Account
    from TransactionAgent import TransactionAgent

import ReceiptInstructionAgent
import TransactionAgent

class DeliveryInstructionAgent(InstructionAgent.InstructionAgent):
    def __init__(self, model: "SettlementModel", uniqueID: str, motherID: str, institution: "InstitutionAgent", securitiesAccount: "Account", cashAccount: "Account", securityType: str, amount: float, isChild: bool, status: str, linkcode: str, creation_time: datetime ,linkedTransaction: Optional["TransactionAgent"] = None, depth: int=0, original_mother_amount: int=None):
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
            depth=depth,
            original_mother_amount=original_mother_amount
        )

        self.model.log_object(
            object_id=self.uniqueID,
            object_type="DeliveryInstruction",
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
            event_type="DeliveryInstruction Created",
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

    def createDeliveryChildren(self):
        if self.depth >= self.model.max_child_depth:
            return(None, None)

        min_settlement_amount = self.original_mother_amount * self.model.min_settlement_percentage

        if self.securitiesAccount.getAccountType() != self.securityType:
            available_securities = 0
        else:
            available_securities = self.securitiesAccount.getBalance()

        receiver = self.linkedTransaction.receiver
        if receiver.cashAccount.getAccountType() != "Cash":
            available_cash = 0
        else:
            available_cash = receiver.cashAccount.getEffectiveAvailableCash()

        #takes the minimum of available securities of deliverer and available cash of seller and not more than the amount
        available_to_settle = min(self.amount, available_cash, available_securities)

        if available_to_settle > min_settlement_amount:
            #create delivery children instructions

            #instant matching and settlement of first child not yet possible, because receipt_child_1 does not yet exist
            delivery_child_1 = DeliveryInstructionAgent(self.model, f"{self.uniqueID}_1", self.uniqueID,
                                                self.institution, self.securitiesAccount, self.cashAccount,
                                                self.securityType, available_to_settle, True, "Validated", f"{self.linkcode}_1", self.model.simulated_time, None, depth = self.depth +1, original_mother_amount=self.original_mother_amount
                                                )
            delivery_child_2 = DeliveryInstructionAgent(self.model, f"{self.uniqueID}_2", self.uniqueID,
                                                self.institution, self.securitiesAccount, self.cashAccount,
                                                self.securityType, self.amount - available_to_settle, True, "Validated", f"{self.linkcode}_2", self.model.simulated_time, None, depth = self.depth +1, original_mother_amount=self.original_mother_amount
                                                )

            #add children to fast lookup list
            if delivery_child_1.linkcode not in self.model.validated_delivery_instructions:
                self.model.validated_delivery_instructions[delivery_child_1.get_linkcode] = []
            self.model.validated_delivery_instructions[delivery_child_1.get_linkcode].append(delivery_child_1)

            if delivery_child_2.linkcode not in self.model.validated_delivery_instructions:
                self.model.validated_delivery_instructions[delivery_child_2.get_linkcode] = []
            self.model.validated_delivery_instructions[delivery_child_2.get_linkcode].append(delivery_child_2)

            #pass intended settlement time of mother to the children
            delivery_child_1.set_intended_settlement_date(self.get_intended_settlement_date())
            delivery_child_2.set_intended_settlement_date(self.get_intended_settlement_date())

            #add child instructions to the model
            self.model.agents.add(delivery_child_1)
            self.model.agents.add(delivery_child_2)
            self.model.instructions.append(delivery_child_1)
            self.model.instructions.append(delivery_child_2)
            self.model.log_event(
                event_type="Delivery Children Created",
                object_ids=[self.uniqueID],
                attributes={"parentInstructionID": self.uniqueID,
                            "parent_depth": self.depth,
                            "child1_depth": self.depth +1,
                            "child2_depth": self.depth +1
                            }
            )
            return (delivery_child_1, delivery_child_2)
        else:
            # self.model.log_event(
            #     event_type="Partial Settlement Failed: Insufficient Funds",
            #     object_ids=[self.uniqueID],
            #     attributes={"status": self.status}
            # )
            return (None, None)

    def match(self):
        """Matches this DeliveryInstructionAgent with a ReceiptInstructionAgent
        that has the same link code and creates a TransactionAgent."""
        #new logging
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

        # Find a matching ReceiptInstructionAgent
        other_instruction = None
        if self.linkcode in self.model.validated_receipt_instructions:
            receipt_candidates = self.model.validated_receipt_instructions[self.linkcode]
            for candidate in receipt_candidates:
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
            deliverer=self,
            receiver=other_instruction,
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
        if self.status == "Exists" or self.status == "Pending" or self.status == "Validated":
            self.status = "Cancelled due to timeout"
            self.model.agents.remove(self)

            #new logging
            self.model.log_event(
                event_type="Cancelled due to timeout",
                object_ids=[self.uniqueID],
                attributes={"status": "Cancelled due to timeout"}
            )


            #self.model.log_event(f"DeliveryInstruction {self.uniqueID} cancelled due to timeout.", self.uniqueID, is_transaction=True)
        if self.status == "Matched":
            self.status = "Cancelled due to timeout"
            self.linkedTransaction.receiver.set_status("Cancelled due to timeout")
            self.linkedTransaction.set_status("Cancelled due to timeout")

            #new logging
            self.model.log_event(
                event_type="Cancelled due to timeout",
                object_ids=[self.uniqueID, self.linkedTransaction.receiver.uniqueID, self.linkedTransaction.transactionID],
                attributes={"status": "Cancelled due to timeout"}
            )

            self.model.remove_transaction(self.linkedTransaction)
            self.model.agents.remove(self.linkedTransaction.receiver)
            self.model.agents.remove(self.linkedTransaction)
            self.model.agents.remove(self)

