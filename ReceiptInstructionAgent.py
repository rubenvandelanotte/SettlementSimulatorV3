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
                 status: str, linkcode: str, creation_time: datetime, linkedTransaction: Optional["TransactionAgent"] = None, depth: int=0):
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
            depth=depth
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
            event_type="ReceiptInstruction Created",
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

        if self.depth >= self.model.MAX_CHILD_DEPTH:
            return (None, None)

        MIN_SETTLEMENT_AMOUNT = self.model.min_settlement_amount
        # Calculate the actual available amounts using getBalance(), ensuring correct account types.
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

        if available_to_settle > MIN_SETTLEMENT_AMOUNT:
            #  Create receipt child instructions with the computed amounts.
            receipt_child_1 = ReceiptInstructionAgent(
                                      self.model,f"{self.uniqueID}_1", self.uniqueID, self.institution, self.securitiesAccount, self.cashAccount, self.securityType, available_to_settle,True,"Validated",f"{self.linkcode}_1", creation_time=self.model.simulated_time, linkedTransaction=None, depth = self.depth + 1
                                        )

            receipt_child_2 = ReceiptInstructionAgent(
                            self.model,f"{self.uniqueID}_2", self.uniqueID, self.institution,
                             self.securitiesAccount, self.cashAccount, self.securityType, self.amount - available_to_settle,
                                 True,"Validated",f"{self.linkcode}_2", creation_time=self.model.simulated_time,
                               linkedTransaction=None, depth = self.depth + 1
                                )



            #ensures that the intended_settlement_time of children = mother
            receipt_child_1.set_intended_settlement_date(self.get_intended_settlement_date())
            receipt_child_2.set_intended_settlement_date(self.get_intended_settlement_date())

            # Add the new child instructions to the agents scheduler.
            self.model.agents.add(receipt_child_1)
            self.model.agents.add(receipt_child_2)

            #add agents to the instruction list
            self.model.instructions.append(receipt_child_1)
            self.model.instructions.append(receipt_child_2)
            self.model.log_event(
                event_type="Receipt Children Created",
                object_ids=[receipt_child_1.uniqueID, receipt_child_2.uniqueID, self.uniqueID],
                attributes={"parentInstructionID": self.uniqueID,
                            "parent_depth": self.depth,
                            "child1_depth": self.depth + 1,
                            "child2_depth": self.depth + 1
                            }
            )
            return receipt_child_1, receipt_child_2
        else:

            #new logging
            self.model.log_event(
                event_type="Partial Settlement Failed: Insufficient Funds",
                object_ids=[self.uniqueID],
                attributes={"status": self.status}
            )



            return (None, None)

    def match(self):
        """Matches this ReceiptInstructionAgent with a DeliveryInstructionAgent
        that has the same link code and creates a TransactionAgent."""


        #new logging
        self.model.log_event(
            event_type="Attempting to Match",
            object_ids=[self.uniqueID],
            attributes={"status": self.status}
        )



        if self.status != "Validated":

            #new logging
            self.model.log_event(
                event_type="Matching Failed: Incorrect State",
                object_ids=[self.uniqueID],
                attributes={"status": self.status}
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

            #new logging
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


        #new logging
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


            self.model.remove_transaction(self.linkedTransaction)
            self.model.agents.remove(self.linkedTransaction.deliverer)
            self.model.agents.remove(self.linkedTransaction)
            self.model.agents.remove(self)


