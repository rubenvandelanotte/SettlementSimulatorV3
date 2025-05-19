
from mesa import Agent
from typing import TYPE_CHECKING
from datetime import datetime, timedelta
import math

if TYPE_CHECKING:
    from ReceiptInstructionAgent import ReceiptInstructionAgent
    from DeliveryInstructionAgent import DeliveryInstructionAgent

class TransactionAgent(Agent):
    def __init__(self, model, transactionID: str, deliverer: "DeliveryInstructionAgent", receiver: "ReceiptInstructionAgent", status: str):
        super().__init__(model)
        self.transactionID = transactionID
        self.deliverer = deliverer
        self.receiver = receiver
        self.retry_count = 0
        if self.deliverer.get_amount() != self.receiver.get_amount():
            self.status = "Cancelled due to error"
            self.model.log_event(
                event_type="Transaction Creation Failed: Instruction Amount Mismatch",
                object_ids=[self.deliverer.uniqueID, self.receiver.uniqueID],
                attributes={
                    "deliverer_amount": self.deliverer.get_amount(),
                    "receiver_amount": self.receiver.get_amount(),
                    "status": self.status
                }
            )
            return
        self.status = status
        self.model.log_object(
            object_id=self.transactionID,
            object_type="Transaction",
            attributes={
                "status": self.status,
                "delivererID": self.deliverer.uniqueID,
                "receiverID": self.receiver.uniqueID
            }
        )

    def get_status(self):
        return self.status

    def set_status(self, new_status: str):
        self.status = new_status

    def get_deliverer(self):
        return self.deliverer

    def meets_selection_criteria(self):
        # Checks if transaction meets selection criteria
        isd_close = (self.model.get_simulated_time() >= self.get_deliverer().get_intended_settlement_date() - timedelta(days=1, minutes=35))
        is_matched = (self.status == "Matched")
        is_selected = (isd_close and is_matched)
        return is_selected

    def settle(self):
        # Settles transaction
        self.retry_count += 1
        self.model.log_event(
            event_type="Attempting to Settle",
            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
            attributes={"status": self.status, "retry": self.retry_count}
        )

        if self.deliverer.get_status() in ["Matched"] and self.receiver.get_status() in ["Matched"] and self.status in ["Matched"]:

            # Checks if transaction can be settled fully
            if not (
                self.deliverer.securitiesAccount.checkBalance(self.deliverer.get_amount(), self.deliverer.get_securityType()) and
                self.receiver.cashAccount.checkBalance(self.receiver.get_amount(), "Cash")
            ):
                # Partial settlement
                if self.deliverer.get_institution().check_partial_allowed() and self.receiver.get_institution().check_partial_allowed():
                    delivery_children = self.deliverer.createDeliveryChildren()
                    receipt_children = self.receiver.createReceiptChildren()

                    delivery_child_1, delivery_child_2 = delivery_children
                    receipt_child_1, receipt_child_2 = receipt_children

                    if None in (delivery_child_1, delivery_child_2, receipt_child_1, receipt_child_2):
                        for child in [delivery_child_1, delivery_child_2, receipt_child_1, receipt_child_2]:
                            if child is not None:
                                # Remove from model agent list and instruction list
                                if child in self.model.agents:
                                    self.model.agents.remove(child)
                                if child in self.model.instructions:
                                    self.model.instructions.remove(child)
                                # Remove from validated instruction dict
                                linkcode_dict = (
                                    self.model.validated_delivery_instructions
                                    if isinstance(child, DeliveryInstructionAgent.DeliveryInstructionAgent)
                                    else self.model.validated_receipt_instructions
                                )
                                if child.get_linkcode() in linkcode_dict:
                                    if child in linkcode_dict[child.get_linkcode()]:
                                        linkcode_dict[child.get_linkcode()].remove(child)

                        self.model.log_event(
                            event_type="Partial Settlement Aborted",
                            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                            attributes={"status": self.status}
                        )
                        return

                    # Checks if child amounts are equal
                    min_amount_1 = min(delivery_child_1.get_amount(), receipt_child_1.get_amount())
                    delivery_child_1.set_amount(min_amount_1)
                    receipt_child_1.set_amount(min_amount_1)

                    # Ensure that sum of amounts of 2 child transactions equals parent that will be cancelled
                    min_amount_2 = self.deliverer.get_amount() - min_amount_1
                    delivery_child_2.set_amount(min_amount_2)
                    receipt_child_2.set_amount(min_amount_2)

                    # Create child transaction 1
                    child_transaction_1 = TransactionAgent(
                        model=self.model,
                        transactionID=f"trans{delivery_child_1.uniqueID}_{receipt_child_1.uniqueID}",
                        deliverer=delivery_child_1,
                        receiver=receipt_child_1,
                        status="Matched"
                    )
                    self.model.register_transaction(child_transaction_1)
                    delivery_child_1.set_linkedTransaction(child_transaction_1)
                    receipt_child_1.set_linkedTransaction(child_transaction_1)
                    delivery_child_1.set_status("Matched")
                    receipt_child_1.set_status("Matched")

                    # Create child transaction 2
                    child_transaction_2 = TransactionAgent(
                        model=self.model,
                        transactionID=f"trans{delivery_child_2.uniqueID}_{receipt_child_2.uniqueID}",
                        deliverer=delivery_child_2,
                        receiver=receipt_child_2,
                        status="Matched"
                    )
                    self.model.register_transaction(child_transaction_2)
                    delivery_child_2.set_linkedTransaction(child_transaction_2)
                    receipt_child_2.set_linkedTransaction(child_transaction_2)
                    delivery_child_2.set_status("Matched")
                    receipt_child_2.set_status("Matched")

                    #settle first child
                    child_transaction_1.settle()

                    #cancel the mother transaction
                    self.cancel_partial()
                    self.deliverer.get_securitiesAccount().set_newSecurities(False)
                    return

                self.deliverer.get_securitiesAccount().set_newSecurities(False)
                self.model.log_event(
                    event_type= "Settlement Failed: Insufficient Funds",
                    object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                    attributes={"status": self.status, "retry": self.retry_count}
                )
                return

            # Set new cash and securities balances
            delivered_securities = self.deliverer.securitiesAccount.deductBalance(
                self.deliverer.get_amount(), self.deliverer.get_securityType())
            received_securities = self.receiver.securitiesAccount.addBalance(
                self.receiver.get_amount(), self.deliverer.get_securityType())

            delivered_cash = self.receiver.cashAccount.deductBalance(
                self.receiver.get_amount(), "Cash")
            received_cash = self.deliverer.cashAccount.addBalance(
                self.deliverer.get_amount(), "Cash")

            if not (delivered_securities == received_securities == delivered_cash == received_cash == self.deliverer.get_amount()):
                self.model.log_event(
                    event_type="Settlement Failed: Insufficient Funds",
                    object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                    attributes={"status": self.status, "retry": self.retry_count}
                )
                return

            #check isd of both deliverer and receiver are close by => should be the same so should be an and
            if self.deliverer.get_intended_settlement_date() < self.model.get_simulated_time() and self.receiver.get_intended_settlement_date() < self.model.get_simulated_time():
                self.deliverer.set_status("Settled late")
                self.receiver.set_status("Settled late")
                self.status = "Settled late"
                label = "Settled Late"
                lateness_seconds = (self.model.get_simulated_time() - self.deliverer.get_intended_settlement_date()).total_seconds()
                lateness_hours = math.ceil(lateness_seconds / 3600)
                depth = max(self.deliverer.get_depth(), self.receiver.get_depth())
                is_child = self.deliverer.isChild
            else:
                self.deliverer.set_status("Settled on time")
                self.receiver.set_status("Settled on time")
                self.status = "Settled on time"
                label = "Settled On Time"
                lateness_hours = 0
                depth = max(self.deliverer.get_depth(), self.receiver.get_depth())
                is_child = self.deliverer.isChild

            self.model.log_event(
                event_type=label,
                object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                attributes={"status": self.status, "lateness_hours": lateness_hours, "depth": depth, "is_child": is_child, "amount": self.deliverer.get_amount()}
            )
            self.model.remove_transaction(self)
            self.model.agents.remove(self.deliverer)
            self.model.agents.remove(self.receiver)
            self.model.agents.remove(self)

            self.receiver.get_securitiesAccount().set_newSecurities(True)

    def step(self):
        # Step method for transactions
        if self.deliverer.is_instruction_time_out():
            self.deliverer.cancel_timeout()
        elif self.receiver.is_instruction_time_out():
            self.receiver.cancel_timeout()
        elif self.status in ["Matched"]:
            if (self.get_deliverer().get_securitiesAccount().get_newSecurities()):
                if self.model.get_simulated_time() >= self.get_deliverer().get_intended_settlement_date() - timedelta(days=1,minutes=35):
                    self.settle()

    def cancel_partial(self):
        # Cancel transaction and instructions due to partial settlement
        self.status = "Cancelled due to partial settlement"
        self.deliverer.set_status("Cancelled due to partial settlement")
        self.receiver.set_status("Cancelled due to partial settlement")
        self.model.log_event(
            event_type="Cancelled due to Partial Settlement",
            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
            attributes={"status": self.status}
        )
        self.model.partial_cancelled_count += 1
        self.model.remove_transaction(self)
        self.model.agents.remove(self.deliverer)
        self.model.agents.remove(self.receiver)
        self.model.agents.remove(self)
