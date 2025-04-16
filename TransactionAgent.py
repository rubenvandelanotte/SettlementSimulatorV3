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

        #new logging (I think this one should be removed)
        self.model.log_event(
            event_type="Transaction Created",
            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
            attributes={"status": self.status}
        )

    def get_transactionID(self):
        return self.transactionID

    def get_status(self):
        return self.status

    def set_status(self, new_status: str):
        self.status = new_status

    # --- Final Merged and Optimized settle() Method ---
    def settle(self):
        # Unified logging - new format
        self.model.log_event(
            event_type="Attempting to Settle",
            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
            attributes={"status": self.status}
        )


        if self.deliverer.get_status() == "Matched" and self.receiver.get_status() == "Matched" and self.status == "Matched":

            if (self.deliverer.securitiesAccount.checkBalance(self.deliverer.get_amount(),
                                                              self.deliverer.get_securityType())
                    and self.receiver.cashAccount.checkBalance(self.receiver.get_amount(), "Cash")):

                if self.deliverer.get_amount() == self.receiver.get_amount():
                    # Transfers
                    delivered_securities = self.deliverer.securitiesAccount.deductBalance(
                        self.deliverer.get_amount(), self.deliverer.get_securityType())
                    received_securities = self.receiver.securitiesAccount.addBalance(
                        self.receiver.get_amount(), self.deliverer.get_securityType())

                    delivered_cash = self.receiver.cashAccount.deductBalance(
                        self.receiver.get_amount(), "Cash")
                    received_cash = self.deliverer.cashAccount.addBalance(
                        self.deliverer.get_amount(), "Cash")

                    if not (
                            delivered_securities == received_securities ==
                            delivered_cash == received_cash ==
                            self.deliverer.get_amount() == self.receiver.get_amount()
                    ):
                        self.deliverer.set_status("Cancelled due to error")
                        self.receiver.set_status("Cancelled due to error")
                        self.status = "Cancelled due to error"

                        self.model.log_event(
                            event_type="Cancelled due to error",
                            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                            attributes={"status": self.status}
                        )


                    if self.status != "Cancelled due to error":
                        # Late settlement check
                        if (self.deliverer.get_intended_settlement_date() < self.model.simulated_time or
                                self.receiver.get_intended_settlement_date() < self.model.simulated_time):
                            self.deliverer.set_status("Settled late")
                            self.receiver.set_status("Settled late")
                            self.status = "Settled late"
                            label = "Settled Late"

                            # calculate lateness
                            lateness_seconds = (
                                        self.model.simulated_time - self.deliverer.get_intended_settlement_date()).total_seconds()
                            lateness_hours = math.ceil(lateness_seconds / 3600)  # Convert to hours

                        else:
                            self.deliverer.set_status("Settled on time")
                            self.receiver.set_status("Settled on time")
                            self.status = "Settled on time"
                            label = "Settled On Time"
                            lateness_hours = 0


                        self.model.log_event(
                            event_type=label,
                            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                            attributes={"status": self.status, "lateness_hours": lateness_hours}
                        )



                        # Clean up
                        self.model.remove_transaction(self)
                        self.model.agents.remove(self.deliverer)
                        self.model.agents.remove(self.receiver)
                        self.model.agents.remove(self)

            elif self.deliverer.get_amount() == 0 or self.receiver.get_amount() == 0:
                self.model.log_event(
                    event_type="Settlement Failed: Insufficient Funds",
                    object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                    attributes={"status": self.status}
                )


            else:
                if self.deliverer.get_institution().check_partial_allowed() and self.receiver.get_institution().check_partial_allowed():
                    delivery_children = self.deliverer.createDeliveryChildren()
                    receipt_children = self.receiver.createReceiptChildren()

                    if receipt_children == (None, None) or delivery_children == (None, None):
                        self.model.log_event(
                            event_type="Partial Settlement Aborted",
                            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                            attributes={"status": self.status}
                        )

                    else:
                        #new implementation:
                        delivery_child_1, delivery_child_2 = delivery_children
                        receipt_child_1, receipt_child_2 = receipt_children

                        child_transaction_1 = TransactionAgent(
                            model=self.model,
                            transactionID=f"trans{delivery_child_1.uniqueID}_{receipt_child_1.uniqueID}",
                            deliverer=delivery_child_1,
                            receiver=receipt_child_1,
                            status="Matched"
                        )
                        self.model.register_transaction(child_transaction_1)

                        # Link transaction to both instructions
                        delivery_child_1.linkedTransaction = child_transaction_1
                        receipt_child_1.linkedTransaction = child_transaction_1

                        # Update statuses
                        delivery_child_1.set_status("Matched")
                        receipt_child_1.set_status("Matched")

                        child_transaction_2 = TransactionAgent(
                            model=self.model,
                            transactionID=f"trans{delivery_child_2.uniqueID}_{receipt_child_2.uniqueID}",
                            deliverer=delivery_child_2,
                            receiver=receipt_child_2,
                            status="Matched"
                        )
                        self.model.register_transaction(child_transaction_2)

                        # Link transaction to both instructions
                        delivery_child_2.linkedTransaction = child_transaction_2
                        receipt_child_2.linkedTransaction = child_transaction_2

                        # Update statuses
                        delivery_child_2.set_status("Matched")
                        receipt_child_2.set_status("Matched")

                        if child_transaction_1:
                            child_transaction_1.settle()
                        if child_transaction_2:
                            child_transaction_2.deliverer.get_securitiesAccount().set_newSecurities(False)
                            child_transaction_2.receiver.get_cashAccount().set_newSecurities(False)

                        self.cancel_partial()


                        #old implementation:
                        # delivery_child_1, delivery_child_2 = delivery_children
                        # receipt_child_1, receipt_child_2 = receipt_children
                        #
                        # child_transaction_1 = delivery_child_1.match()
                        # child_transaction_2 = delivery_child_2.match()
                        #
                        # if child_transaction_1:
                        #     child_transaction_1.settle()
                        # if child_transaction_2:
                        #     child_transaction_2.deliverer.get_securitiesAccount().set_newSecurities(False)
                        #     child_transaction_2.receiver.get_cashAccount().set_newSecurities(False)
                        #
                        # if child_transaction_1 or child_transaction_2:
                        #     self.cancel_partial()
                        # else:
                        #     print("[WARNING] Partial settlement failed: no children succeeded.")

        else:
            self.model.log_event(
                event_type="Settlement Failed: Incorrect Status",
                object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                attributes={"status": self.status}
            )


        # Reset flags
        self.deliverer.get_securitiesAccount().set_newSecurities(False)
        self.receiver.get_cashAccount().set_newSecurities(False)

    def step(self):
        #time_of_day = self.model.simulated_time.time()
        #if not (self.deliverer.get_securitiesAccount().get_newSecurities() == True or
        #    self.receiver.get_cashAccount().get_newSecurities() == True ):
            #if no new securities or cash where added to an account, no settlement gets tried
        #    return

        if self.deliverer.is_instruction_time_out():
            self.deliverer.cancel_timeout()
        elif self.receiver.is_instruction_time_out():
            self.receiver.cancel_timeout()

            #No longer settled only during mini batches
        #elif self.status not in ["Cancelled due to timeout","Cancelled due to partial settlement", "Settled late", "Settled on time", "Cancelled due to error"]:
         #   if self.model.trading_start <= timedelta(hours=time_of_day.hour, minutes=time_of_day.minute) <= self.model.trading_end:
         #       self.settle()
        #self.model.simulated_time = self.model.simulated_time + timedelta(seconds=1)

    def cancel_partial(self):
        self.status = "Cancelled due to partial settlement"
        self.deliverer.set_status("Cancelled due to partial settlement")
        self.receiver.set_status("Cancelled due to partial settlement")

        #new logging
        self.model.log_event(
            event_type="Cancelled due to Partial Settlement",
            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
            attributes={"status": self.status}
        )



        #remove transition and instructions from the model when cancelled
        self.model.partial_cancelled_count += 1
        self.model.remove_transaction(self)
        self.model.agents.remove(self.deliverer)
        self.model.agents.remove(self.receiver)
        self.model.agents.remove(self)




