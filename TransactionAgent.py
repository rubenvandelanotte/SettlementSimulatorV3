from mesa import Agent
from typing import TYPE_CHECKING
from datetime import datetime, timedelta

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

        #logging ( don't know why is_transaction = True)
        # self.model.log_event(f"Transaction {self.transactionID} created from account {self.deliverer.get_securitiesAccount().getAccountID()} to account {self.receiver.get_cashAccount().getAccountID()}", self.transactionID, is_transaction = True)

        self.model.log_object(
            object_id=self.transactionID,
            object_type="Transaction",
            attributes={
                "status": self.status,
                "delivererID": self.deliverer.uniqueID,
                "receiverID": self.receiver.uniqueID
            }
        )

        self.model.log_event(
            event_type="transaction_created",
            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
            attributes={"status": self.status}
        )

    def get_transactionID(self):
        return self.transactionID

    def get_status(self):
        return self.status

    def set_status(self, new_status: str):
        self.status = new_status

    def settle(self):
        #logging
        #self.model.log_event(f"Transaction {self.transactionID} attempting to settle.", self.transactionID, is_transaction = True)
        #new logging
        self.model.log_event(
            event_type="transaction_attempting_to_settle",
            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
            attributes={"status": self.status}
        )

        #old logging
        self.model.log_ocel_event(
            activity="Attempting to Settle",
            object_refs=[
                {"object_id": self.transactionID, "object_type": "Transaction"},
                {"object_id": self.deliverer.uniqueID, "object_type": "DeliveryInstruction"},
                {"object_id": self.receiver.uniqueID, "object_type": "ReceiptInstruction"}
            ]
        )

        if self.deliverer.get_status() == "Matched" and self.receiver.get_status() == "Matched" and self.status == "Matched":
            if (self.deliverer.securitiesAccount.checkBalance(self.deliverer.get_amount(), self.deliverer.get_securityType())
                    and self.receiver.cashAccount.checkBalance(self.receiver.get_amount(), "Cash")
            ):
                if self.deliverer.get_amount() == self.receiver.get_amount():
                    #additional check that to be settled amounts are equal

                    #transfer of securities
                    delivered_securities = self.deliverer.securitiesAccount.deductBalance(self.deliverer.get_amount(), self.deliverer.get_securityType())
                    received_securities = self.receiver.securitiesAccount.addBalance(self.receiver.get_amount(), self.deliverer.get_securityType())

                    #transfer of cash
                    delivered_cash = self.receiver.cashAccount.deductBalance(self.receiver.get_amount(), "Cash")
                    received_cash = self.deliverer.cashAccount.addBalance(self.deliverer.get_amount(), "Cash")

                    #extra check for safety
                    if not delivered_securities == received_securities == delivered_cash == received_cash == self.deliverer.get_amount() == self.receiver.get_amount():
                        self.deliverer.set_status("Cancelled due to error")
                        self.receiver.set_status("Cancelled due to error")
                        self.status = "Cancelled due to error"

                        #new logging
                        self.model.log_event(
                            event_type="transaction_cancelled_error",
                            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                            attributes={"status": self.status}
                        )

                        #old logging
                        self.model.log_ocel_event(
                            activity="Cancelled due to error",
                            object_refs=[
                                {"object_id": self.transactionID, "object_type": "Transaction"},
                                {"object_id": self.deliverer.uniqueID, "object_type": "DeliveryInstruction"},
                                {"object_id": self.receiver.uniqueID, "object_type": "ReceiptInstruction"}
                            ]
                        )
                    if self.status != "Cancelled due to error":
                        #checks if settled on time (T+2)
                        if self.deliverer.get_creation_time() + timedelta(hours = 48) < self.model.simulated_time or self.receiver.get_creation_time() + timedelta(hours = 48) < self.model.simulated_time:
                            self.deliverer.set_status("Settled late")
                            self.receiver.set_status("Settled late")
                            self.status = "Settled late"
                            # logging
                            #self.model.log_event(f"Transaction {self.transactionID} settled fully late.", self.transactionID,
                            #                     is_transaction=True)
                            #new logging

                            self.model.log_event(
                                event_type="transaction_settled_late",
                                object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                                attributes={"status": self.status}
                            )

                            self.model.log_ocel_event(
                                activity="Settled Late",
                                object_refs=[
                                    {"object_id": self.transactionID, "object_type": "Transaction"},
                                    {"object_id": self.deliverer.uniqueID, "object_type": "DeliveryInstruction"},
                                    {"object_id": self.receiver.uniqueID, "object_type": "ReceiptInstruction"}
                                ]
                            )
                        else:
                            self.deliverer.set_status("Settled on time")
                            self.receiver.set_status("Settled on time")
                            self.status = "Settled on time"
                            # logging
                            #self.model.log_event(f"Transaction {self.transactionID} settled fully on time.",
                            #                     self.transactionID,
                                               #  is_transaction=True)

                            #new logging

                            self.model.log_event(
                                event_type="transaction_settled_on_time",
                                object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                                attributes={"status": self.status}
                            )


                            #old logging
                            self.model.log_ocel_event(
                            activity = "Settled On Time",
                            object_refs = [
                                {"object_id": self.transactionID, "object_type": "Transaction"},
                                {"object_id": self.deliverer.uniqueID, "object_type": "DeliveryInstruction"},
                                {"object_id": self.receiver.uniqueID, "object_type": "ReceiptInstruction"}
                                ]
                            )





                    #remove the transaction and instructions from the model if fully settled or cancelled due to error
                    self.model.remove_transaction(self)
                    self.model.agents.remove(self.deliverer)
                    self.model.agents.remove(self.receiver)
                    self.model.agents.remove(self)


            elif self.deliverer.get_amount() == 0 or self.receiver.get_amount() == 0:
                #will do nothing if there is no cash or securities available

                #new logging
                self.model.log_event(
                    event_type="transaction_settlement_failed_due_to_insufficient_funds",
                    object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
                    attributes={"status": self.status}
                )

                #old logging
                self.model.log_ocel_event(
                    activity="Settlement Failed: Insufficient Funds",
                    object_refs=[
                        {"object_id": self.transactionID, "object_type": "Transaction"},
                        {"object_id": self.deliverer.uniqueID, "object_type": "DeliveryInstruction"},
                        {"object_id": self.receiver.uniqueID, "object_type": "ReceiptInstruction"}
                    ]
                )

                #logging
                #self.model.log_event(f"Error: Transaction {self.transactionID} failed due to no cash or securities available", self.transactionID, is_transaction = True)
            else:
                #handless partial settlement
                # Inside the partial settlement block in TransactionAgent.settle()
                if self.deliverer.get_institution().check_partial_allowed() and self.receiver.get_institution().check_partial_allowed():
                    delivery_children = self.deliverer.createDeliveryChildren()
                    receipt_children = self.receiver.createReceiptChildren()

                    # Check if partial settlement children were created.
                    if receipt_children == (None, None) or delivery_children == (None, None):
                        #self.model.log_event(
                        #    f"Transaction {self.transactionID} partial settlement aborted due to insufficient funds.",
                        #    self.transactionID,
                        #    is_transaction=True
                        #)
                        #new logging
                        self.model.log_event(
                            event_type="Partial_settlement_aborted",
                            object_ids=[self.transactionID,self.deliverer.uniqueID,self.receiver.uniqueID],
                            attributes={"status": self.status}
                        )

                        #old logging
                        self.model.log_ocel_event(
                            activity="Partial Settlement Aborted",
                            object_refs=[
                                {"object_id": self.transactionID, "object_type": "Transaction"},
                                {"object_id": self.deliverer.uniqueID, "object_type": "DeliveryInstruction"},
                                {"object_id": self.receiver.uniqueID, "object_type": "ReceiptInstruction"}
                            ]
                        )

                    else:
                        delivery_child_1, delivery_child_2 = delivery_children
                        receipt_child_1, receipt_child_2 = receipt_children

                        child_transaction_1 = delivery_child_1.match()
                        child_transaction_2 = delivery_child_2.match()
                        child_transaction_1.settle()
                        child_transaction_2.deliverer.get_securitiesAccount().set_newSecurities(False)
                        child_transaction_2.receiver.get_cashAccount().set_newSecurities(False)
                        #self.model.log_event(
                        #    f"Transaction {self.transactionID} partially settled. Children {receipt_child_1.get_uniqueID()}, "
                        #    f"{receipt_child_2.get_uniqueID()}, {delivery_child_1.get_uniqueID()} and {delivery_child_2.get_uniqueID()} created. "
                        #    f"Transactions {child_transaction_1.transactionID} and {child_transaction_2.transactionID} created.",
                        #    self.transactionID,
                        #    is_transaction=True
                        #)
                        #new logging
                        self.model.log_event(
                            event_type="transaction_partially_settled",
                            object_ids=[self.transactionID,self.deliverer.uniqueID,self.receiver.uniqueID],
                            attributes={"status": self.status}
                        )

                        #old logging
                        self.model.log_ocel_event(
                            activity="Partially Settled",
                            object_refs=[
                                {"object_id": self.transactionID, "object_type": "Transaction"},
                                {"object_id": self.deliverer.uniqueID, "object_type": "DeliveryInstruction"},
                                {"object_id": self.receiver.uniqueID, "object_type": "ReceiptInstruction"}
                            ]
                        )
                        self.cancel_partial()
        else:
        #    self.model.log_event(f"One of the instructions or transaction not in the correct state", self.transactionID, is_transaction = True)
            self.model.log_event(
                event_type="transaction_settlement_failed_incorrect_status",
                object_ids=[self.transactionID,self.deliverer.uniqueID,self.receiver.uniqueID],
                attributes={"status": self.status}
            )

            #old logging
            self.model.log_ocel_event(
                activity="Settlement Failed: Incorrect Status",
                object_refs=[
                    {"object_id": self.transactionID, "object_type": "Transaction"},
                    {"object_id": self.deliverer.uniqueID, "object_type": "DeliveryInstruction"},
                    {"object_id": self.receiver.uniqueID, "object_type": "ReceiptInstruction"}
                ]
            )

        self.deliverer.get_securitiesAccount().set_newSecurities(False)
        self.receiver.get_cashAccount().set_newSecurities(False)

    def step(self):
        time_of_day = self.model.simulated_time.time()
        if not (self.deliverer.get_securitiesAccount().get_newSecurities() == True or
            self.receiver.get_cashAccount().get_newSecurities() == True ):
            #if no new securities or cash where added to an account, no settlement gets tried
            return

        if self.deliverer.is_instruction_time_out():
            self.deliverer.cancel_timout()
        elif self.receiver.is_instruction_time_out():
            self.receiver.cancel_timout()
        elif self.status not in ["Cancelled due to timeout","Cancelled due to partial settlement", "Settled late", "Settled on time", "Cancelled due to error"]:
            if self.model.trading_start <= timedelta(hours=time_of_day.hour, minutes=time_of_day.minute) <= self.model.trading_end:
                self.settle()
        self.model.simulated_time = self.model.simulated_time + timedelta(seconds=1)

    def cancel_partial(self):
        self.status = "Cancelled due to partial settlement"
        self.deliverer.set_status("Cancelled due to partial settlement")
        self.receiver.set_status("Cancelled due to partial settlement")

        #new logging
        self.model.log_event(
            event_type="transaction_cancelled_partial_settlement",
            object_ids=[self.transactionID, self.deliverer.uniqueID, self.receiver.uniqueID],
            attributes={"status": self.status}
        )

        #old logging
        self.model.log_ocel_event(
            activity="Cancelled due to Partial Settlement",
            object_refs=[
                {"object_id": self.transactionID, "object_type": "Transaction"},
                {"object_id": self.deliverer.uniqueID, "object_type": "DeliveryInstruction"},
                {"object_id": self.receiver.uniqueID, "object_type": "ReceiptInstruction"}
            ]
        )
        #logging
        #self.model.log_event(f"Transaction {self.transactionID} cancelled due to partial settlement.", self.transactionID, is_transaction = True)
        #remove transition and instructions from the model when cancelled
        self.model.remove_transaction(self)
        self.model.agents.remove(self.deliverer)
        self.model.agents.remove(self.receiver)
        self.model.agents.remove(self)




