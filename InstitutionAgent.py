#InstitutionAgent
import time
import random
from datetime import datetime, timedelta
import SettlementModel
from mesa import Agent, Model
import ReceiptInstructionAgent
import InstructionAgent
import DeliveryInstructionAgent
import Account


class InstitutionAgent(Agent):

    def __init__(self, model:SettlementModel, institutionID:str, accounts:list[Account] = [],allowPartial:bool = True):
        super().__init__(model)

        self.institutionID = institutionID
        self.accounts = accounts
        self.allowPartial = allowPartial

        self.model.log_object(
            object_id=self.institutionID,
            object_type="Institution",
            attributes={
                "allowPartial": self.allowPartial
            }
        )

    def opt_out_partial(self):
        if not self.allowPartial:
            print("Institution Already opted out, cannot opt out again")
        else:
            self.allowPartial = False
            print("Institution opted out of partial settlement")
            self.model.log_event(
                event_type="institution_opt_out_partial",
                object_ids=[self.institutionID],
                attributes={"allowPartial": self.allowPartial}
            )

    def opt_in_partial(self):
        if self.allowPartial:
            print("Institution Already opted in, cannot opt in again")
        else:
            self.allowPartial = True
            print("Institution opted in of partial settlements")
            self.model.log_event(
                event_type="institution_opt_in_partial",
                object_ids=[self.institutionID],
                attributes={"allowPartial": self.allowPartial}
            )

    def check_partial_allowed(self):
        if self.allowPartial == True:
            return True
        else:
            return False

    def getSecurityAccounts(self, securityType:str):

        for account in self.accounts:
            if account.accountType == securityType:
                return account



    def create_instruction(self):
        instruction_type = random.choice(['delivery', 'receipt'])

        cash_account = self.getSecurityAccounts(securityType= "Cash")
        # Filter available security accounts from the institution's accounts
        security_accounts = [acc for acc in self.accounts if acc.accountType in self.model.bond_types]
        if not security_accounts:
            raise ValueError("No security accounts available for institution " + self.institutionID)
        security_account = random.choice(security_accounts)
        random_security = security_account.accountType  # Use the type from the chosen account
        amount = self.model.sample_instruction_amount()
        model = self.model
        linkedTransaction = None
        uniqueID = len(self.model.instructions) + 1
        otherID = len(self.model.instructions) + 2
        motherID = "mother"
        institution = self
        other_institution = random.choice([inst for inst in self.model.institutions if inst != self])
        securityType = random_security
        other_institution_cash_account= other_institution.getSecurityAccounts(securityType= "Cash")
        other_institution_security_account = other_institution.getSecurityAccounts(securityType=securityType)
        if other_institution_security_account is None:
            # Create a new security account for the counterparty institution
            new_security_account_id = SettlementModel.generate_iban()  # Generates an IBAN-like string
            new_security_balance = int(random.uniform(600e7, 900e7))  # Mimic the balance generation logic
            new_security_account = Account.Account(
                accountID=new_security_account_id,
                accountType=securityType,
                balance=new_security_balance,
                creditLimit=0
            )
            # Add the new account to the institution's list of accounts
            other_institution.accounts.append(new_security_account)
            other_institution_security_account = new_security_account
            self.model.accounts.append(new_security_account)
        isChild = False
        status = "Exists"
        linkcode = f"LINK-{uniqueID}L{otherID}"
        instruction_creation_time = self.model.simulated_time
        delay_seconds = random.uniform(0.5, 5)  # random delay between 0.5 and 5 seconds
        counter_instruction_creation_time = instruction_creation_time + timedelta(seconds=delay_seconds)

        if instruction_type == 'delivery':
            new_instructionAgent = DeliveryInstructionAgent.DeliveryInstructionAgent(uniqueID=uniqueID, model = model, linkedTransaction = linkedTransaction, motherID=motherID, institution= institution, securitiesAccount = security_account, cashAccount = cash_account, securityType=securityType, amount= amount, isChild=isChild, status=status, linkcode=linkcode, creation_time = instruction_creation_time)
            counter_instructionAgent = ReceiptInstructionAgent.ReceiptInstructionAgent(uniqueID=otherID, model = model, linkedTransaction = linkedTransaction, motherID=motherID, institution= other_institution, securitiesAccount = other_institution_security_account, cashAccount = other_institution_cash_account, securityType=securityType, amount= amount, isChild=isChild, status=status, linkcode=linkcode, creation_time = counter_instruction_creation_time)
            self.model.instructions.append(new_instructionAgent)
            self.model.instructions.append(counter_instructionAgent)
        else:
            new_instructionAgent = ReceiptInstructionAgent.ReceiptInstructionAgent(uniqueID=uniqueID, model = model, linkedTransaction = linkedTransaction, motherID=motherID, institution= institution, securitiesAccount = security_account, cashAccount = cash_account, securityType=securityType, amount= amount, isChild=isChild, status=status, linkcode=linkcode, creation_time = instruction_creation_time)
            counter_instructionAgent = DeliveryInstructionAgent.DeliveryInstructionAgent(uniqueID=otherID, model = model, linkedTransaction = linkedTransaction, motherID=motherID, institution= other_institution, securitiesAccount = other_institution_security_account, cashAccount = other_institution_cash_account, securityType=securityType, amount= amount, isChild=isChild, status=status, linkcode=linkcode, creation_time = counter_instruction_creation_time)
            self.model.instructions.append(new_instructionAgent)
            self.model.instructions.append(counter_instructionAgent)

        #new logging
        self.model.log_event(
            event_type="instruction_pair_created",
            object_ids=[
                new_instructionAgent.uniqueID,
                counter_instructionAgent.uniqueID,
                self.institutionID,
                other_institution.institutionID
            ],
            attributes={
                "securityType": securityType,
                "amount": amount,
                "linkcode": linkcode
            }
        )
        #old logging
        self.model.log_ocel_event(
            activity="Instruction Pair Created",
            object_refs=[
                {"object_id": new_instructionAgent.uniqueID, "object_type": "Instruction"},
                {"object_id": counter_instructionAgent.uniqueID, "object_type": "Instruction"},
                {"object_id": self.institutionID, "object_type": "Institution"}
            ]
        )



    #
    # def create_cancelation_instruction(self):
    #
    #     #to implement later on
    #     return


    def create_account(self):
        #not really relevant so far
        return

    def step(self):

        #if selected create an instruction and with low probability allow/ disallow partial settlements

        if random.random() <0.1:
            self.create_instruction()
        # if random.random() <0.05:
        #     self.create_cancelation_instruction()

        # if random.random() < 0.01:
        #     if self.allowPartial:
        #         self.opt_out_partial()
        #     else:
        #         self.opt_in_partial()
        #self.model.simulated_time = self.model.simulated_time + timedelta(seconds=1)

    def get_full_institution_info(self):
        """
        Return a dictionary with all institution attributes,
        including a detailed list of its accounts.
        """
        return {
            "institutionID": self.institutionID,
            "allowPartial": self.allowPartial,
            "accounts": [account.get_full_account_info() for account in self.accounts]
        }

    def __repr__(self):
        return (
            f"InstitutionAgent(institutionID={self.institutionID}, "
            f"allowPartial={self.allowPartial}, "
            f"accounts={self.accounts})"
        )
#