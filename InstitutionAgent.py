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


def get_creation_time(isd: datetime, delay: int, simulated_time:datetime) -> datetime:
    creation_date = isd - timedelta(days=delay)

    #pick random time on that day
    start = timedelta(hours=1, minutes=30)
    end = timedelta(hours=19, minutes=30)
    delta_seconds = int((end - start).total_seconds())
    random_offset = timedelta(seconds=random.randint(0, delta_seconds))

    #merge with the day chosen
    creation_time = creation_date.replace(hour=0, minute=0, second=0, microsecond=0) + start + random_offset
    #ensures no creations in the past
    return max(creation_time, simulated_time)

def sample_instruction_creation_times_and_isd(simulated_time: datetime):
    """
    Samples two distinct creation times and a shared ISD (intended settlement date),
    ensuring:
    - ISD is 2–5 days in the future
    - ISD always falls at 22:30
    - Creation times are generated using empirical delay distribution
    - No creation occurs before simulated_time
    """

    # Step 1: Choose ISD 2–5 days from now and set it on 22:30
    raw_isd_day = simulated_time + timedelta(days=random.randint(2, 5))
    isd = raw_isd_day.replace(hour=22, minute=30, second=0, microsecond=0)

    # Step 2: Define delay distribution
    delays = [-1, 0, 1, 2, 3, 4, 5]  # Delay in days from ISD
    weights = [0.01, 0.29, 0.22, 0.20, 0.07, 0.16, 0.05]  # Empirical weights

    # Step 3: Sample delays for both instruction legs
    delay_a = random.choices(delays, weights=weights, k=1)[0]
    delay_b = random.choices(delays, weights=weights, k=1)[0]


    # Step 4: Generate creation times
    creation_time_a = get_creation_time(isd, delay_a, simulated_time)
    creation_time_b = get_creation_time(isd, delay_b, simulated_time)

    return creation_time_a, creation_time_b, isd

class InstitutionAgent(Agent):

    def __init__(self, model:SettlementModel, institutionID:str, accounts:list[Account] = [], allowPartial:bool = True):
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
        instruction_creation_time, counter_instruction_creation_time, isd = sample_instruction_creation_times_and_isd(self.model.simulated_time)

        if instruction_type == 'delivery':
            new_instructionAgent = DeliveryInstructionAgent.DeliveryInstructionAgent(uniqueID=uniqueID, model = model, linkedTransaction = linkedTransaction, motherID=motherID, institution= institution, securitiesAccount = security_account, cashAccount = cash_account, securityType=securityType, amount= amount, isChild=isChild, status=status, linkcode=linkcode, creation_time = instruction_creation_time)
            counter_instructionAgent = ReceiptInstructionAgent.ReceiptInstructionAgent(uniqueID=otherID, model = model, linkedTransaction = linkedTransaction, motherID=motherID, institution= other_institution, securitiesAccount = other_institution_security_account, cashAccount = other_institution_cash_account, securityType=securityType, amount= amount, isChild=isChild, status=status, linkcode=linkcode, creation_time = counter_instruction_creation_time)

            #set isds
            new_instructionAgent.set_intended_settlement_date(isd)
            counter_instructionAgent.set_intended_settlement_date(isd)

            self.model.instructions.append(new_instructionAgent)
            self.model.instructions.append(counter_instructionAgent)
        else:
            new_instructionAgent = ReceiptInstructionAgent.ReceiptInstructionAgent(uniqueID=uniqueID, model = model, linkedTransaction = linkedTransaction, motherID=motherID, institution= institution, securitiesAccount = security_account, cashAccount = cash_account, securityType=securityType, amount= amount, isChild=isChild, status=status, linkcode=linkcode, creation_time = instruction_creation_time)
            counter_instructionAgent = DeliveryInstructionAgent.DeliveryInstructionAgent(uniqueID=otherID, model = model, linkedTransaction = linkedTransaction, motherID=motherID, institution= other_institution, securitiesAccount = other_institution_security_account, cashAccount = other_institution_cash_account, securityType=securityType, amount= amount, isChild=isChild, status=status, linkcode=linkcode, creation_time = counter_instruction_creation_time)

            #set isds
            new_instructionAgent.set_intended_settlement_date(isd)
            counter_instructionAgent.set_intended_settlement_date(isd)

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

    def create_account(self):
        #not really relevant so far
        return

    def step(self):
        if random.random() <0.1:
            self.create_instruction()



