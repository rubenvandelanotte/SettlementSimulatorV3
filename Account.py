class Account:

    #account object class
    def __init__(self, accountID: str, accountType:str,balance:float, newSecurities: bool = True, creditLimit:float = 0):
        self.accountID = accountID
        self.accountType = accountType
        self.balance = balance
        self.creditLimit = creditLimit
        self.newSecurities = newSecurities
        self.usedCredit = 0


        #logging
        #self.model.log_event(f"Account {accountID} of type {accountType} created with balance {balance} and credit limit {creditLimit}", accountID, is_transaction = False)

    def getAccountID(self):
        return self.accountID

    def getCreditLimit(self):
        return self.creditLimit

    def getBalance(self):
        return self.balance

    def getAccountType(self):
        return self.accountType

    def getUsedCredit(self):
        return self.usedCredit

    def get_newSecurities(self):
        return self.newSecurities

    def set_newSecurities(self, new:bool):
        self.newSecurities = new

    def checkBalance(self, amount : float, securityType: str):
        if self.accountType == "Cash" and securityType == "Cash":
           return  self.balance + self.creditLimit >= amount
        elif self.accountType == securityType:
            return self.balance >= amount
        else:
            return False

    def getEffectiveAvailableCash(self):
        if self.accountType == "Cash":
            return self.balance + self.creditLimit - self.usedCredit
        return 0



    def addBalance(self, amount:float, securityType:str):

        #if cash account
        if self.accountType == securityType and self.accountType == "Cash":

            #check whether this account used credit already: condition only satisfies if usedCredit ==0
            if self.creditLimit == (self.creditLimit - self.usedCredit):

                self.balance = self.balance + amount
                self.set_newSecurities(True)
                #logging
                #self.model.log_event(f"Account {self.accountID} added {amount} cash. Total cash amount: {self.balance}, total credit amount: {self.usedCredit}", self.accountID, is_transaction = False)
                return amount

            elif self.creditLimit != self.creditLimit - self.usedCredit:
                #credit used so far, meaning balance should be 0 and credit used for self.usedCredit
                if self.usedCredit >= amount:
                    #reset the used credit with the amount
                    self.usedCredit = self.usedCredit - amount
                    self.set_newSecurities(True)
                    # logging
                    #self.model.log_event(f"Account {self.accountID} added {amount} cash. Total cash amount: {self.balance}, total credit amount: {self.usedCredit}", self.accountID, is_transaction=False)
                    return amount
                else:
                    #set used credit to 0 and add remaining to balance
                    remaining = amount - self.usedCredit
                    self.usedCredit = 0
                    self.balance = self.balance + remaining
                    self.set_newSecurities(True)
                    # logging
                    #self.model.log_event(f"Account {self.accountID} added {amount} cash. Total cash amount: {self.balance}, total credit amount: {self.usedCredit}",self.accountID, is_transaction=False)
                    return amount

        #if security account:
        elif self.accountType == securityType:
            self.balance = self.balance + amount
            self.set_newSecurities(True)
            #logging
            #self.model.log_event(f"Account {self.accountID} added {amount} securities of type {self.accountType}. New amount: {self.balance}", self.accountID, is_transaction = False)
            return amount

        else:
            print("Error: account doesn't allow to add this type of assets")
            #logging
            #self.model.log_event(f"ERROR: account {self.accountID} doesn't allow to add this type of assets", self.accountID, is_transaction = False)
            return 0


    def deductBalance(self, amount: float, securityType:str):

        #deduct cash
        if self.accountType == securityType and self.accountType == "Cash":
            total_available = self.balance + (self.creditLimit - self.usedCredit)
            if total_available < amount:
                #print(f"[DEDUCT FAILED] Account {self.accountID} | Requested: {amount}, Available: {total_available} (Balance: {self.balance}, UsedCredit: {self.usedCredit}, CreditLimit: {self.creditLimit})")
                return 0
                # logging
                #self.model.log_event(f"Account {self.accountID} deducted {amount} cash. Total cash amount: {self.balance}, total credit amount: {self.usedCredit}",self.accountID, is_transaction=False)

            if self.balance >= amount:
                self.balance= self.balance - amount
                #print(f"[DEDUCT CASH] Account {self.accountID} | Deducted {amount} from balance. New Balance: {self.balance}, UsedCredit: {self.usedCredit}")
                return amount

            else:
                needed_from_credit = amount - self.balance
                print(f"[PARTIAL CASH] Account {self.accountID} | Deducted {self.balance} from balance and {needed_from_credit} from credit.")
                self.balance = 0
                self.usedCredit += needed_from_credit
                print(f"[UPDATED] Account {self.accountID} | New Balance: {self.balance}, UsedCredit: {self.usedCredit}")
                return amount


        #deduct securities
        elif self.accountType == securityType:
            if self.balance >= amount:
                self.balance = self.balance -amount
                # logging
                #self.model.log_event(f"Account {self.accountID} deducted {amount} securities of type {self.accountType}. New amount: {self.balance}", self.accountID, is_transaction=False)
                print(f"[DEDUCT SECURITIES] Account {self.accountID} | Deducted {amount} of {securityType}. New Balance: {self.balance}")

                return amount
            else:
                #should never happen
                print(f"[DEDUCT FAILED] Account {self.accountID} | Insufficient {securityType}. Balance: {self.balance}, Requested: {amount}")
                return 0

        else:
            print(f"[ERROR] Account {self.accountID} | Cannot deduct {securityType} from account type {self.accountType}")
            # logging
            #self.model.log_event(f"ERROR: account {self.accountID} doesn't allow to deduct this type of assets", self.accountID, is_transaction=False)
            return 0



    def get_full_account_info(self):
            """Return a dictionary of all account attributes."""
            return {
                "accountID": self.accountID,
                "accountType": self.accountType,
                "balance": self.balance,
                "creditLimit": self.creditLimit,
                "usedCredit": self.usedCredit}

    def __repr__(self):
            """String representation useful for debugging."""
            return (
                f"Account(accountID={self.accountID}, "
                f"accountType={self.accountType}, "
                f"balance={self.balance}, "
                f"creditLimit={self.creditLimit}, "
                f"usedCredit={self.usedCredit})"
            )
#