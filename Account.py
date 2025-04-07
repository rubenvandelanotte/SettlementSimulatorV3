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
            if total_available <= amount:
                deducted = total_available
                self.usedCredit = self.creditLimit
                self.balance = 0
                # logging
                #self.model.log_event(f"Account {self.accountID} deducted {amount} cash. Total cash amount: {self.balance}, total credit amount: {self.usedCredit}",self.accountID, is_transaction=False)
                return deducted

            elif total_available > amount and self.balance == 0:
                #adjust the creditLimit accordingly
                self.usedCredit = self.usedCredit + amount
                # logging
                #self.model.log_event(f"Account {self.accountID} deducted {amount} cash. Total cash amount: {self.balance}, total credit amount: {self.usedCredit}", self.accountID, is_transaction=False)
                return amount

            elif self.balance > 0:
                if self.balance >= amount:
                    self.balance = self.balance - amount
                    # logging
                    #self.model.log_event(f"Account {self.accountID} deducted {amount} cash. Total cash amount: {self.balance}, total credit amount: {self.usedCredit}", self.accountID, is_transaction=False)
                    return amount
                else:
                    deductedFromBalance = self.balance
                    self.balance = 0
                    self.usedCredit = self.usedCredit + deductedFromBalance
                    # logging
                    #self.model.log_event(f"Account {self.accountID} deducted {amount} cash. Total cash amount: {self.balance}, total credit amount: {self.usedCredit}",self.accountID, is_transaction=False)
                    return amount

        #deduct securities
        elif self.accountType == securityType:
            if self.balance >= amount:
                self.balance = self.balance -amount
                # logging
                #self.model.log_event(f"Account {self.accountID} deducted {amount} securities of type {self.accountType}. New amount: {self.balance}", self.accountID, is_transaction=False)
                return amount
            else:
                #should never happen
                return 0

        else:
            print("Error: account doesn't allow to deduct this type of assets")
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