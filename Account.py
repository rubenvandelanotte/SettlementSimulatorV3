class Account:

    #account object class
    def __init__(self, accountID: str, accountType:str, balance:int, newSecurities: bool = True, creditLimit: int = 0):
        self.accountID = accountID
        self.accountType = accountType
        self.balance = balance
        self.creditLimit = creditLimit
        self.newSecurities = newSecurities
        self.usedCredit = 0  #why not int declaration?

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

    def checkBalance(self, amount : int, securityType: str):
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

    def addBalance(self, amount:int, securityType:str):
        #if cash account
        if self.accountType == securityType and self.accountType == "Cash":
            #check whether this account used credit already: condition only satisfies if usedCredit ==0
            if self.creditLimit == (self.creditLimit - self.usedCredit):
                self.balance = self.balance + amount
                self.set_newSecurities(True)
                return amount

            elif self.creditLimit != self.creditLimit - self.usedCredit:
                #credit used so far, meaning balance should be 0 and credit used for self.usedCredit
                if self.usedCredit >= amount:
                    #reset the used credit with the amount
                    self.usedCredit = self.usedCredit - amount
                    self.set_newSecurities(True)
                    return amount
                else:
                    #set used credit to 0 and add remaining to balance
                    remaining = amount - self.usedCredit
                    self.usedCredit = 0
                    self.balance = self.balance + remaining
                    self.set_newSecurities(True)
                    return amount

        #if security account:
        elif self.accountType == securityType:
            self.balance = self.balance + amount
            self.set_newSecurities(True)
            return amount

        else:
            print("Error: account doesn't allow to add this type of assets")
            return 0


    def deductBalance(self, amount: int, securityType:str):

        #deduct cash
        if self.accountType == securityType and self.accountType == "Cash":
            total_available = self.balance + (self.creditLimit - self.usedCredit)
            if total_available < amount:
                return 0


            if self.balance >= amount:
                self.balance= self.balance - amount
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
                print(f"[DEDUCT SECURITIES] Account {self.accountID} | Deducted {amount} of {securityType}. New Balance: {self.balance}")

                return amount
            else:
                #should never happen
                print(f"[DEDUCT FAILED] Account {self.accountID} | Insufficient {securityType}. Balance: {self.balance}, Requested: {amount}")
                return 0

        else:
            print(f"[ERROR] Account {self.accountID} | Cannot deduct {securityType} from account type {self.accountType}")
            return 0



