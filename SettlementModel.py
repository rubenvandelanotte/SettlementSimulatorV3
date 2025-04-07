import pandas
from mesa import Model
from datetime import datetime, timedelta
import pandas as pd
import InstitutionAgent
import Account
import random
import os
from jsonocellogger import JSONOCELLogger

def generate_iban():
    """Generate a simple IBAN-like string.
    Example: 'DE45' + 16 digits.
    """
    country_code = random.choice(["DE", "FR", "NL", "GB"])
    check_digits = str(random.randint(10, 99))
    bban = ''.join(random.choices("0123456789", k=5))
    return f"{country_code}{check_digits}{bban}"


class SettlementModel(Model):
    def __init__(self, partialsallowed:tuple):
        super().__init__()
        self.partialsallowed= partialsallowed
        #parameters of the model
        self.num_institutions = 10
        self.min_total_accounts = 4
        self.max_total_accounts = 10
        self.simulation_duration_days = 15 #number of measured days (so simulation is longer)
        self.min_settlement_amount = 100
        self.bond_types = ["Bond-A", "Bond-B", "Bond-C", "Bond-D", "Bond-E", "Bond-F", "Bond-G", "Bond H", "Bond I"]
        self.logger = JSONOCELLogger()
        self.log_only_main_events= True



        self.simulation_start = datetime(2025, 4, 1, 1, 30)
        self.warm_up_period = timedelta(days=2)
        self.cool_down_period = timedelta(days=2)

        self.simulation_main_duration = timedelta(days=self.simulation_duration_days)
        self.simulation_total_duration = self.warm_up_period + self.simulation_main_duration + self.cool_down_period
        self.simulation_end = self.simulation_start + self.simulation_total_duration

        self.simulated_time = self.simulation_start

        self.trading_start = timedelta(hours=1, minutes=30)
        self.trading_end = timedelta(hours=19, minutes=30)
        self.batch_start = timedelta(hours=22, minutes=0)
        self.day_end = timedelta(hours=23, minutes=59, seconds=59)

        self.batch_processed = False
        self.institutions = []
        self.accounts = []
        self.instructions = []
        self.transactions = []

        self.activity_log = []

        #OCEL logging reqs:

        self.event_counter = 1
        self.event_log = []  # List to store OCEL events
        self.objects_catalog = {}  # Dictionary for static object info


        self.generate_data()

    def in_main_period(self):
        """Helper method to determine if the current simulated time is within the main period."""
        main_start = self.simulation_start + self.warm_up_period
        main_end = self.simulation_end - self.cool_down_period
        return main_start <= self.simulated_time <= main_end

    def next_event_id(self):
        event_id = f"e{self.event_counter}"
        self.event_counter += 1
        return event_id

    def log_object(self, object_id: str, object_type: str, attributes: dict = None):
        if attributes is None:
            attributes = {}
        attributes_list = [{"name": k, "value": v, "time": self.simulated_time.isoformat() + "Z"}
                        for k, v in attributes.items()]
        self.logger.log_object(oid=object_id, otype=object_type, attributes=attributes_list)

    def log_event(self, event_type: str, object_ids: list, attributes: dict = None):

        #allow for logging only in the main period (no warm-up / cooldown
        if self.log_only_main_events and not self.in_main_period():
            return


        if attributes is None:
            attributes = {}
        self.logger.log_event(
            event_type=event_type,
            object_ids=object_ids,
            event_attributes=attributes,
            timestamp=self.simulated_time.isoformat() + "Z"
        )

    def save_ocel_log(self, filename: str = "simulation_log.jsonocel"):
        self.logger.export_log(filename)

    #old logger
    def log_ocel_event(self, activity: str, object_refs: list):
        """
        Log an event in OCEL format.
        :param activity: The activity name (e.g., 'DeliveryInstruction Created').
        :param object_refs: List of dictionaries with keys 'object_id' and 'object_type'.
        """
        event_entry = {
            "event_id": self.next_event_id(),
            "timestamp": self.simulated_time.strftime('%Y-%m-%dT%H:%M:%S'),
            "activity": activity,
            "object_refs": object_refs
        }
        self.event_log.append(event_entry)
        print(f"Logged event: {event_entry}")

    #old logger
    def register_object(self, object_id: str, object_type: str, attributes: dict):
        self.objects_catalog[object_id] = {
            "object_id": object_id,
            "object_type": object_type,
            "attributes": attributes
        }
    #old logger
    def save_log(self, filename=None, activity_filename=None):
        if filename is None:
            filename = "ocel_event_log.csv"
        df = pd.DataFrame(self.event_log)
        df.to_csv(filename, index=False)
        if activity_filename is None:
            activity_filename = "objects_catalog.csv"
        df_activity = pd.DataFrame(list(self.objects_catalog.values()))
        df_activity.to_csv(activity_filename, index=False)
        print(f"Object catalog saved to {activity_filename}")
        print(f"Event Log saved to {filename}")

    def random_timestamp(self):
        delta = self.simulation_end - self.simulated_time
        random_seconds = random.uniform(0, delta.total_seconds())
        random_time = self.simulation_start + timedelta(seconds=random_seconds)
        return random_time  # Now returns a datetime object

   # def log_event(self, message, agent_id, is_transaction=True):
   #     timestamp = self.simulated_time.strftime('%Y-%m-%d %H:%M:%S')
   #     log_entry = {'Timestamp': timestamp, 'Agent ID': agent_id, 'Event': message}
    #
    #    if is_transaction:
    #        if log_entry not in self.event_log:
    #            print(f"{timestamp} | Agent ID: {log_entry['Agent ID']} | {message}")
    #            self.event_log.append(log_entry)  # Ensures no duplicates
     #   else:
     #       if log_entry not in self.activity_log:
     #           print(f"{timestamp} | Agent ID: {log_entry['Agent ID']} | {message}")
                  # Ensures no duplicates

      #  self.activity_log.append(log_entry)



    def sample_instruction_amount(self):
        """
        Samples an instruction amount (in EUR) based on a two-point mixture distribution.
        With probability ~88% return a 'low' amount (around €20 million, ±10% noise),
        and with probability ~12% return a 'high' amount (around €2.57 billion, ±5% noise)
        so that overall: mean ≈ €324M, std ≈ €829M, and median ≈ €20M.
        """
        if random.random() < 0.881:
            return random.uniform(18e6, 22e6)
        else:
            return random.uniform(2.45e9, 2.70e9)

    def generate_data(self):
        print("Generate Accounts & Institutions:")
        print("-----------------------------------------")
        for i in range(1, self.num_institutions+ 1):
            print("-------------------------------------")
            inst_bondtypes= []
            inst_id = f"INST-{i}"
            inst_accounts = []
            total_accounts = random.randint(self.min_total_accounts, self.max_total_accounts)
            #generate cash account => there has to be at least 1 cash account
            new_cash_accountID = generate_iban()
            new_cash_accountType = "Cash"
            new_cash_balance =  round(random.uniform(6e9, 9e9), 2)  # Increased balance range
            new_cash_creditLimit = round(random.uniform(0.25, 1), 2)*new_cash_balance
            new_cash_Account = Account.Account(accountID=new_cash_accountID, accountType= new_cash_accountType, balance= new_cash_balance, creditLimit=new_cash_creditLimit)
            inst_accounts.append(new_cash_Account)
            self.accounts.append(new_cash_Account)
            self.log_object(
                object_id=new_cash_accountID,
                object_type="Account",
                attributes={
                    "accountType": new_cash_accountType,
                    "balance": new_cash_balance,
                    "creditLimit": new_cash_creditLimit
                }
            )
            print(new_cash_Account.__repr__())
            for _ in range(total_accounts - 1):
                new_security_accountID = generate_iban()
                new_security_accountType = random.choicenew_security_accountType = random.choice([bt for bt in self.bond_types if bt not in inst_bondtypes])

                new_security_balance = round(random.uniform(600e7, 900e7), 2)
                new_security_creditLimit = 0
                new_security_Account = Account.Account(accountID=new_security_accountID, accountType= new_security_accountType, balance= new_security_balance, creditLimit= new_security_creditLimit)
                inst_accounts.append(new_security_Account)
                inst_bondtypes.append(new_security_accountType)
                self.accounts.append(new_security_Account)
                self.log_object(
                    object_id=new_security_accountID,
                    object_type="Account",
                    attributes={
                        "accountType": new_security_accountType,
                        "balance": new_security_balance,
                        "creditLimit": new_security_creditLimit
                    }
                )
                print(new_security_Account.__repr__())
            new_institution = InstitutionAgent.InstitutionAgent(institutionID= inst_id, accounts= inst_accounts, model=self, allowPartial=self.partialsallowed[i-1])
            self.institutions.append(new_institution)
            print(new_institution.__repr__())
        print("-------------------------------------------------------")
        print("Accounts & Institutions generated")
        #input("enter")




    def step(self):
            # Determine which period we are in:
            print(f"Running simulation step {self.steps}...")
            main_start = self.simulation_start + self.warm_up_period
            main_end = self.simulation_end - self.cool_down_period
            if self.simulated_time < main_start:
                current_period = "warm-up"
            elif self.simulated_time > main_end:
                current_period = "cool-down"
            else:
                current_period = "main"
            print("Current simulation period:", current_period)

            time_of_day = self.simulated_time.time()

            if self.trading_start <= timedelta(hours=time_of_day.hour, minutes=time_of_day.minute) <= self.trading_end:
                #real-time processing
                self.batch_processed = False

                #shuffles all agents and then executes their step module once for all of them
                self.agents.shuffle_do("step")
                print(f"{len(self.agents)} Agents executed their step module")

            elif timedelta(hours=time_of_day.hour, minutes=time_of_day.minute) >= self.batch_start:
                if not self.batch_processed: #batch processing at 22:00 only one loop of batch_processing
                    self.batch_processing()
                    self.batch_processed = True

            self.simulated_time += timedelta(seconds=1)

            if self.simulated_time >= datetime.combine(self.simulated_time.date(), datetime.min.time()) + self.day_end:
                self.simulated_time = datetime.combine(self.simulated_time.date() + timedelta(days=1), datetime.min.time()) + self.trading_start

    def register_transaction(self,t):
        self.transactions.append(t)

    def remove_transaction(self,t):
        self.transactions.remove(t)

    def batch_processing(self):
        for instruction in self.instructions:
            if instruction.get_status() == "Exists":
                instruction.insert()
            if instruction.get_status() == "Pending":
                instruction.validate()
            if instruction.get_status() == "Validated":
                instruction.match()
        for transaction in self.transactions:
            if transaction.get_status() == "Matched":
                transaction.settle()

    def get_recursive_settled_amount(self, parent_instruction):
        """
        Recursively sum the settled amounts of all descendant (child) instructions
        for a given parent instruction.

        Parameters:
            parent_instruction: The instruction whose children (and their descendants)
                                are to be summed.
        Returns:
            The total settled amount from all descendant instructions.
        """
        total = 0.0
        for inst in self.instructions:
            # Check if this instruction is a child of the parent_instruction.
            if inst.isChild and inst.motherID == parent_instruction.get_uniqueID():
                # If the child instruction is settled, add its amount.
                if inst.get_status() in ["Settled on time"]:
                    total += inst.get_amount()
                # Recursively include settled amounts from further descendant instructions.
                total += self.get_recursive_settled_amount(inst)
        return total

    def calculate_settlement_efficiency(self):
        """
        Calculates settlement efficiency based on instruction pairs, including recursive
        child instructions.

        Two metrics are computed:
         - Instruction Efficiency: the percentage of eligible original instruction pairs that
           reached settled status, either directly or through their child instructions.
         - Value Efficiency: the ratio (in percent) of the total settled (effective) value to the total
           intended settlement value.

        Original instructions are those with motherID == "mother", and they are grouped by their linkcode.
        For partial settlements, we check if the child instructions ultimately reached settlement.

        Returns:
            A tuple (instruction_efficiency_percentage, value_efficiency_percentage)
        """
        fully_settled_pairs = 0
        eligible_original_pairs = 0
        total_intended_value = 0.0
        total_settled_value = 0.0

        main_start = self.simulation_start + self.warm_up_period
        main_end = self.simulation_end - self.cool_down_period

        # Group original (mother) instructions by linkcode.
        original_pairs = {}
        for inst in self.instructions:
            if inst.motherID == "mother" and main_start <= inst.get_creation_time() <= main_end:
                original_pairs.setdefault(inst.linkcode, []).append(inst)

        for linkcode, pair in original_pairs.items():
            if len(pair) < 2:
                # We expect a pair (delivery and receipt); if not, skip this group.
                continue

            # We assume both instructions have the same intended settlement amount.
            intended_amount = pair[0].get_amount()
            total_intended_value += intended_amount

            # Case 1: Fully settled directly (both instructions settled on time or late).
            if (pair[0].get_status() in ["Settled on time", "Settled late"] and
                    pair[1].get_status() in ["Settled on time", "Settled late"]):
                eligible_original_pairs += 1
                fully_settled_pairs += 1
                total_settled_value += intended_amount

            # Case 2: Partial settlement – check if child instructions reached settlement
            elif (pair[0].get_status() == "Cancelled due to partial settlement" and
                  pair[1].get_status() == "Cancelled due to partial settlement"):
                # Calculate settled value for value efficiency
                settled_child_value = (self.get_recursive_settled_amount(pair[0]) +
                                       self.get_recursive_settled_amount(pair[1]))
                effective_settled = min(settled_child_value, intended_amount)
                total_settled_value += effective_settled

                # NEW: Check if all child instructions are settled
                # Get all child instructions for this pair
                child_instructions = self.get_all_child_instructions(pair[0].uniqueID, pair[1].uniqueID)

                # Check if all child instructions are either settled or cancelled due to further partial settlement
                all_children_settled = True
                for child in child_instructions:
                    if child.get_status() not in ["Settled on time", "Settled late",
                                                  "Cancelled due to partial settlement"]:
                        all_children_settled = False
                        break

                # If all children are settled or further partially settled, count this pair as successful
                if all_children_settled and effective_settled >= intended_amount * 0.99:  # Allow for small rounding differences
                    eligible_original_pairs += 1
                    fully_settled_pairs += 1

            # Case 3: Other statuses (failed, rejected, etc.)
            else:
                eligible_original_pairs += 1  # Count in denominator but not in numerator

        instruction_efficiency = (
                    fully_settled_pairs / eligible_original_pairs * 100) if eligible_original_pairs > 0 else 0
        value_efficiency = (total_settled_value / total_intended_value * 100) if total_intended_value > 0 else 0

        return instruction_efficiency, value_efficiency

    def get_all_child_instructions(self, instruction_id1, instruction_id2):
        """
        Recursively get all child instructions for a pair of mother instructions.

        Args:
            instruction_id1: ID of the first mother instruction
            instruction_id2: ID of the second mother instruction

        Returns:
            List of all child instructions
        """
        children = []

        # First level children
        for inst in self.instructions:
            if inst.motherID in [instruction_id1, instruction_id2]:
                children.append(inst)
                # Recursively add all descendants
                further_children = self.get_all_descendants(inst.uniqueID)
                children.extend(further_children)

        return children

    def get_all_descendants(self, instruction_id):
        """
        Recursively get all descendants of an instruction.

        Args:
            instruction_id: ID of the parent instruction

        Returns:
            List of all descendant instructions
        """
        descendants = []

        for inst in self.instructions:
            if inst.motherID == instruction_id:
                descendants.append(inst)
                # Recursively add further descendants
                further_descendants = self.get_all_descendants(inst.uniqueID)
                descendants.extend(further_descendants)

        return descendants



    def print_settlement_efficiency(self):
        """
        Quickly prints the settlement efficiency metrics.
        """
        instruction_eff, value_eff = self.calculate_settlement_efficiency()
        print("Settlement Efficiency:")
        print("  Instruction Efficiency: {:.2f}%".format(instruction_eff))
        print("  Value Efficiency: {:.2f}%".format(value_eff))

    def save_settlement_efficiency_to_csv(self, filename="settlement_efficiency.csv"):
        """
        Saves the settlement efficiency metrics to a CSV file.
        """
        instruction_eff, value_eff = self.calculate_settlement_efficiency()
        data = [
            {"Metric": "Instruction Efficiency (%)", "Value": round(instruction_eff, 2)},
            {"Metric": "Value Efficiency (%)", "Value": round(value_eff, 2)}
        ]
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Settlement efficiency metrics saved to {filename}")


#if __name__ == "__main__":
#    print("Starting simulation...")
#    log_path = input("Enter the path to save the log (press Enter for default): ")
#    if not log_path.strip():
#        log_path = "event_log.csv"
#    partial1 = (False,False,False,False,False, False, False, False, False, False)
#    partial2 =(True,False,False,False,False)
#    partial3= (True, True, False, False, False)
#    partial4= (True,True,True,False,False)
#    partial5=(True,True,True,True,False)
#    partial6=(True, True, True, True, True, True, True, True, True, True)
#    partials = list()
#    partials.append(partial1)
    #partials.append(partial2)
    #partials.append(partial3)
    #partials.append(partial4)
    #partials.append(partial5)
  #  partials.append(partial6)
  #  efficiencies = []
  #  for p in partials:

     #   for i in range(5):
     #       model = SettlementModel(partialsallowed=p)
     #       try:
      #          while model.simulated_time < model.simulation_end:
       #             model.step()
       #     except RecursionError:
       #         print("RecursionError encountered: maximum recursion depth exceeded. Terminating simulation gracefully.")

        #    print("Final Event Log:")
        #    for event in model.event_log:
        #        print(event)
        #    print("Saving final event log...")
         #   model.save_log(log_path)
         #   model.save_ocel_log(filename="simulation_more_securities.jsonocel")
         #   print("---------------------------------------------------------------")
         #   model.print_settlement_efficiency()
         #   model.save_settlement_efficiency_to_csv()
         #   new_ins_eff, new_val_eff = model.calculate_settlement_efficiency()
         #   new_eff = {'Partial': str(p), 'instruction efficiency': new_ins_eff, 'value efficiency': new_val_eff}
         #   efficiencies.append(new_eff)

        #print(efficiencies)

    #df = pd.DataFrame(efficiencies)
    #df.to_csv("15 days all partials, 10 runs, new params")


#dd