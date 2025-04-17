import pandas
from mesa import Model
from datetime import datetime, timedelta
import pandas as pd
from sympy.physics.units import seconds

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
    def __init__(self, partialsallowed:tuple, seed: int = None):
        super().__init__()
        self.partialsallowed= partialsallowed
        #parameters of the model
        self.num_institutions = 10
        self.min_total_accounts = 4
        self.max_total_accounts = 10
        self.simulation_duration_days = 15 #number of measured days (so simulation is longer)
        self.min_settlement_percentage = 0.05
        self.max_child_depth = 10
        self.bond_types = ["Bond-A", "Bond-B", "Bond-C", "Bond-D", "Bond-E", "Bond-F", "Bond-G", "Bond H", "Bond I"]
        self.logger = JSONOCELLogger()
        self.log_only_main_events= True
        self.seed = seed
        if seed is not None:
            random.seed(seed)  # Set the random seed
            print(f"[INFO] Random seed set to: {seed}")



        self.simulation_start = datetime(2025, 4, 1, 1, 30)
        self.warm_up_period = timedelta(days=5)
        self.cool_down_period = timedelta(days=5)

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

        #mini batching
        self.mini_batch_times = [timedelta(hours=h, minutes=m) for h in range(2, 19) for m in [0, 30]] + [
            timedelta(hours=19, minutes=0)]
        self.mini_batches_processed = set()

        # Instruction indices for fast lookup
        self.validated_delivery_instructions = {}  # linkcode -> list of delivery instructions
        self.validated_receipt_instructions = {}  # linkcode -> list of receipt instructions

        #OCEL logging reqs:

        self.event_counter = 1
        self.event_log = []  # List to store OCEL events => probably redudant


        self.partial_cancelled_count = 0

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


    def random_timestamp(self):
        delta = self.simulation_end - self.simulated_time
        random_seconds = random.uniform(0, delta.total_seconds())
        random_time = self.simulation_start + timedelta(seconds=random_seconds)
        return random_time  # Now returns a datetime object



    def sample_instruction_amount(self):
        """
        Samples an instruction amount (in EUR) based on a two-point mixture distribution.
        With probability ~88% return a 'low' amount (around €20 million, ±10% noise),
        and with probability ~12% return a 'high' amount (around €2.57 billion, ±5% noise)
        so that overall: mean ≈ €324M, std ≈ €829M, and median ≈ €20M.
        """
        if random.random() < 0.881:
            return int(random.uniform(18e6, 22e6))
        else:
            return int(random.uniform(2.45e9, 2.70e9))

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
            new_cash_balance =  int(random.uniform(6e9, 9e9))  # Increased balance range
            new_cash_creditLimit = int(random.uniform(0.25, 1))*new_cash_balance
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
                new_security_accountType = random.choice([bt for bt in self.bond_types if bt not in inst_bondtypes])

                new_security_balance = int(random.uniform(600e7, 900e7))
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

    def step(self):
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

            # Reset mini-batch tracker
            if self.simulated_time.time() == self.trading_start:
                self.mini_batches_processed = set()

            time_of_day = self.simulated_time.time()

            if self.trading_start <= timedelta(hours=time_of_day.hour, minutes=time_of_day.minute) <= self.trading_end:
                #real-time processing
                self.batch_processed = False

                #shuffles all agents and then executes their step module once for all of them
                self.agents.shuffle_do("step")
                print(f"{len(self.agents)} Agents executed their step module")

                for batch_time in self.mini_batch_times:
                    key = (self.simulated_time.date(), batch_time)
                    if batch_time <= timedelta(hours=time_of_day.hour, minutes = time_of_day.minute, seconds= time_of_day.second) and key not in self.mini_batches_processed:
                        self.mini_batch_settlement()
                        self.mini_batches_processed.add(key)


            elif timedelta(hours=self.simulated_time.hour, minutes= self.simulated_time.minute, seconds=self.simulated_time.second) >= self.batch_start and not self.batch_processed:
                #batch processing at 22:00 only one loop of batch_processing
                    self.batch_processing()
                    self.batch_processed = True

            self.simulated_time += timedelta(seconds=600)

            if self.simulated_time >= datetime.combine(self.simulated_time.date(), datetime.min.time()) + self.day_end:
                self.simulated_time = datetime.combine(self.simulated_time.date() + timedelta(days=1), datetime.min.time()) + self.trading_start

    def register_transaction(self,t):
        self.transactions.append(t)

    def remove_transaction(self,t):
        self.transactions.remove(t)

    def batch_processing(self):
        # --- Phase 1: Insert all ---
        for instruction in self.instructions:
            if instruction.get_status() == "Exists":
                instruction.insert()

        # --- Phase 2: Validate all ---
        for instruction in self.instructions:
            if instruction.get_status() == "Pending":
                instruction.validate()

        # --- Phase 3: Match with retries ---
        matching_changed = True
        attempts = 0
        while matching_changed and attempts <3 :
            matching_changed = False
            attempts = attempts + 1

            # Process one linkcode at a time where both delivery and receipt instructions exist
            common_linkcodes = set(self.validated_delivery_instructions.keys()) & set(
                self.validated_receipt_instructions.keys())

            for linkcode in common_linkcodes:
                delivery_list = self.validated_delivery_instructions.get(linkcode, [])
                if delivery_list:  # If there are delivery instructions with this linkcode
                    delivery = delivery_list[0]  # Take the first one
                    if delivery.get_status() == "Validated":
                        result = delivery.match()
                        if result is not None:
                            matching_changed = True

        # Continue with settlement phase
        for transaction in self.transactions:
            if transaction.get_status() == "Matched":
                transaction.settle()

    def mini_batch_settlement(self):
        print(f"[INFO] Running mini-batch settlement at {self.simulated_time}")
        for transaction in self.transactions:
            if transaction.get_status() == "Matched":
                transaction.settle()

    def get_main_period_mothers_and_descendants(self):
        main_start = self.simulation_start + self.warm_up_period
        main_end = self.simulation_end - self.cool_down_period

        mother_instructions = [
            inst for inst in self.instructions
            if inst.get_motherID() == "mother" and main_start <= inst.get_creation_time() <= main_end
        ]

        descendants = set(mother_instructions)
        queue = list(mother_instructions)

        while queue:
            current = queue.pop()
            children = [
                inst for inst in self.instructions
                if inst.isChild and inst.get_motherID() == current.get_uniqueID()
            ]
            descendants.update(children)
            queue.extend(children)

        return list(descendants)

    def get_main_period_mothers_and_descendants_optimized(self):
        """
        Optimized version that uses a more efficient breadth-first traversal
        to collect all relevant instructions.
        """
        main_start = self.simulation_start + self.warm_up_period
        main_end = self.simulation_end - self.cool_down_period

        # Get mother instructions from main period
        mother_instructions = [
            inst for inst in self.instructions
            if inst.get_motherID() == "mother" and main_start <= inst.get_creation_time() <= main_end
        ]

        # Use a set for faster membership testing
        descendants = set(mother_instructions)

        # Create a mapping from parent ID to children for faster lookup
        parent_to_children = {}
        for inst in self.instructions:
            if inst.isChild:
                parent_id = inst.get_motherID()
                if parent_id not in parent_to_children:
                    parent_to_children[parent_id] = []
                parent_to_children[parent_id].append(inst)

        # Use queue for breadth-first traversal
        queue = [inst for inst in mother_instructions]
        while queue:
            current = queue.pop(0)
            current_id = current.get_uniqueID()

            # Get children from pre-computed mapping
            children = parent_to_children.get(current_id, [])

            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)

        return list(descendants)

    def get_recursive_settled_amount(self, parent_instruction, instruction_pool=None):
        instruction_pool = instruction_pool if instruction_pool is not None else self.instructions
        total = 0.0
        processed = set()

        # Pre-compute child relationships for faster lookup
        child_map = {}
        for inst in instruction_pool:
            if inst.isChild:
                if inst.motherID not in child_map:
                    child_map[inst.motherID] = []
                child_map[inst.motherID].append(inst)

        # Start with direct children of parent
        queue = child_map.get(parent_instruction.get_uniqueID(), [])

        # Process all descendants
        while queue:
            current = queue.pop(0)
            current_id = current.get_uniqueID()

            if current_id in processed:
                continue
            processed.add(current_id)

            if current.get_status() in ["Settled on time"]:
                total += current.get_amount()

            # Add all children to the queue
            queue.extend(child_map.get(current_id, []))

        return total

    def get_settled_amount_iterative(self, parent_id, child_map, status_cache, amount_cache):
        """
        Iterative calculation of total settled amount for all descendants of an instruction.

        Args:
            parent_id (str): ID of the parent instruction
            child_map (dict): Dictionary mapping parent IDs to lists of child IDs
            status_cache (dict): Dictionary mapping instruction IDs to their status
            amount_cache (dict): Dictionary mapping instruction IDs to their amounts

        Returns:
            float: Total settled amount from all descendants
        """
        total = 0.0
        processed = set()

        # Start with direct children of parent
        queue = child_map.get(parent_id, [])

        # Process all descendants
        while queue:
            current_id = queue.pop(0)

            if current_id in processed:
                continue
            processed.add(current_id)

            # Add settled amount if this instruction is settled on time
            if status_cache.get(current_id) == "Settled on time":
                total += amount_cache.get(current_id, 0)

            # Add all children to the queue
            queue.extend(child_map.get(current_id, []))

        return total

    def calculate_settlement_efficiency(self):
        """
        Calculates settlement efficiency based on instruction pairs, including recursive
        child instructions.

        Two metrics are computed:
         - Instruction Efficiency: the percentage of original instruction pairs (mother instructions)
           that ended up fully settled (either directly or via child instructions covering the full amount).
         - Value Efficiency: the ratio (in percent) of the total settled (effective) value to the total
           intended settlement value.

        Original instructions are those with motherID == "mother", and they are grouped by their linkcode.
        In case of partial settlement (both instructions cancelled due to partial settlement), this method
        recursively aggregates settled amounts from child instructions.

        Returns:
            A tuple (instruction_efficiency_percentage, value_efficiency_percentage)
        """

        relevant_instructions = self.get_main_period_mothers_and_descendants()
        original_pairs = {}

        for inst in relevant_instructions:
            if inst.get_motherID() == "mother":
                original_pairs.setdefault(inst.get_linkcode(), []).append(inst)

        total_original_pairs = 0
        fully_settled_pairs = 0
        total_intended_value = 0.0
        total_settled_value = 0.0


        for linkcode, pair in original_pairs.items():

            if not pair:
                #should never be empty, just in case
                continue

            # We include both single leg pairs as double leg pairs in the denominator of calculation.
            intended_amount = pair[0].get_amount()
            total_original_pairs += 1
            total_intended_value += intended_amount

            if len(pair) <2:
                #end here for instruction without a counter instruction, won't be settled anyway
                continue

            # Case 1: Fully settled directly (both instructions settled on time or late).
            if (pair[0].get_status() in ["Settled on time"] and
                    pair[1].get_status() in ["Settled on time"]):
                fully_settled_pairs += 1
                total_settled_value += intended_amount

            # Case 2: Partial settlement – both instructions were cancelled due to partial settlement.
            elif (pair[0].get_status() == "Cancelled due to partial settlement" and
                  pair[1].get_status() == "Cancelled due to partial settlement"):
                # Recursively sum settled amounts from child instructions (and their descendants).
                settled_child_value = (self.get_recursive_settled_amount(parent_instruction=pair[0],instruction_pool=relevant_instructions,depth=0))
                # The effective settled amount is capped at the intended amount.
                effective_settled = min(settled_child_value, intended_amount)
                total_settled_value += effective_settled
                # Count the pair as fully settled if the effective settled value equals the intended value.
                if effective_settled == intended_amount:
                    fully_settled_pairs += 1
            # Other statuses are considered as not settled.

        instruction_efficiency = (fully_settled_pairs / total_original_pairs * 100) if total_original_pairs > 0 else 0
        value_efficiency = (total_settled_value / total_intended_value * 100) if total_intended_value > 0 else 0

        return instruction_efficiency, value_efficiency

    def calculate_settlement_efficiency_optimized(self):
        """
        Optimized version of settlement efficiency calculation that:
        1. Pre-computes child relationships
        2. Uses efficient data structures
        3. Eliminates redundant calculations
        4. Uses iteration instead of recursion

        Returns:
            tuple: (instruction_efficiency_percentage, value_efficiency_percentage)
        """
        # Get all relevant instructions only once
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()

        # Pre-build parent-child mapping
        child_map = {}
        status_cache = {}
        amount_cache = {}

        for inst in relevant_instructions:
            # Cache instruction properties
            inst_id = inst.get_uniqueID()
            status_cache[inst_id] = inst.get_status()
            amount_cache[inst_id] = inst.get_amount()

            # Build child relationships
            if inst.isChild:
                parent_id = inst.get_motherID()
                if parent_id not in child_map:
                    child_map[parent_id] = []
                child_map[parent_id].append(inst_id)

        # Group mothers by linkcode
        original_pairs = {}
        mother_instructions = []

        for inst in relevant_instructions:
            if inst.get_motherID() == "mother":
                mother_instructions.append(inst)
                linkcode = inst.get_linkcode()
                if linkcode not in original_pairs:
                    original_pairs[linkcode] = []
                original_pairs[linkcode].append(inst)

        # Initialize counters
        total_original_pairs = 0
        fully_settled_pairs = 0
        total_intended_value = 0.0
        total_settled_value = 0.0

        # Pre-compute settled amounts for all parent instructions that will need it
        # (specifically, those with "Cancelled due to partial settlement" status)
        cached_settled_amounts = {}
        for inst in mother_instructions:
            parent_id = inst.get_uniqueID()
            if status_cache.get(parent_id) == "Cancelled due to partial settlement":
                if parent_id not in cached_settled_amounts:
                    settled_amount = self.get_settled_amount_iterative(
                        parent_id, child_map, status_cache, amount_cache
                    )
                    cached_settled_amounts[parent_id] = settled_amount

        # Process all pairs
        for linkcode, pair in original_pairs.items():
            if not pair:
                continue

            # Process each pair only once
            intended_amount = pair[0].get_amount()
            total_original_pairs += 1
            total_intended_value += intended_amount

            if len(pair) < 2:
                # Skip instructions without a counter instruction
                continue

            # Get cached statuses
            pair_0_id = pair[0].get_uniqueID()
            pair_1_id = pair[1].get_uniqueID()
            pair_0_status = status_cache[pair_0_id]
            pair_1_status = status_cache[pair_1_id]

            # Case 1: Fully settled directly
            if (pair_0_status in ["Settled on time"] and
                    pair_1_status in ["Settled on time"]):
                fully_settled_pairs += 1
                total_settled_value += intended_amount

            # Case 2: Partial settlement
            elif (pair_0_status == "Cancelled due to partial settlement" and
                  pair_1_status == "Cancelled due to partial settlement"):

                # Use pre-computed settled amount or compute it if not in cache
                if pair_0_id in cached_settled_amounts:
                    settled_child_value = cached_settled_amounts[pair_0_id]
                else:
                    # Compute on-demand if somehow not in cache
                    settled_child_value = self.get_settled_amount_iterative(
                        pair_0_id, child_map, status_cache, amount_cache
                    )
                    cached_settled_amounts[pair_0_id] = settled_child_value

                # The effective settled amount is capped at the intended amount
                effective_settled = min(settled_child_value, intended_amount)
                total_settled_value += effective_settled

                # Count as fully settled if effective settled equals intended amount
                if effective_settled == intended_amount:
                    fully_settled_pairs += 1

        # Calculate final metrics
        instruction_efficiency = (fully_settled_pairs / total_original_pairs * 100) if total_original_pairs > 0 else 0
        value_efficiency = (total_settled_value / total_intended_value * 100) if total_intended_value > 0 else 0

        return instruction_efficiency, value_efficiency

    def count_settled_instructions(self):
        """
        Counts the total number of instructions that reached "Settled on time" status
        during the main simulation period.

        Returns:
            int: The total count of settled instructions, includes mothers & children
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants()
        return sum(1 for inst in relevant_instructions if inst.get_status() in ["Settled on time", "Settled late"])

    def get_total_settled_amount(self):
        """
        Calculates the total amount that was settled during the main simulation period.
        This includes both directly settled instructions and recursively settled amounts
        through child instructions.

        Returns:
            int: The total settled amount
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants()
        total_settled_amount = 0

        # First, add amounts from directly settled instructions
        for inst in relevant_instructions:
            if inst.get_status() in ["Settled on time", "Settled late"]:
                total_settled_amount += inst.get_amount()

        return total_settled_amount



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



    def get_avg_instruction_age_before_settlement(self):
        relevant_instructions = self.get_main_period_mothers_and_descendants()
        settled = [inst for inst in relevant_instructions if inst.get_status() in ["Settled on time", "Settled late"]]
        ages = [(self.simulated_time - inst.get_creation_time()).total_seconds() / 3600 for inst in settled]
        return round(sum(ages) / len(ages), 2) if ages else 0

    def get_original_pair_count(self):
        relevant_instructions = self.get_main_period_mothers_and_descendants()
        mothers = [inst for inst in relevant_instructions if inst.get_motherID() == "mother"]
        return len(mothers) // 2  # 2 instructions per pair

    def get_partial_settlement_count(self):
        relevant_instructions = self.get_main_period_mothers_and_descendants()
        return sum(1 for inst in relevant_instructions if inst.get_status() == "Cancelled due to partial settlement")

    def get_error_cancellation_count(self):
        relevant_instructions = self.get_main_period_mothers_and_descendants()
        return sum(1 for inst in relevant_instructions if inst.get_status() == "Cancelled due to error")

    def get_average_tree_depth(self):
        relevant_instructions = self.get_main_period_mothers_and_descendants()
        mothers = [inst for inst in relevant_instructions if inst.get_motherID() == "mother"]

        def get_depth(inst):
            children = [child for child in relevant_instructions if child.isChild and child.motherID == inst.get_uniqueID()]
            if not children:
                return 1
            return 1 + max(get_depth(child) for child in children)


        depths = [get_depth(mother) for mother in mothers]
        return round(sum(depths) / len(depths), 2) if depths else 0

    def generate_depth_statistics(self):
        """
        Generate statistics about instruction depth distribution for process mining visualization.
        """
        # Dictionary to store counts by depth
        relevant_instructions = self.get_main_period_mothers_and_descendants()
        depth_counts = {}
        # Dictionary to store counts by depth and status
        depth_status_counts = {}
        # Dictionary to store parent-child relationships for tree building
        parent_child_map = {}

        for inst in relevant_instructions:
            depth = inst.get_depth()
            status = inst.get_status()

            # Count by depth
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

            # Count by depth and status
            if depth not in depth_status_counts:
                depth_status_counts[depth] = {}
            depth_status_counts[depth][status] = depth_status_counts[depth].get(status, 0) + 1

        return {
            "depth_counts": depth_counts,
            "depth_status_counts": depth_status_counts,
            "original_pairs":   self.get_original_pair_count(),
            "partial_settlements": self.get_partial_settlement_count(),
            "avg_tree_depth": self.get_average_tree_depth(),
            "cancellations_due_to_error": self.get_error_cancellation_count(),
        }

    def save_depth_statistics(self, filename="depth_statistics.json"):
        """Save depth statistics to a JSON file for external visualization"""
        import json

        stats = self.generate_depth_statistics()
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Depth statistics saved to {filename}")

