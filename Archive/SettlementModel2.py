from mesa import Model
from datetime import datetime, timedelta
import InstitutionAgent
import Account
import random
from jsonocellogger import JSONOCELLogger
from statistics_tracker import SettlementStatisticsTracker
from collections import defaultdict


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
        self.statistics_tracker = SettlementStatisticsTracker()
        self.num_institutions = 10
        self.min_total_accounts = 4
        self.max_total_accounts = 10
        self.simulation_duration_days = 15 #number of measured days (so simulation is longer)
        self.min_settlement_percentage = 0.05
        self.max_child_depth = 15
        self.bond_types = ["Bond-A", "Bond-B", "Bond-C", "Bond-D", "Bond-E", "Bond-F", "Bond-G", "Bond H", "Bond I"]
        self.logger = JSONOCELLogger()
        self.log_only_main_events= True
        self.seed = seed
        if seed is not None:
            random.seed(seed)  # Set the random seed
            print(f"[INFO] Random seed set to: {seed}")



        self.simulation_start = datetime(2025, 4, 1, 1, 30)
        self.warm_up_period = timedelta(days=15)
        self.cool_down_period = timedelta(days=15)

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
        self.mini_batch_times = [
            timedelta(hours=h, minutes=0)
            for h in range(2, 20, 3)
        ]
        self.mini_batches_processed = set()

        # Instruction indices for fast lookup
        self.validated_delivery_instructions = {}  # linkcode -> list of delivery instructions
        self.validated_receipt_instructions = {}  # linkcode -> list of receipt instructions

        #OCEL logging reqs:

        self.event_counter = 1
        self.event_log = []  # List to store OCEL events => probably redudant

        # Add a cache for relevant instructions
        self.relevant_instructions_cache = None
        self.relevant_instructions_id_set = None
        self.last_cache_update = None

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
        if not self.log_only_main_events:
            return

        if attributes is None:
            attributes = {}

        timestamp_iso = self.simulated_time.isoformat() + "Z"

        self.logger.log_event(
            event_type=event_type,
            object_ids=object_ids,
            event_attributes=attributes,
            timestamp=timestamp_iso
        )
        if event_type in ["Settled On Time", "Settled Late"]:
            self.statistics_tracker.classify_settlement(
                event_type=event_type,
                event_timestamp_str=timestamp_iso,
                lateness_hours=attributes.get("lateness_hours"),
                depth=attributes.get("depth")
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
        Samples an instruction amount (in EUR) from a log-normal distribution
        that approximates the mean (€324M), std (€829M), and median (€20M)
        of the original two-point distribution.
        """
        mu = 18.0857  # ln(median)
        sigma = 1  # controls skewness and std
        amount = random.lognormvariate(mu, sigma)

        # Apply a cap to further prevent extreme outliers (optional)
        cap = 2e9  # 2 billion cap
        amount = min(amount, cap)

        return int(amount)

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
            new_cash_balance =  int(random.uniform(6e11, 9e11))  # Increased balance range
            new_cash_creditLimit = 0.1*new_cash_balance
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

                new_security_balance = int(random.uniform(13e8, 16e8))
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

        # --- Phase 3: Match all ---
        # Process one linkcode at a time where both delivery and receipt instructions exist
        common_linkcodes = set(self.validated_delivery_instructions.keys()) & set(
            self.validated_receipt_instructions.keys())

        for linkcode in common_linkcodes:
            delivery_list = self.validated_delivery_instructions.get(linkcode, [])
            if delivery_list:  # If there are delivery instructions with this linkcode
                delivery = delivery_list[0]  # Take the first one
                if delivery.get_status() == "Validated":
                    delivery.match()

        # Continue with settlement phase (5 attempts)
        for attempt in range(5):
            for transaction in self.transactions:
                if transaction.get_status() == "Matched":
                    if (transaction.get_deliverer().get_securitiesAccount().get_newSecurities()):
                        if self.simulated_time >= transaction.get_deliverer().get_intended_settlement_date() - timedelta(days=1,minutes=35):
                            transaction.settle()

    def mini_batch_settlement(self):
        print(f"[INFO] Running mini-batch settlement at {self.simulated_time}")
        for transaction in self.transactions:
            if transaction.get_status() == "Matched":
                if (transaction.get_deliverer().get_securitiesAccount().get_newSecurities()):
                    if self.simulated_time >= transaction.get_deliverer().get_intended_settlement_date() - timedelta(days=1, minutes=30):
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
            int: Total settled amount from all descendants
        """
        total = 0
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
                settled_child_value = (self.get_recursive_settled_amount(parent_instruction=pair[0],instruction_pool=relevant_instructions))
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

            if len(pair) <2:
                continue

            # Process each pair only once
            intended_amount = pair[0].get_amount()
            total_original_pairs += 1
            total_intended_value += intended_amount

            #if len(pair) < 2:
                # Skip instructions without a counter instruction
            #    continue

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

    def get_days_with_no_pairs(self):
        """
        Identify dates in the main period that have no settlement pairs at all.
        Returns:
            list of dates (datetime.date) with zero mother-instructions.
        """
        # collect all dates with mother instructions
        rel = self.get_main_period_mothers_and_descendants_optimized()
        present_dates = {inst.get_intended_settlement_date().date()
                         for inst in rel if inst.get_motherID() == 'mother'}


        # build full range from min to max
        start, end = min(present_dates), max(present_dates)
        missing = []
        current = start
        while current <= end:
            if current not in present_dates:
                missing.append(current)
            current += timedelta(days=1)
        return missing

    from collections import defaultdict
    from datetime import datetime

    # Refactored SettlementModel efficiency methods using grouped-first approach

    from collections import defaultdict
    from datetime import timedelta

    # Refactored SettlementModel efficiency methods for consistent output with optimized logic

    from collections import defaultdict
    from datetime import timedelta

    # Refactored SettlementModel efficiency methods for consistent output with optimized logic

    from collections import defaultdict
    from datetime import timedelta

    # Refactored SettlementModel efficiency methods for consistent output with optimized logic

    from collections import defaultdict
    from datetime import timedelta

    # Refactored SettlementModel efficiency methods for consistent output with optimized logic

    from collections import defaultdict
    from datetime import timedelta

    # SettlementModel efficiency methods

    from collections import defaultdict
    from datetime import datetime, timedelta

    # SettlementModel efficiency methods

    def calculate_settlement_efficiency_per_day_optimized(self):
        """
        Calculate instruction and value efficiency per day, mirroring calculate_settlement_efficiency_optimized.
        Only include days with at least one complete pair (i.e., both legs on that date).
        Includes all single-leg pairs in denominators, but only complete pairs (and on-time children) in numerators.
        Returns:
            dict: {date_str: (instruction_eff_pct, value_eff_pct)} sorted by date
        """
        # Cache all relevant instructions
        rel = self.get_main_period_mothers_and_descendants_optimized()
        status_cache = {}
        amount_cache = {}
        child_map = defaultdict(list)
        mothers = []
        for inst in rel:
            iid = inst.get_uniqueID()
            status_cache[iid] = inst.get_status()
            amount_cache[iid] = inst.get_amount()
            if inst.isChild:
                child_map[inst.get_motherID()].append(iid)
            elif inst.get_motherID() == 'mother':
                mothers.append(inst)

        # Group mother instructions by intended settlement date
        mothers_by_date = defaultdict(list)
        for m in mothers:
            date = m.get_intended_settlement_date().date()
            mothers_by_date[date].append(m)

        results = {}
        for date in sorted(mothers_by_date):
            pairs_by_link = defaultdict(list)
            for m in mothers_by_date[date]:
                pairs_by_link[m.get_linkcode()].append(m)

            total_pairs = complete_pairs = fully_settled = 0
            total_value = settled_value = 0.0

            for pair in pairs_by_link.values():
                if len(pair) <2:
                    continue
                total_pairs += 1
                m0 = pair[0]
                intended = amount_cache[m0.get_uniqueID()]
                total_value += intended
                if len(pair) < 2:
                    continue
                complete_pairs += 1
                m1 = pair[1]
                s0 = status_cache[m0.get_uniqueID()]
                s1 = status_cache[m1.get_uniqueID()]
                if s0 == 'Settled on time' and s1 == 'Settled on time':
                    fully_settled += 1
                    settled_value += intended
                elif s0 == 'Cancelled due to partial settlement' and s1 == 'Cancelled due to partial settlement':
                    child_amt = sum(
                        amount_cache[cid]
                        for cid in child_map.get(m0.get_uniqueID(), [])
                        if status_cache.get(cid) == 'Settled on time'
                    )
                    settled_value += child_amt
                    if child_amt >= intended:
                        fully_settled += 1

            if complete_pairs == 0:
                continue

            instr_eff = (fully_settled / total_pairs * 100)
            val_eff = (settled_value / total_value * 100)
            results[date.isoformat()] = (instr_eff, val_eff)

        return results

    def calculate_settlement_efficiency_by_participant(self):
        """
        Calculate instruction and value efficiency per participant, mirroring calculate_settlement_efficiency_optimized.
        Includes all single-leg pairs in denominators, but only complete pairs (and on-time children) in numerators.
        Returns:
            dict: {participantID: (instruction_eff_pct, value_eff_pct)} sorted by participantID
        """
        rel = self.get_main_period_mothers_and_descendants_optimized()
        status_cache = {}
        amount_cache = {}
        child_map = defaultdict(list)
        mothers = []
        for inst in rel:
            iid = inst.get_uniqueID()
            status_cache[iid] = inst.get_status()
            amount_cache[iid] = inst.get_amount()
            if inst.isChild:
                child_map[inst.get_motherID()].append(iid)
            elif inst.get_motherID() == 'mother':
                mothers.append(inst)

        pairs_by_link = defaultdict(list)
        for m in mothers:
            pairs_by_link[m.get_linkcode()].append(m)

        participants = sorted({m.get_institution().institutionID for m in mothers})
        stats = {pid: {'pairs': 0, 'settled_pairs': 0, 'value': 0.0, 'settled_value': 0.0} for pid in participants}

        for pair in pairs_by_link.values():
            if not pair:
                continue
            if len(pair) <2:
                continue
            intended = amount_cache[pair[0].get_uniqueID()]
            pids = [m.get_institution().institutionID for m in pair]
            for pid in pids:
                stats[pid]['pairs'] += 1
                stats[pid]['value'] += intended
            if len(pair) < 2:
                continue
            m0, m1 = pair[0], pair[1]
            s0 = status_cache[m0.get_uniqueID()]
            s1 = status_cache[m1.get_uniqueID()]
            for pid in pids:
                if s0 == 'Settled on time' and s1 == 'Settled on time':
                    stats[pid]['settled_pairs'] += 1
                    stats[pid]['settled_value'] += intended
                elif s0 == 'Cancelled due to partial settlement' and s1 == 'Cancelled due to partial settlement':
                    child_amt = sum(
                        amount_cache[cid]
                        for cid in child_map.get(m0.get_uniqueID(), [])
                        if status_cache.get(cid) == 'Settled on time'
                    )
                    stats[pid]['settled_value'] += child_amt
                    if child_amt >= intended:
                        stats[pid]['settled_pairs'] += 1

        results = {}
        for pid in participants:
            c = stats[pid]
            instr_eff = (c['settled_pairs'] / c['pairs'] * 100) if c['pairs'] else 0.0
            val_eff = (c['settled_value'] / c['value'] * 100) if c['value'] else 0.0
            results[pid] = (instr_eff, val_eff)

        return results

    def calculate_average_daily_efficiency(self):
        """
        Compute the weighted instruction and value efficiency across all days,
        by pulling per-day metrics and weighting them by the number of pairs and total value per day.
        Returns:
            tuple: (instruction_eff_pct, value_eff_pct)
        """
        daily_metrics = self.calculate_settlement_efficiency_per_day_optimized()
        if not daily_metrics:
            return 0.0, 0.0
        rel = self.get_main_period_mothers_and_descendants_optimized()
        amount_cache = {inst.get_uniqueID(): inst.get_amount() for inst in rel}
        mothers_by_date = defaultdict(list)
        for inst in rel:
            if inst.get_motherID() == 'mother':
                mothers_by_date[inst.get_intended_settlement_date().date()].append(inst)

        instr_num = instr_den = val_num = val_den = 0.0
        for date_str, (eff_instr, eff_val) in daily_metrics.items():
            date = datetime.fromisoformat(date_str).date()
            pairs_by_link = defaultdict(list)
            for m in mothers_by_date.get(date, []):
                pairs_by_link[m.get_linkcode()].append(m)
            num_pairs = len(pairs_by_link)
            total_value = sum(amount_cache[pair[0].get_uniqueID()] for pair in pairs_by_link.values())
            instr_num += eff_instr * num_pairs
            instr_den += num_pairs
            val_num += eff_val * total_value
            val_den += total_value

        instr_eff = (instr_num / instr_den) if instr_den else 0.0
        val_eff = (val_num / val_den) if val_den else 0.0
        return instr_eff, val_eff

    def calculate_average_participant_efficiency(self):
        """
        Compute the weighted instruction and value efficiency across all participants,
        by pulling per-participant metrics and weighting each trade equally (split among participants).
        Returns:
            tuple: (instruction_eff_pct, value_eff_pct)
        """
        part_metrics = self.calculate_settlement_efficiency_by_participant()
        if not part_metrics:
            return 0.0, 0.0
        rel = self.get_main_period_mothers_and_descendants_optimized()
        amount_cache = {inst.get_uniqueID(): inst.get_amount() for inst in rel}
        original_pairs = defaultdict(list)
        for inst in rel:
            if inst.get_motherID() == 'mother':
                original_pairs[inst.get_linkcode()].append(inst)

        instr_num = instr_den = val_num = val_den = 0.0
        for link, pair in original_pairs.items():
            if not pair:
                continue
            intended = amount_cache[pair[0].get_uniqueID()]
            pids = [m.get_institution().institutionID for m in pair]
            weight = 1.0 / len(pids)
            for pid in pids:
                eff_instr, eff_val = part_metrics.get(pid, (0.0, 0.0))
                instr_num += eff_instr * weight
                instr_den += weight
                val_num += eff_val * intended * weight
                val_den += intended * weight
        instr_eff = (instr_num / instr_den) if instr_den else 0.0
        val_eff = (val_num / val_den) if val_den else 0.0
        return instr_eff, val_eff

    def count_settled_instructions(self):
        """
        Counts the total number of instructions that reached "Settled on time" status
        during the main simulation period.

        Returns:
            int: The total count of settled instructions, includes mothers & children
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        return sum(1 for inst in relevant_instructions if inst.get_status() in ["Settled on time", "Settled late"])

    def count_effectively_settled_mother_instructions(self):
        """
         Counts the number of mother instructions that were effectively settled,
        either directly or via child settlements, using pre-cached relationships.

        Returns:
            int: Number of effectively settled mother instructions
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()

        # Pre-build maps and caches
        child_map = {}
        status_cache = {}
        amount_cache = {}

        for inst in relevant_instructions:
            inst_id = inst.get_uniqueID()
            status_cache[inst_id] = inst.get_status()
            amount_cache[inst_id] = inst.get_amount()
            if inst.isChild:
                parent_id = inst.get_motherID()
                if parent_id not in child_map:
                    child_map[parent_id] = []
                child_map[parent_id].append(inst_id)

        # Identify mothers
        mothers = [
            inst for inst in relevant_instructions
            if inst.get_motherID() == "mother"
        ]

        # Pre-compute settled child values for mothers with partial cancellations
        cached_settled_amounts = {}
        for mother in mothers:
            mother_id = mother.get_uniqueID()
            if status_cache.get(mother_id) == "Cancelled due to partial settlement":
                settled = self.get_settled_amount_iterative(
                    parent_id=mother_id,
                    child_map=child_map,
                    status_cache=status_cache,
                    amount_cache=amount_cache
                )
                cached_settled_amounts[mother_id] = settled

        # Now count effectively settled mothers
        effectively_settled_count = 0

        for mother in mothers:
            mother_id = mother.get_uniqueID()
            status = status_cache.get(mother_id, "")
            intended_amount = amount_cache.get(mother_id, 0)

            if status in ["Settled on time", "Settled late"]:
                effectively_settled_count += 1
            elif status == "Cancelled due to partial settlement":
                settled_child_amount = cached_settled_amounts.get(mother_id, 0)
                if settled_child_amount >= intended_amount:
                    effectively_settled_count += 1

        return effectively_settled_count

    def get_total_settled_amount(self):
        """
        Calculates the total amount that was settled during the main simulation period.
        This includes both directly settled instructions and recursively settled amounts
        through child instructions.

        Returns:
            int: The total settled amount
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        total_settled_amount = 0

        # First, add amounts from directly settled instructions
        for inst in relevant_instructions:
            if inst.get_status() in ["Settled on time", "Settled late"]:
                total_settled_amount += inst.get_amount()

        return total_settled_amount



    #def print_settlement_efficiency(self):
    #    """
    #    Quickly prints the settlement efficiency metrics.
    #    """
    #    instruction_eff, value_eff = self.calculate_settlement_efficiency()
    #    print("Settlement Efficiency:")
    #    print("  Instruction Efficiency: {:.2f}%".format(instruction_eff))
    #    print("  Value Efficiency: {:.2f}%".format(value_eff))

    #def save_settlement_efficiency_to_csv(self, filename="settlement_efficiency.csv"):
    #    """
    #    Saves the settlement efficiency metrics to a CSV file.
    #    """
    #    instruction_eff, value_eff = self.calculate_settlement_efficiency()
    #    data = [
    #        {"Metric": "Instruction Efficiency (%)", "Value": round(instruction_eff, 2)},
    #        {"Metric": "Value Efficiency (%)", "Value": round(value_eff, 2)}
    #    ]
    #    df = pd.DataFrame(data)
    #    df.to_csv(filename, index=False)
    #    print(f"Settlement efficiency metrics saved to {filename}")

    def get_settled_on_time_amount(self):
        """
        Calculates the total amount settled on time during the main simulation period.
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        total = sum(inst.get_amount() for inst in relevant_instructions if inst.get_status() == "Settled on time")
        return total / 2

    def get_settled_late_amount(self):
        """
        Calculates the total amount settled late during the main simulation period.
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        total = sum(inst.get_amount() for inst in relevant_instructions if inst.get_status() == "Settled late")
        return total / 2

    def get_cancelled_partial_amount(self):
        """
        Calculates the total amount of instructions cancelled due to partial settlement during the main simulation period.
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        total = sum(inst.get_amount() for inst in relevant_instructions if
                    inst.get_status() == "Cancelled due to partial settlement")
        return total / 2

    def get_cancelled_error_amount(self):
        """
        Calculates the total amount of instructions cancelled due to error during the main simulation period.
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        total = sum(
            inst.get_amount() for inst in relevant_instructions if inst.get_status() == "Cancelled due to error")
        return total / 2

    def get_avg_instruction_age_before_settlement(self):
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        settled = [inst for inst in relevant_instructions if inst.get_status() in ["Settled on time", "Settled late"]]
        ages = [(self.simulated_time - inst.get_creation_time()).total_seconds() / 3600 for inst in settled]
        return round(sum(ages) / len(ages), 2) if ages else 0

    def get_original_pair_count(self):
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        mothers = [inst for inst in relevant_instructions if inst.get_motherID() == "mother"]
        linkcodes = set(inst.get_linkcode() for inst in mothers)
        return len(linkcodes)

    def get_original_instruction_count(self):
        """
        Returns the number of original mother instructions
        created during the main simulation period.
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        mothers = [inst for inst in relevant_instructions if inst.get_motherID() == "mother"]
        return len(mothers)

    def get_total_intended_amount_from_pairs(self):
        """
        Returns the total intended settlement amount
        based on original instruction pairs.
        Only one side of each pair is counted.
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        mothers = [inst for inst in relevant_instructions if inst.get_motherID() == "mother"]

        # Group mothers by linkcode
        linkcode_to_mothers = {}
        for inst in mothers:
            linkcode = inst.get_linkcode()
            if linkcode not in linkcode_to_mothers:
                linkcode_to_mothers[linkcode] = []
            linkcode_to_mothers[linkcode].append(inst)

        total_amount = 0
        for pair in linkcode_to_mothers.values():
            total_amount += pair[0].get_amount()


        return total_amount

    def get_child_instruction_count(self):
        """
        Returns the number of child instructions created
        during the main simulation period.
        """
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        children = [inst for inst in relevant_instructions if inst.isChild]
        return len(children)

    def get_partial_settlement_count(self):
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        return sum(1 for inst in relevant_instructions if inst.get_status() == "Cancelled due to partial settlement")

    def get_error_cancellation_count(self):
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
        return sum(1 for inst in relevant_instructions if inst.get_status() == "Cancelled due to error")

    def get_average_tree_depth(self):
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
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
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
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

    def generate_depth_counts_and_depth_status_counts(self):
        """
        Generate statistics about instruction depth distribution for process mining visualization.
        """
        # Dictionary to store counts by depth
        relevant_instructions = self.get_main_period_mothers_and_descendants_optimized()
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

        return  depth_counts,depth_status_counts

    def export_all_statistics(self, filename: str, config_metadata: dict = None):
        """
        Unified export method for all analyses.
        Exports flat JSON with all important simulation metrics.
        Accepts optional metadata (e.g., seed, runtime, memory) to include in output.
        """
        import json
        depth_counts, depth_status_counts = self.generate_depth_counts_and_depth_status_counts()

        result = {
            # Efficiency & core metrics
            "instruction_efficiency": self.calculate_settlement_efficiency_optimized()[0],
            "value_efficiency": self.calculate_settlement_efficiency_optimized()[1],
            "original_pairs": self.get_original_pair_count(),
            "original_instructions_count": self.get_original_instruction_count(),
            "child_instruction_count": self.get_child_instruction_count(),
            "mothers_effectively_settled": self.count_effectively_settled_mother_instructions(),
            "intended_amount": self.get_total_intended_amount_from_pairs(),

            # Cancellations
            "partial_settlements": self.get_partial_settlement_count(),
            "cancellations_due_to_error": self.get_error_cancellation_count(),
            "cancelled_due_to_partial_settlement_amount": self.get_cancelled_partial_amount(),
            "cancelled_due_to_error_amount": self.get_cancelled_error_amount(),

            # Amounts settled
            "settled_on_time_amount": self.get_settled_on_time_amount(),
            "settled_late_amount": self.get_settled_late_amount(),

            # Volume & dynamics
            "instructions_settled_total": self.count_settled_instructions(),
            "avg_instruction_age_hours": self.get_avg_instruction_age_before_settlement(),
            "avg_tree_depth": self.get_average_tree_depth(),

            # Depth distributions
            "depth_counts": depth_counts,
            "depth_status_counts": depth_status_counts,
        }

        result.update(self.statistics_tracker.export_summary())

        # Add daily metrics
        daily_metrics = {}
        for date, (instr_eff, val_eff) in self.calculate_settlement_efficiency_per_day_optimized().items():
            key = date.isoformat() if hasattr(date, 'isoformat') else str(date)
            daily_metrics[key] = {
                "instruction_efficiency": instr_eff,
                "value_efficiency": val_eff
            }
        result["daily_metrics"] = daily_metrics
        result["average_daily"] = self.calculate_average_daily_efficiency()

        # Add participant metrics
        participant_metrics = {}
        for pid, (instr_eff, val_eff) in self.calculate_settlement_efficiency_by_participant().items():
            participant_metrics[pid] = {
                "instruction_efficiency": instr_eff,
                "value_efficiency": val_eff
            }
        result["participant_metrics"] = participant_metrics
        result["average_particpant"] = self.calculate_average_participant_efficiency()
        result["missing_days"] = self.get_days_with_no_pairs()

        # Optional metadata: seed, runtime, memory, config info
        if config_metadata:
            result.update(config_metadata)

        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)



