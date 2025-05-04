from collections import defaultdict
from datetime import datetime, time

class SettlementStatisticsTracker:
    def __init__(self):
        # RTP and Batch Counters
        self.settled_ontime_rtp = 0
        self.settled_ontime_batch = 0
        self.settled_late_rtp = 0
        self.settled_late_batch = 0

        # Lateness Tracking
        self.lateness_hours_total = 0.0
        self.lateness_count = 0
        self.lateness_by_depth_records = defaultdict(list)

        # Add these new tracking variables
        self.normal_settlements_count = 0
        self.partial_settlements_count = 0
        self.normal_settled_amount = 0
        self.partial_settled_amount = 0

    def classify_settlement(self, event_type, event_timestamp_str, lateness_hours=None, depth=None, is_child=False, amount=0):
        """
        Call this method for each settlement event (on-time or late).
        """
        try:
            event_time = datetime.fromisoformat(event_timestamp_str.replace("Z", "+00:00")).time()
        except Exception:
            return  # Skip bad timestamps safely

        is_rtp = time(1, 30) <= event_time <= time(19, 30)
        is_batch = event_time >= time(22, 0)

        # Track by settlement type (normal vs partial)
        if is_child:
            self.partial_settlements_count += 1
            self.partial_settled_amount += amount
        else:
            self.normal_settlements_count += 1
            self.normal_settled_amount += amount

        if event_type.lower() == "settled on time":
            if is_rtp:
                self.settled_ontime_rtp += 1
            elif is_batch:
                self.settled_ontime_batch += 1

        elif event_type.lower() == "settled late":
            if is_rtp:
                self.settled_late_rtp += 1
            elif is_batch:
                self.settled_late_batch += 1

            # Only Late events contribute lateness hours
            if lateness_hours is not None:
                try:
                    lateness = float(lateness_hours)
                    self.lateness_hours_total += lateness
                    self.lateness_count += 1
                    if depth is not None:
                        self.lateness_by_depth_records[int(depth)].append(lateness)
                except Exception:
                    pass

    def export_summary(self):
        """
        Call this at the end of simulation to get full summary stats ready for export.
        """
        lateness_by_depth_avg = {
            str(depth): (sum(hours) / len(hours)) for depth, hours in self.lateness_by_depth_records.items()
        } if self.lateness_by_depth_records else {}

        summary = {
            "settled_ontime_rtp": self.settled_ontime_rtp,
            "settled_ontime_batch": self.settled_ontime_batch,
            "settled_late_rtp": self.settled_late_rtp,
            "settled_late_batch": self.settled_late_batch,
            "lateness_hours_total": self.lateness_hours_total,
            "lateness_hours_average": (self.lateness_hours_total / self.lateness_count) if self.lateness_count > 0 else 0,
            "lateness_by_depth": lateness_by_depth_avg,

            # Settlement type statistics (new)
            "normal_settlements_count": self.normal_settlements_count,
            "partial_settlements_count": self.partial_settlements_count,
            "normal_settled_amount": self.normal_settled_amount,
            "partial_settled_amount": self.partial_settled_amount
        }
        return summary
