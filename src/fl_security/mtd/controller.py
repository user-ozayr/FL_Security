"""Round-scoped MTD for client sampling and aggregation committee routing."""
import numpy as np
from fl_security.mtd.client_selection import select_clients

class MTDController:
    def __init__(self, config: dict):
        self.config = config.get("mtd", {})
        self.enabled = bool(self.config.get("enabled", False))
        self.selection_strategy = self.config.get("selection_strategy", "random")
        self.committee_size = int(self.config.get("committee_size", 3))
        self.seed = int(config.get("experiment", {}).get("seed", 42))
        smpc = config.get("smpc", {})
        self.num_aggregation_parties = int(smpc.get(
            "num_aggregation_parties",
            max(self.committee_size, config.get("federated_learning", {}).get("num_clients", 1))
        ))
        if self.committee_size < 2 or self.num_aggregation_parties < 2:
            raise ValueError("secure aggregation committees and party count must be >= 2")

    def generate_round_config(self, round_num: int, num_clients: int, fraction: float) -> dict:
        if round_num <= 0:
            raise ValueError("round_num must be positive")
        moving = self.enabled and self.config.get("change_every_round", True)
        round_seed = self.seed + round_num if moving else self.seed
        strategy = self.selection_strategy if self.enabled else "fixed"
        selected = select_clients(num_clients, fraction, strategy=strategy, seed=round_seed)
        rng = np.random.default_rng(round_seed + 10_000)
        size = min(self.committee_size, self.num_aggregation_parties)
        committees = {
            int(cid): sorted(int(pid) for pid in rng.choice(self.num_aggregation_parties, size, replace=False))
            for cid in selected
        }
        return {
            "round": round_num,
            "selected_clients": selected,
            "committees": committees,
            "seed": round_seed,
            "num_aggregation_parties": self.num_aggregation_parties,
        }
