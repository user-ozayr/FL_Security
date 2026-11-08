import numpy as np
from fl_security.mtd.client_selection import select_clients

class MTDController:
    def __init__(self, config):
        self.config = config.get("mtd", {})
        self.enabled = self.config.get("enabled", False)
        self.selection_strategy = self.config.get("selection_strategy", "random")
        self.committee_size = self.config.get("committee_size", 3)
        self.seed = config["experiment"].get("seed", 42)

    def generate_round_config(self, round_num, num_clients, fraction):
        seed = self.seed + round_num if self.enabled else self.seed
        selected_clients = select_clients(num_clients, fraction, self.selection_strategy, seed=seed)
        
        # Committee assignment
        np.random.seed(seed)
        committees = {}
        for client_id in selected_clients:
            committee_members = np.random.choice(num_clients, size=min(self.committee_size, num_clients), replace=False).tolist()
            committees[client_id] = committee_members
            
        return {
            "round": round_num,
            "selected_clients": selected_clients,
            "committees": committees,
            "seed": seed
        }
