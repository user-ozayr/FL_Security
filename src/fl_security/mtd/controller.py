"""Round-scoped MTD with balanced committees and secret-keyed randomness."""
import hashlib
import hmac
import os

from fl_security.mtd.client_selection import select_clients
from fl_security.mtd.randomness import HMACCounterRNG


class MTDController:
    def __init__(self, config: dict):
        self.config = config.get("mtd", {})
        self.enabled = bool(self.config.get("enabled", False))
        self.selection_strategy = self.config.get("selection_strategy", "random")
        self.committee_size = int(self.config.get("committee_size", 3))
        configured_seed = int(config.get("experiment", {}).get("seed", 42))

        # The live schedule uses an HMAC-derived PRF key. Without this secret,
        # schedules are reproducible simulation artifacts and are not secret.
        secret = os.environ.get("FL_MTD_SECRET", "")
        material = f"fl-security-mtd-seed-v1:{configured_seed}".encode("utf-8")
        if secret:
            secret_bytes = secret.encode("utf-8")
            if len(secret_bytes) < 32:
                raise ValueError("FL_MTD_SECRET must contain at least 32 bytes")
            self.seed = int.from_bytes(
                hmac.new(secret_bytes, material, hashlib.sha256).digest()[:8],
                byteorder="big",
                signed=False,
            )
            self._random_key = hmac.new(
                secret_bytes, b"fl-security-mtd-rng-v1:" + material, hashlib.sha256
            ).digest()
            self.secret_seeded = True
        else:
            self.seed = configured_seed
            self._random_key = hashlib.sha256(
                b"fl-security-mtd-simulation-v1:" + material
            ).digest()
            self.secret_seeded = False

        smpc = config.get("smpc", {})
        self.smpc_enabled = bool(smpc.get("enabled", False))
        self.min_clients_per_round = int(smpc.get("min_clients_per_round", 2))
        self.num_aggregation_parties = int(smpc.get(
            "num_aggregation_parties",
            max(self.committee_size, config.get("federated_learning", {}).get("num_clients", 1)),
        ))
        if self.committee_size < 2 or self.num_aggregation_parties < 2:
            raise ValueError("secure aggregation committees and party count must be >= 2")
        if self.min_clients_per_round < 2:
            raise ValueError("smpc.min_clients_per_round must be >= 2")

    def generate_round_config(self, round_num: int, num_clients: int, fraction: float) -> dict:
        if isinstance(round_num, bool) or not isinstance(round_num, int) or round_num <= 0:
            raise ValueError("round_num must be positive")
        moving = self.enabled and self.config.get("change_every_round", True)
        round_seed = self.seed + round_num if moving else self.seed
        round_material = f"round:{round_num}:seed:{round_seed}".encode("ascii")
        round_key = hmac.new(self._random_key, round_material, hashlib.sha256).digest()
        rng = HMACCounterRNG(round_key)

        strategy = self.selection_strategy if self.enabled else "fixed"
        selected = select_clients(
            num_clients, fraction, strategy=strategy, seed=round_seed, rng=rng
        )

        committee_size = min(self.committee_size, self.num_aggregation_parties)
        if self.smpc_enabled:
            if len(selected) < self.min_clients_per_round:
                raise ValueError(
                    f"SMPC requires at least {self.min_clients_per_round} selected clients per round"
                )
            if len(selected) * committee_size < 2 * self.num_aggregation_parties:
                raise ValueError(
                    "committee assignment cannot give every used party shares from at least "
                    "two clients; increase client fraction, reduce aggregator count, "
                    "or reduce committee_size"
                )

        # Balanced cyclic assignment: no committee repeats a party, party loads
        # differ by at most one, and every party gets >=2 shares when feasible.
        party_order = list(range(self.num_aggregation_parties))
        rng.shuffle(party_order)
        offset = rng.randbelow(self.num_aggregation_parties)
        cycle = party_order[offset:] + party_order[:offset]
        slots = [
            cycle[index % self.num_aggregation_parties]
            for index in range(len(selected) * committee_size)
        ]
        committees = {
            int(cid): sorted(
                int(pid)
                for pid in slots[index * committee_size:(index + 1) * committee_size]
            )
            for index, cid in enumerate(selected)
        }
        return {
            "round": round_num,
            "selected_clients": selected,
            "committees": committees,
            "seed": round_seed,
            "secret_seeded": self.secret_seeded,
            "num_aggregation_parties": self.num_aggregation_parties,
            "committee_size": committee_size,
        }
