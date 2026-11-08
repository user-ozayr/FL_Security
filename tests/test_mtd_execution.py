import unittest
from fl_security.mtd.controller import MTDController
from fl_security.mtd.client_selection import select_clients

class TestMTDExecution(unittest.TestCase):
    def test_mtd_client_selection_variance(self):
        """Verify that selected clients vary between rounds when MTD is enabled (random strategy)."""
        config = {
            "experiment": {"seed": 42},
            "mtd": {"enabled": True, "selection_strategy": "random", "committee_size": 3}
        }
        controller = MTDController(config)
        num_clients = 20
        fraction = 0.5
        
        round_1_cfg = controller.generate_round_config(1, num_clients, fraction)
        round_2_cfg = controller.generate_round_config(2, num_clients, fraction)
        
        clients_r1 = round_1_cfg["selected_clients"]
        clients_r2 = round_2_cfg["selected_clients"]
        
        print(f"\n[MTD Log] Round 1 selected clients: {clients_r1}")
        print(f"[MTD Log] Round 2 selected clients: {clients_r2}")
        print(f"[MTD Log] Round 1 committees: {round_1_cfg['committees']}")
        
        # Verify participation count matches fraction
        expected_count = int(num_clients * fraction)
        self.assertEqual(len(clients_r1), expected_count)
        self.assertEqual(len(clients_r2), expected_count)
        
        # Verify actual selection variance between rounds
        self.assertNotEqual(clients_r1, clients_r2)

    def test_fixed_baseline_stability(self):
        """Verify fixed baseline selection remains stable across rounds when MTD is disabled."""
        num_clients = 10
        fraction = 0.5
        # Fixed selection (e.g. sorted first K clients)
        selected_r1 = select_clients(num_clients, fraction, strategy='fixed', seed=42)
        selected_r2 = select_clients(num_clients, fraction, strategy='fixed', seed=42)
        
        print(f"\n[Baseline Log] Round 1 selected clients: {selected_r1}")
        print(f"[Baseline Log] Round 2 selected clients: {selected_r2}")
        
        self.assertEqual(selected_r1, selected_r2)

if __name__ == '__main__':
    unittest.main()
