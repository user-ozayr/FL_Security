import unittest
from fl_security.mtd.client_selection import select_clients
from fl_security.mtd.controller import MTDController

class TestMTD(unittest.TestCase):
    def test_client_selection_reproducibility(self):
        """Verify that client selection is reproducible given the same seed."""
        num_clients = 10
        fraction = 0.5
        selected_1 = select_clients(num_clients, fraction, strategy='random', seed=42)
        selected_2 = select_clients(num_clients, fraction, strategy='random', seed=42)
        self.assertEqual(selected_1, selected_2)

    def test_controller_generation(self):
        """Verify that MTD controller generates valid round configurations and committees."""
        config = {
            "experiment": {"seed": 42},
            "mtd": {"enabled": True, "selection_strategy": "random", "committee_size": 3}
        }
        controller = MTDController(config)
        round_cfg = controller.generate_round_config(1, num_clients=10, fraction=0.4)
        
        self.assertIn("selected_clients", round_cfg)
        self.assertIn("committees", round_cfg)
        self.assertEqual(len(round_cfg["selected_clients"]), 4)
        self.assertEqual(round_cfg["round"], 1)

if __name__ == '__main__':
    unittest.main()

