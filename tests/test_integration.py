import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from fl_security.federated.model import CIFAR10CNN
from fl_security.federated.training import train_local, evaluate_model
from fl_security.mtd.controller import MTDController

class TestIntegration(unittest.TestCase):
    def test_complete_round_workflow(self):
        """Test a complete local training, MTD round generation, and evaluation cycle."""
        # Create small dummy tensor dataset for testing round integration
        images = torch.randn(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=8)

        model = CIFAR10CNN()
        config = {
            "experiment": {"seed": 42},
            "mtd": {"enabled": True, "selection_strategy": "random", "committee_size": 2}
        }
        controller = MTDController(config)
        round_cfg = controller.generate_round_config(1, num_clients=3, fraction=1.0)
        
        self.assertEqual(len(round_cfg["selected_clients"]), 3)

        # Execute local training
        state_dict = train_local(model, dataloader, epochs=1, lr=0.01)
        self.assertIsNotNone(state_dict)

        # Execute evaluation
        loss, acc = evaluate_model(model, dataloader)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)

if __name__ == '__main__':
    unittest.main()
