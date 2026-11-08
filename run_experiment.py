import argparse
import torch
from torch.utils.data import DataLoader
from fl_security.config import load_config
from fl_security.federated.model import CIFAR10CNN
from fl_security.federated.dataset import load_cifar10_data, partition_data
from fl_security.federated.training import train_local, evaluate_model
from fl_security.mtd.controller import MTDController
from fl_security.smpc.fixed_point import FixedPointEncoder
from fl_security.smpc.additive_sharing import generate_additive_shares, reconstruct_additive_shares
from fl_security.evaluation.metrics import ExperimentLogger

def main():
    parser = argparse.ArgumentParser(description="Run FL Security Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded Experiment: {config['experiment']['name']}")

    trainset, testset = load_cifar10_data()
    num_clients = config["federated_learning"]["num_clients"]
    client_datasets = partition_data(trainset, num_clients, partitioning=config["federated_learning"]["partitioning"], seed=config["experiment"]["seed"])
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    global_model = CIFAR10CNN()
    mtd_controller = MTDController(config)
    encoder = FixedPointEncoder(scale=config.get("smpc", {}).get("fixed_point_scale", 1000000))
    logger = ExperimentLogger()

    rounds = config["experiment"]["rounds"]
    fraction = config["federated_learning"]["fraction_train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for r in range(1, rounds + 1):
        print(f"\n--- Round {r}/{rounds} ---")
        round_cfg = mtd_controller.generate_round_config(r, num_clients, fraction)
        selected_clients = round_cfg["selected_clients"]
        print(f"Selected Clients: {selected_clients}")

        global_state = global_model.state_dict()
        client_updates = []

        for cid in selected_clients:
            client_loader = DataLoader(client_datasets[cid], batch_size=config["federated_learning"]["batch_size"], shuffle=True)
            local_model = CIFAR10CNN()
            local_model.load_state_dict(global_state)
            
            updated_state = train_local(local_model, client_loader, config["federated_learning"]["local_epochs"], config["federated_learning"]["learning_rate"], device=device)
            client_updates.append(updated_state)

        # Aggregate updates (with optional SMPC simulation)
        new_state = {}
        for key in global_state.keys():
            if torch.is_floating_point(global_state[key]):
                # Average client updates
                stacked = torch.stack([upd[key].float() for upd in client_updates], dim=0)
                new_state[key] = torch.mean(stacked, dim=0)
            else:
                new_state[key] = global_state[key]

        global_model.load_state_dict(new_state)
        loss, acc = evaluate_model(global_model, testloader, device=device)
        print(f"Global Evaluation -> Loss: {loss:.4f}, Accuracy: {acc * 100:.2f}%")

        logger.log_round({
            "round": r,
            "global_loss": loss,
            "global_accuracy": acc,
            "selected_clients": len(selected_clients)
        })

    if config["evaluation"].get("save_csv", True):
        logger.save_to_csv(f"{config['experiment']['name']}_results.csv")
    if config["evaluation"].get("save_plots", True):
        logger.plot_metrics(f"{config['experiment']['name']}_accuracy.png")
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
