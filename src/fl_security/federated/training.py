import torch
import torch.nn as nn
import torch.optim as optim

def train_local(model, dataloader, epochs, lr, device="cpu", label_transform=None):
    """Train locally, optionally applying a controlled data-poisoning transform."""
    if epochs <= 0 or lr <= 0:
        raise ValueError("epochs and learning rate must be positive")
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if label_transform is not None:
                labels = label_transform(labels)
                if not torch.is_tensor(labels):
                    labels = torch.as_tensor(labels, device=device)
                labels = labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(images), labels).backward()
            optimizer.step()
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

def evaluate_model(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    loss_total = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise ValueError("evaluation dataloader is empty")
    return loss_total / total, correct / total
