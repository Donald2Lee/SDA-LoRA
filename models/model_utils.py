import torch
from sklearn.metrics import accuracy_score, f1_score


def print_trainable_parameters(model):
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return f"trainable={trainable:,} | total={total:,} | ratio={100 * trainable / total:.4f}%"


@torch.no_grad()
def evaluate(loader, model, criterion=None, input_chans=None):
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_num = 0.0, 0
    predictions, labels = [], []
    for batch in loader:
        data = batch["data"].unsqueeze(2).to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(data, input_chans=input_chans)
        if criterion is not None:
            total_loss += criterion(logits, label).item() * label.size(0)
            total_num += label.size(0)
        predictions.append(logits.argmax(dim=1).cpu())
        labels.append(label.cpu())

    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()
    return {
        "acc": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
        "avg_loss": total_loss / total_num if total_num else 0.0,
    }
