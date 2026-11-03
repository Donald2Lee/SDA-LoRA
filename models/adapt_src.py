import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.model_utils import evaluate
from utils.lr_sch import CosineAnnealingLRWithWarmup


def train_adapt_src(target, config, logger, device, checkpoints_folder,
                    foundation_model, src_train_dataset, src_val_dataset):
    train_cfg = config["adapt_src"]
    checkpoint_path = checkpoints_folder / f"checkpoint-target{target}-adaptsrc.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        logger.log_info(f"using existing source checkpoint: {checkpoint_path}")
        return (
            checkpoint["best_tr_acc"], checkpoint["best_tr_f1"],
            checkpoint["best_val_acc"], checkpoint["best_val_f1"],
        )

    model = copy.deepcopy(foundation_model)
    model.reset_classifier(train_cfg["num_classes"])
    model.requires_grad_(False)
    trainable_layers = train_cfg["trainable_layers"]
    selected_blocks = model.blocks if trainable_layers == 0 else model.blocks[-trainable_layers:]
    for block in selected_blocks:
        block.requires_grad_(True)
    model.fc_norm.requires_grad_(True)
    model.head.requires_grad_(True)
    model.to(device)

    input_chans = src_train_dataset.dataset.input_chans.tolist()
    train_loader = DataLoader(
        src_train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        src_val_dataset, batch_size=train_cfg["batch_size"], shuffle=False,
        pin_memory=True, drop_last=False,
    )
    parameter_groups = build_parameter_groups(
        model, 0.0, train_cfg["lr_layer_decay"], train_cfg["weight_decay"]
    )
    optimizer = torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    total_steps = train_cfg["epochs"] * len(train_loader)
    warmup_steps = train_cfg["warmup_epochs"] * len(train_loader)
    warmup_lrs = [train_cfg["base_lr"] * group["lr_scale"] for group in optimizer.param_groups]
    min_lrs = [train_cfg["min_lr"] * group["lr_scale"] for group in optimizer.param_groups]
    scheduler = CosineAnnealingLRWithWarmup(
        optimizer, T_max=total_steps, warmup=warmup_steps,
        warmup_lr=warmup_lrs, min_lr=min_lrs, last_epoch=0,
    )

    best = {"tr_acc": 0.0, "tr_f1": 0.0, "val_acc": 0.0, "val_f1": 0.0}
    start = time.time()
    for epoch in tqdm(range(1, train_cfg["epochs"] + 1)):
        model.train()
        for batch in train_loader:
            data = batch["data"].unsqueeze(2).to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data, input_chans=input_chans), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_metrics = evaluate(train_loader, model, criterion, input_chans)
        val_metrics = evaluate(val_loader, model, criterion, input_chans)
        if val_metrics["f1"] > best["val_f1"]:
            best = {
                "tr_acc": train_metrics["acc"], "tr_f1": train_metrics["f1"],
                "val_acc": val_metrics["acc"], "val_f1": val_metrics["f1"],
            }
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "best_tr_acc": best["tr_acc"], "best_tr_f1": best["tr_f1"],
                "best_val_acc": best["val_acc"], "best_val_f1": best["val_f1"],
            }, checkpoint_path)

        logger.log_info(
            f"target={target} | Epoch {epoch}/{train_cfg['epochs']}"
            f" | LossTrVl: {train_metrics['avg_loss']:.4f}/{val_metrics['avg_loss']:.4f}"
            f" | AccTrVlbest: {train_metrics['acc']:.4f}/{val_metrics['acc']:.4f}/{best['val_acc']:.4f}"
            f" | F1TrVlbest: {train_metrics['f1']:.4f}/{val_metrics['f1']:.4f}/{best['val_f1']:.4f}"
        )
    logger.log_info(f"Adapt src done, time: {time.time() - start:.2f}")
    return best["tr_acc"], best["tr_f1"], best["val_acc"], best["val_f1"]


def build_parameter_groups(model, base_lr, layer_decay, weight_decay):
    depth = len(model.blocks)
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("head.", "fc_norm.", "norm.")):
            lr_scale = 1.0
        elif name.startswith("blocks."):
            lr_scale = layer_decay ** (depth - int(name.split(".")[1]))
        else:
            lr_scale = layer_decay ** (depth + 1)
        no_decay = parameter.ndim == 1 or name.endswith(".bias") or name in {
            "cls_token", "pos_embed", "time_embed"
        }
        group_key = (float(lr_scale), 0.0 if no_decay else float(weight_decay))
        if group_key not in grouped:
            grouped[group_key] = {
                "params": [], "lr": base_lr * lr_scale, "lr_scale": lr_scale,
                "weight_decay": group_key[1],
            }
        grouped[group_key]["params"].append(parameter)
    return list(grouped.values())
