import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.model_utils import evaluate, print_trainable_parameters


def train_source_sda_lora(target, args, config, logger, device,
                          foundation_model, src_dataset):
    sda_cfg = config["adapt"]["sda_lora"]
    adapted_checkpoint = (
        Path(args.output) / sda_cfg["adaptsrc_checkpoint"] / "checkpoints"
        / f"checkpoint-target{target}-adaptsrc.pt"
    )
    source_checkpoint = (
        Path(args.save_dir) / "checkpoints" / f"checkpoint-target{target}-sda-source.pt"
    )

    model = copy.deepcopy(foundation_model)
    model.reset_classifier(sda_cfg["num_classes"])
    checkpoint = torch.load(adapted_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    inject_sda_lora_into_attention(
        model=model,
        transformer_lora_layers=sda_cfg["transformer_lora_layers"],
        rank=sda_cfg["rank"],
        alpha=sda_cfg["alpha"],
        num_branches=sda_cfg["num_branches"],
        dropout=sda_cfg["lora_dropout"],
        lora_q=sda_cfg["lora_q"],
        lora_k=sda_cfg["lora_k"],
        lora_v=sda_cfg["lora_v"],
        lora_o=sda_cfg["lora_o"],
    )
    mark_only_sda_lora_trainable(model)
    model.to(device)

    if source_checkpoint.exists():
        saved = torch.load(source_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        logger.log_info(f"using existing SDA-LoRA source checkpoint: {source_checkpoint}")
        return model

    logger.log_info(f"SDA-LoRA source training: {print_trainable_parameters(model)}")
    loader = DataLoader(
        src_dataset, batch_size=sda_cfg["source_batch_size"], shuffle=True,
        pin_memory=True, drop_last=False,
    )
    input_chans = src_dataset.input_chans.tolist()
    optimizer = torch.optim.AdamW(
        build_sda_parameter_groups(
            model, sda_cfg["base_lr"], sda_cfg["lr_layer_decay"], sda_cfg["weight_decay"]
        ),
        betas=(0.9, 0.999),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=sda_cfg["label_smoothing"])
    best_acc = -1.0
    best_model = None

    for epoch in tqdm(range(1, sda_cfg["source_epochs"] + 1)):
        model.train()
        cls_loss_sum, orth_loss_sum, batch_count = 0.0, 0.0, 0
        for batch in loader:
            data = batch["data"].unsqueeze(2).to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            cls_loss = criterion(model(data, input_chans=input_chans), label)
            orth_loss = orthogonal_branch_loss(model)
            loss = cls_loss + sda_cfg["lambda_orth"] * orth_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            cls_loss_sum += cls_loss.item()
            orth_loss_sum += orth_loss.item()
            batch_count += 1

        metrics = evaluate(loader, model, criterion, input_chans)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_model = copy.deepcopy(model)
        logger.log_info(
            f"target={target} | SDA source epoch {epoch}/{sda_cfg['source_epochs']}"
            f" | ClsLoss={cls_loss_sum / batch_count:.4f}"
            f" | OrthLoss={orth_loss_sum / batch_count:.4f}"
            f" | Acc/F1={metrics['acc']:.4f}/{metrics['f1']:.4f}"
            f" | BestAcc={best_acc:.4f}"
        )

    torch.save({"model": best_model.state_dict(), "best_source_acc": best_acc}, source_checkpoint)
    logger.log_info(f"saved SDA-LoRA source checkpoint: {source_checkpoint}")
    return best_model


def calibrate_target_gate(target, shot, config, logger, device,
                          source_model, tgt_support_dataset, tgt_query_dataset):
    sda_cfg = config["adapt"]["sda_lora"]
    model = copy.deepcopy(source_model)
    model.requires_grad_(False)
    mark_only_sda_lora_trainable(model, only_router=True)
    model.to(device)
    logger.log_info(f"SDA-LoRA target calibration: {print_trainable_parameters(model)}")

    support_loader = DataLoader(
        tgt_support_dataset, batch_size=len(tgt_support_dataset), shuffle=False,
        pin_memory=True, drop_last=False,
    )
    query_loader = DataLoader(
        tgt_query_dataset, batch_size=sda_cfg["query_batch_size"], shuffle=False,
        pin_memory=True, drop_last=False,
    )
    input_chans = tgt_support_dataset.dataset.input_chans.tolist()
    optimizer = torch.optim.AdamW(
        build_sda_parameter_groups(
            model, sda_cfg["base_lr"], sda_cfg["lr_layer_decay"], sda_cfg["weight_decay"]
        ),
        betas=(0.9, 0.999),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=sda_cfg["label_smoothing"])
    target_steps = sda_cfg["target_steps_per_shot"] * shot

    model.train()
    for step in tqdm(range(1, target_steps + 1)):
        for batch in support_loader:
            data = batch["data"].unsqueeze(2).to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data, input_chans=input_chans), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if step == 1 or step % 50 == 0 or step == target_steps:
            logger.log_info(
                f"target={target} | shot={shot} | gate step {step}/{target_steps}"
                f" | support loss={loss.item():.4f}"
            )

    query_metrics = evaluate(query_loader, model, criterion, input_chans)
    logger.log_info(
        f"target={target} | shot={shot} | query Acc/F1: "
        f"{query_metrics['acc']:.4f}/{query_metrics['f1']:.4f}"
    )
    return query_metrics["acc"], query_metrics["f1"]


def orthogonal_branch_loss(model):
    losses = []
    for module in model.modules():
        if isinstance(module, SDALoRAAdapter):
            branches = F.normalize(module.lora_B.flatten(1), p=2, dim=1, eps=1e-12)
            correlation = torch.abs(branches @ branches.transpose(0, 1))
            branch_ids = torch.triu_indices(
                module.num_branches, module.num_branches, offset=1,
                device=correlation.device,
            )
            losses.append(correlation[branch_ids[0], branch_ids[1]].mean())
    return torch.stack(losses).mean()


def build_sda_parameter_groups(model, base_lr, layer_decay, weight_decay):
    depth = len(model.blocks)
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        block_idx = int(name.split(".")[1])
        group_lr = base_lr * layer_decay ** (depth - block_idx)
        group_decay = 0.0 if parameter.ndim == 1 or name.endswith(".bias") else weight_decay
        group_key = (block_idx, float(group_lr), float(group_decay))
        if group_key not in grouped:
            grouped[group_key] = {
                "params": [], "lr": group_lr, "weight_decay": group_decay,
                "group_type": f"sda_lora_block_{block_idx}",
            }
        grouped[group_key]["params"].append(parameter)
    return list(grouped.values())


def mark_only_sda_lora_trainable(model, only_router=False):
    for name, parameter in model.named_parameters():
        trainable = "lora_router." in name if only_router else "lora_" in name
        parameter.requires_grad_(trainable)


def inject_sda_lora_into_attention(model, transformer_lora_layers, rank, alpha,
                                   num_branches, dropout, lora_q, lora_k, lora_v, lora_o):
    selected_blocks = (
        model.blocks if transformer_lora_layers == 0
        else model.blocks[-transformer_lora_layers:]
    )
    for block in selected_blocks:
        if lora_q or lora_k or lora_v:
            block.attn.qkv = SDALoRAQKVLinear(
                block.attn.qkv, rank, alpha, num_branches, dropout,
                lora_q, lora_k, lora_v,
            )
        if lora_o:
            block.attn.proj = SDALoRALinear(
                block.attn.proj, rank, alpha, num_branches, dropout
            )
    return model


class SDALoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank, num_branches, dropout):
        super().__init__()
        self.num_branches = num_branches
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_branches, out_features, rank))
        self.lora_router = nn.Linear(in_features, num_branches, bias=True)
        self.lora_dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.lora_router.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_router.bias)

    def forward(self, x):
        low_rank_features = F.linear(self.lora_dropout(x), self.lora_A)
        routing_weights = F.softmax(self.lora_router(x), dim=-1, dtype=torch.float32).to(x.dtype)
        branch_outputs = torch.einsum("...r,nor->...no", low_rank_features, self.lora_B)
        return torch.einsum("...n,...no->...o", routing_weights, branch_outputs)


class SDALoRAQKVLinear(nn.Module):
    def __init__(self, base_layer, rank, alpha, num_branches, dropout,
                 lora_q, lora_k, lora_v):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = alpha / rank
        features = base_layer.out_features // 3
        self.adapters = nn.ModuleDict()
        for name, enabled in (("q", lora_q), ("k", lora_k), ("v", lora_v)):
            if enabled:
                self.adapters[name] = SDALoRAAdapter(
                    base_layer.in_features, features, rank, num_branches, dropout
                )
        self.base_layer.requires_grad_(False)

    def forward(self, x):
        outputs = list(self.base_layer(x).chunk(3, dim=-1))
        indices = {"q": 0, "k": 1, "v": 2}
        for name, adapter in self.adapters.items():
            outputs[indices[name]] = outputs[indices[name]] + self.scaling * adapter(x)
        return torch.cat(outputs, dim=-1)


class SDALoRALinear(nn.Module):
    def __init__(self, base_layer, rank, alpha, num_branches, dropout):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = alpha / rank
        self.adapter = SDALoRAAdapter(
            base_layer.in_features, base_layer.out_features, rank, num_branches, dropout
        )
        self.base_layer.requires_grad_(False)

    def forward(self, x):
        return self.base_layer(x) + self.scaling * self.adapter(x)
