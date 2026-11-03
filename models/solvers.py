from collections import OrderedDict
from pathlib import Path

import torch
from timm.models import create_model
from torch.utils.data import Subset

from FoundationModel import modeling_finetune
from models.adapt_src import train_adapt_src
from models.sda_lora import calibrate_target_gate, train_source_sda_lora


class Solvers:
    def __init__(self, config, args, target, dataset_dict, logger):
        self.config = config
        self.args = args
        self.target = target
        self.dataset_dict = dataset_dict
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints_folder = Path(args.save_dir) / "checkpoints"
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)

    def load_foundation_model(self):
        model = create_model(
            "labram_base_patch200_200", pretrained=False, num_classes=0,
            drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
            drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
            use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1,
            qkv_bias=False,
        )
        checkpoint = torch.load(
            Path(self.config["foundation_model"]["checkpoint"]),
            map_location="cpu", weights_only=False,
        )
        checkpoint_state = checkpoint.get("model", checkpoint.get("module", checkpoint))
        student_state = OrderedDict(
            (key[len("student."):], value)
            for key, value in checkpoint_state.items()
            if key.startswith("student.")
        )
        if student_state:
            checkpoint_state = student_state
        model_state = model.state_dict()
        loadable_state = OrderedDict(
            (key, value) for key, value in checkpoint_state.items()
            if "relative_position_index" not in key
            and key in model_state
            and value.shape == model_state[key].shape
        )
        model.load_state_dict(loadable_state, strict=False)
        return model

    def adapt_src(self):
        self.logger.log_info(f"{self.args.name} | start source adaptation")
        train_dataset, val_dataset = self.split_dataset(
            mode="source", split_type=self.config["data"]["src_dataset_split_type"]
        )
        metrics = train_adapt_src(
            target=self.target, config=self.config, logger=self.logger,
            device=self.device, checkpoints_folder=self.checkpoints_folder,
            foundation_model=self.load_foundation_model(),
            src_train_dataset=train_dataset, src_val_dataset=val_dataset,
        )
        return {
            "source_train_acc": metrics[0], "source_train_f1": metrics[1],
            "source_val_acc": metrics[2], "source_val_f1": metrics[3],
        }

    def adapt(self):
        method = self.config["adapt"]["method"]
        self.logger.log_info(f"{self.args.name} | start {method} adaptation")
        source_model = train_source_sda_lora(
            target=self.target, args=self.args, config=self.config,
            logger=self.logger, device=self.device,
            foundation_model=self.load_foundation_model(),
            src_dataset=self.dataset_dict["source_dataset"],
        )

        shot_metrics = {}
        for shot in self.config["data"]["target_shot_per_class"]:
            support_dataset, query_dataset = self.split_dataset(
                mode="target",
                split_type=self.config["data"]["tgt_dataset_split_type"],
                shot=shot,
            )
            acc, f1 = calibrate_target_gate(
                target=self.target, shot=shot, config=self.config,
                logger=self.logger, device=self.device, source_model=source_model,
                tgt_support_dataset=support_dataset, tgt_query_dataset=query_dataset,
            )
            shot_metrics[shot] = {"target_query_acc": acc, "target_query_f1": f1}
        return {method: shot_metrics}

    def split_dataset(self, mode, split_type, shot=None):
        if mode == "source":
            dataset = self.dataset_dict["source_dataset"]
            generator = torch.Generator().manual_seed(self.args.seed)
            if split_type == "source_random":
                indices = torch.randperm(len(dataset), generator=generator)
                val_size = int(len(dataset) * 0.2)
                val_indices = indices[:val_size].tolist()
                train_indices = indices[val_size:].tolist()
            else:
                subjects = torch.unique(dataset.subject).sort().values
                subject_order = torch.randperm(len(subjects), generator=generator)
                train_subjects = subjects[subject_order[:10]]
                val_subjects = subjects[subject_order[10:]]
                train_indices = torch.nonzero(
                    torch.isin(dataset.subject, train_subjects), as_tuple=True
                )[0].tolist()
                val_indices = torch.nonzero(
                    torch.isin(dataset.subject, val_subjects), as_tuple=True
                )[0].tolist()
            print(f"src train/val split done | train: {len(train_indices)}, val: {len(val_indices)}")
            return Subset(dataset, train_indices), Subset(dataset, val_indices)

        dataset = self.dataset_dict["target_dataset"]
        generator = torch.Generator().manual_seed(self.args.seed + self.target)
        support_indices, query_indices = [], []
        if split_type == "target_per_trial":
            for trial_id in torch.unique(dataset.trial).sort().values:
                trial_indices = torch.nonzero(dataset.trial == trial_id, as_tuple=True)[0]
                shuffled = trial_indices[torch.randperm(len(trial_indices), generator=generator)]
                support_indices.extend(shuffled[:shot].tolist())
                query_indices.extend(shuffled[shot:].tolist())
        elif split_type == "target_one_trial":
            for label_id in torch.unique(dataset.label).sort().values:
                class_mask = dataset.label == label_id
                class_trials = torch.unique(dataset.trial[class_mask]).sort().values
                support_trial = class_trials[
                    torch.randperm(len(class_trials), generator=generator)[0]
                ]
                support_pool = torch.nonzero(
                    class_mask & (dataset.trial == support_trial), as_tuple=True
                )[0]
                support_pool = support_pool[
                    torch.randperm(len(support_pool), generator=generator)
                ]
                support_indices.extend(support_pool[:shot].tolist())
                query_indices.extend(torch.nonzero(
                    class_mask & (dataset.trial != support_trial), as_tuple=True
                )[0].tolist())
        else:
            for label_id in torch.unique(dataset.label).sort().values:
                class_indices = torch.nonzero(dataset.label == label_id, as_tuple=True)[0]
                shuffled = class_indices[
                    torch.randperm(len(class_indices), generator=generator)
                ]
                support_indices.extend(shuffled[:shot].tolist())
                query_indices.extend(shuffled[shot:].tolist())

        print(
            f"tgt support/query split done | support: {len(support_indices)}, "
            f"query: {len(query_indices)}"
        )
        return Subset(dataset, support_indices), Subset(dataset, query_indices)
