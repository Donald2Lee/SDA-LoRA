# SDA-LoRA

Official PyTorch implementation of **SDA-LoRA: Subject-Difference-Aware Low-Rank Adaptation for Cross-Subject Few-Shot Neural Decoding**.

## Method

- Represents cross-subject heterogeneity with multiple reusable low-rank adaptation directions.
- Uses a shared low-dimensional space to capture common adaptation information across branches.
- Dynamically recombines distinct adaptation directions through an input-dependent gate.
- Calibrates only the gate for an unseen target subject while keeping the backbone and source-learned low-rank parameters frozen.

## Structure

```text
SDA-LoRA/
|-- main.py
|-- process_data.py
|-- configs/
|   `-- main.yaml
|-- models/
|   |-- adapt_src.py
|   |-- sda_lora.py
|   `-- solvers.py
|-- utils/
`-- FoundationModel/
    |-- modeling_finetune.py
    `-- labram-base.pth
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

First adapt LaBraM to the source subjects:

```bash
python main.py --name adaptsrc_12 --config_file configs/main.yaml --adapt_src --cudnn_deterministic --tensorboard
```

Then train SDA-LoRA and calibrate the gate for target subjects:

```bash
python main.py --name sda_lora_12_qv_onlyrouter --config_file configs/main.yaml --cudnn_deterministic --tensorboard
```

Training settings are defined in `configs/main.yaml`. The released code retains the original experiment-code data splitting protocol. The orthogonal-loss coefficient `lambda_orth` is configurable and defaults to `0.1`.

## LaBraM

The backbone implementation and `labram-base.pth` are from the LaBraM project:

- Paper: https://openreview.net/forum?id=QzTpTRVtrP
- Code: https://github.com/935963004/LaBraM

The original LaBraM MIT License is included in `FoundationModel/LaBraM_LICENSE`.
