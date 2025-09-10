#!/bin/zsh


### Math reasoning
# Please download the augmented training [data](https://github.com/da03/implicit_chain_of_thought/blob/main/data/gsm8k/train.txt) and
# put the `train.txt` file in `./data/gsm8k`.

#pip install "git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/rotary"

export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1


python sft/distillation_finetune_mdm_gsm8k.py --model 1028 \
    --pretrain_path models/mdm-1028M-3300e18-rsl-0.01-bs-1024.safetensors