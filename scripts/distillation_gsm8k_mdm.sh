#!/bin/zsh


### Math reasoning
# Please download the augmented training [data](https://github.com/da03/implicit_chain_of_thought/blob/main/data/gsm8k/train.txt) and
# put the `train.txt` file in `./data/gsm8k`.

#pip install "git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/rotary"

export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1



lightning run model sft/distillation_finetune_mdm_gsm8k.py --devices=2 --model 170 \
    --pretrain_path models/mdm-170M-100e18.safetensors \
    --teacher_path models/mdm-170M-100e18.safetensors