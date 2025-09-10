# Scaling up Masked Diffusion Models on Text

[![arXiv](https://img.shields.io/badge/arXiv-2410.18514-red.svg)](https://arxiv.org/abs/2410.18514)
[![deploy](https://img.shields.io/badge/Huggingface%20-SMDM%20-blue)](https://huggingface.co/nieshen/SMDM)

Masked diffusion models (MDMs) have shown promise in language modeling, yet their scalability and effectiveness in core 
language tasks, such as text generation and language understanding, remain underexplored. This paper establishes the 
first scaling law for MDMs, demonstrating a scaling rate comparable to autoregressive models (ARMs) and a relatively 
small compute gap. Motivated by their scalability, we train a family of MDMs with up to 1.1 billion (B) parameters to 
systematically evaluate their performance against ARMs of comparable or larger sizes. Fully leveraging the probabilistic 
formulation of MDMs, we propose a simple yet effective *unsupervised classifier-free* guidance that effectively 
exploits large-scale unpaired data, boosting performance for conditional inference. In language understanding, the 
1.1B MDM outperforms the 1.1B TinyLlama model trained on the same data across four of eight zero-shot benchmarks. 
Notably, it achieves competitive math reasoning ability with the 7B Llama-2 model on the GSM8K dataset. In text 
generation, MDMs provide a flexible trade-off compared to ARMs utilizing KV-cache: MDMs match the performance of 
ARMs while being 1.4 times faster or achieving higher quality than ARMs at a higher computational cost. Moreover, 
MDMs address challenging tasks for ARMs by effectively handling bidirectional reasoning and adapting to temporal 
shifts in data. Notably, a 1.1B MDM breaks the *reverse curse* encountered by much larger ARMs with significantly 
more data and computation, such as 13B Llama-2 and 175B GPT-3.




## Dependency

You need to run the script in `CONDA.md`.

We can build the Anaconda environment based on [TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md). First install the [TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md) Anaconda environment and then run
```sh
pip install lm-eval==0.4.4 numpy==1.25.0 bitsandbytes==0.43.1
pip install openai==0.28 fschat==0.2.34 anthropic
```
In addition, we provide the conda installation commands in the [CONDA.md](CONDA.md) file for reference and completeness.

## Pretrained models
We provided all pretrained models on [Huggingface](https://huggingface.co/nieshen/SMDM), including those 
for the scaling laws experiment, the conditional generation experiment, 
and the reverse curse experiment. 




## Supervised fine-tuning
### Math reasoning
Please download the augmented training [data](https://github.com/da03/implicit_chain_of_thought/blob/main/data/gsm8k/train.txt) and
put the `train.txt` file in `./data/gsm8k`.
```angular2html
./scripts/distillation_gsm8k_mdm.sh
```


## Evaluation



### Math reasoning
Please download the GSM8K test [data](https://github.com/hao-ai-lab/Consistency_LLM/blob/main/eval/gsm8k/test.jsonl)
and put the `test.jsonl` into `./data/gsm8k`
```angular2html
python evaluate_gsm8k.py --ckpt_path "models/mdm-1028M-3300e18-rsl-gsm8k.safetensors"
```
