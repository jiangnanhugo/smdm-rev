import re
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union
import math
import lightning as L
import torch
import torch.nn.functional as F
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.diffmodel import TransEncoder, Block, Config

from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import get_default_supported_precision, num_parameters, step_csv_logger
from pytorch_lightning.loggers import WandbLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from gsm8k_data import preprocess_gsm8k
from transformers import AutoTokenizer

import argparse
from safetensors.torch import load_file


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, help='model parameters')
    parse.add_argument('--bs', type=int, default=2, help='batch size')
    parse.add_argument('--epoch', type=int, default=40, help='training epoch')
    parse.add_argument('--pretrain_path', type=str, help='pretrain ckpt path')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of devices')
    parse.add_argument('--teacher_path', type=str, default='', help='teacher ckpt path or HF spec')
    args = parse.parse_args()
    return args


args = parse_args()
model_name = f'Diff_LLaMA_{args.model}M'  # config
out_dir = Path('workdir')

# Hyperparameters
num_of_devices = 2
global_batch_size = int(args.bs / args.nodes_num)
learning_rate = 2e-4
micro_batch_size = 1
max_step = int(769240 * args.epoch / args.bs)
warmup_steps = int(max_step * 0.01)
log_step_interval = 10
save_step_interval = 5000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = learning_rate / 10

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
print(f'gradient_accumulation_steps: {gradient_accumulation_steps}')
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps

max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", model_name, flush_logs_every_n_steps=log_iter_interval)


def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, p_mask


# ========================= Distillation utilities (DUO-style) =========================

# Distillation hyperparameters (aligned with algo_simp.Distillation defaults where sensible)
distill_cfg = {
    "gamma_min": -1.0,            # logit schedule min
    "gamma_max": 1.0,             # logit schedule max
    "update_teacher_every": 100,  # how often to refresh teacher weights
    "teacher_ema": 0.999,         # EMA decay for teacher
    "linear_growth_dt": True,     # enable linear growth of dt
    "linear_growth_min": 0.01,    # min dt
    "linear_growth_max": 0.5,     # max dt over training
    "freeze_teacher": False,      # if True, disable EMA updating of teacher
}


@torch.no_grad()
def _maybe_update_teacher_weights(student, teacher, step_count: int):
    if distill_cfg["freeze_teacher"]:
        return
    if step_count % distill_cfg["update_teacher_every"] != 0:
        return
    ema = distill_cfg["teacher_ema"]
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        if not tp.requires_grad:
            continue
        tp.data.mul_(ema).add_(sp.data, alpha=(1.0 - ema))


def _compute_dt(curr_iter: int, max_iters: int) -> float:
    if distill_cfg["linear_growth_dt"]:
        scale = curr_iter / max_iters if max_iters > 0 else 0.0
        return distill_cfg["linear_growth_min"] + scale * (
            distill_cfg["linear_growth_max"] - distill_cfg["linear_growth_min"]
        )
    # fallback exponential growth similar to algo_simp
    # not used by default in this script
    return distill_cfg["linear_growth_min"]


def _gamma_to_alphat(gamma: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(-gamma).sqrt()


def _sample_trajectory_discrete(x0_ids: torch.Tensor, gamma_t: torch.Tensor, gamma_s: torch.Tensor, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # x0 one-hot
    x0_one_hot = F.one_hot(x0_ids, vocab_size).to(torch.float32)
    # expand time dims
    b, l = x0_ids.shape
    gamma_t = gamma_t.view(b, 1, 1)
    gamma_s = gamma_s.view(b, 1, 1)
    alpha_t = torch.sigmoid(-gamma_t).sqrt()
    sigma_t = torch.sigmoid(gamma_t).sqrt()
    alpha_s = torch.sigmoid(-gamma_s).sqrt()
    sigma_s = torch.sigmoid(gamma_s).sqrt()
    # shared noise epsilon as in algo_simp
    epsilon = torch.randn_like(x0_one_hot)
    xt = alpha_t * x0_one_hot + sigma_t * epsilon
    xs = alpha_s * x0_one_hot + sigma_s * epsilon
    # discretize
    xt_discrete = xt.argmax(dim=-1)
    xs_discrete = xs.argmax(dim=-1)
    return xt_discrete, xs_discrete


def _kl_forward_with_mask(student_logits: torch.Tensor, teacher_logits: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    # logits: [B, L, V]; mask: [B, L] with 1.0 for tokens to include
    log_p_t = torch.log_softmax(teacher_logits, dim=-1)
    log_p_s = torch.log_softmax(student_logits, dim=-1)
    p_t = log_p_t.exp()
    kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)  # [B, L]
    loss = (kl * token_mask).sum() / (token_mask.sum().clamp_min(1.0))
    return loss


def extract_number(filename):
    match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
    return int(match.group(1)) if match else 0


def setup(
        devices: int = 2,
        precision: Optional[str] = None,
        tpu: bool = False,
        resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f'mdm-gsm8k-{args.model}M'
    out_dir = Path('workdir/finetune') / hp_name
    pretrain_path = args.pretrain_path
    wandb_logger = WandbLogger(name=hp_name, save_dir=out_dir, project='scaling')

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    # fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, pretrain_path, resume)


def main(fabric, pretrain_path, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                              padding_side="right", use_fast=True)

    train_set = preprocess_gsm8k(tokenizer, max_length=256)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)
    train_dataloader = DataLoader(train_set, batch_size=micro_batch_size, shuffle=True, drop_last=True,
                                  num_workers=1, pin_memory=True, persistent_workers=True)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

        ckpt_dic = load_file(pretrain_path)
        model.load_state_dict(ckpt_dic)
        fabric.print(f"Loading model from {pretrain_path}")

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)


    # # Teacher initialization: load from HF or local safetensors if provided, otherwise clone student
    
    with fabric.init_module(empty_init=False):
        teacher = TransEncoder(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

        ckpt_dic = load_file(args.teacher_path)
        teacher.load_state_dict(ckpt_dic)
        fabric.print(f"Loading teacher model from {args.teacher_path}")
        distill_cfg["freeze_teacher"] = False
    for p in teacher.parameters():
        p.requires_grad_(False)

    fabric.print(f"Time to instantiate TEACHER: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(teacher):,}")
    
    teacher = fabric.setup(teacher)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}
    # attach teacher & tokenizer for convenience
    state["teacher"] = teacher
    state["tokenizer"] = tokenizer

    if resume is True:
        try:
            resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
        except:
            resume = False
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)


    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, monitor, resume):
    """
    training!
    """
    model = state["model"]
    optimizer = state["optimizer"]
    teacher = state["teacher"]
    tokenizer = state["tokenizer"]

    with torch.device("meta"):
        meta_model = TransEncoder(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        # measured_flops = measure_flops(meta_model, x)
        # fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    initial_iter = state["iter_num"]
    curr_iter = 0

    def get_train_dataloader(dataset_loader):
        while True:
            for data in dataset_loader:
                yield data

    train_dataloader_ = get_train_dataloader(train_dataloader)

    # loss_func = CrossEntropyLoss(reduction='none')
    for train_data in train_dataloader_:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()
        input_ids = train_data['data']  # [prompt + answer + padding], length=2048
        prompt_length = train_data['input_length']  # prompt length
        max_length = 256
        input_ids = input_ids[:, :max_length]

        # Distillation: compute dt schedule
        dt = _compute_dt(state["iter_num"], max_iters)
        # sample t in (0,1]
        t = torch.rand((input_ids.size(0),), device=input_ids.device)
        t = torch.clamp(t + dt, 0.0, 1.0)
        gamma_min = distill_cfg["gamma_min"]
        gamma_max = distill_cfg["gamma_max"]
        gamma_t = gamma_min + t * (gamma_max - gamma_min)
        gamma_s = gamma_min + (t - dt) * (gamma_max - gamma_min)

        # build discrete noisy inputs following algo_simp.Distillation
        vocab_size = tokenizer.vocab_size
        xt_discrete, xs_discrete = _sample_trajectory_discrete(input_ids, gamma_t, gamma_s, vocab_size)

        # compute logits
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            student_logits = model(xt_discrete)
            with torch.no_grad():
                _maybe_update_teacher_weights(model, teacher, state["iter_num"])
                teacher_logits = teacher(xs_discrete)

            # mask: only supervise answer tokens (positions >= prompt_length)
            temp_tensor = torch.arange(input_ids.size(1), device=input_ids.device).expand(input_ids.size(0),
                                                                                          input_ids.size(1))
            answer_mask = (temp_tensor >= prompt_length.unsqueeze(1)).to(student_logits.dtype)
            # DISTILLATION LOSS
            loss = _kl_forward_with_mask(student_logits, teacher_logits, answer_mask)

            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
            f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. "
            # print days as well
            f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
        )

        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss.item()
        )

        if not is_accumulating and (state["step_count"] % save_step_interval == 0 or state["step_count"] == max_step):
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    # torch.set_float32_matmul_precision("high")
    setup()
