"""
K2.5 MTP Phase A Training — Smoke test + real training script.

Phase A: Freeze decoder layer (K2.5 layer 60), train only MTP-exclusive params:
  embed_tokens, enorm, hnorm, eh_proj, shared_head, shared_head_norm

Usage (single GPU smoke test):
    python3 examples/mtp_k25_train.py \
        --data-path /data/mtp_eval/ \
        --output-dir output/mtp_k25_smoke_test \
        --epochs 3 \
        --total-seq-len 4096

Usage (2-GPU FSDP real training):
    torchrun --nproc_per_node=2 examples/mtp_k25_train.py \
        --data-path /data/mtp_train_k25/ \
        --output-dir output/mtp_k25_phase_a \
        --epochs 10 \
        --total-seq-len 8192
"""

import argparse
import sys
from pathlib import Path

# Ensure speculators is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoConfig
from torch.utils.data import DataLoader

from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig
from speculators.train.data import (
    Eagle3SampleFileDataset,
    create_collate_fn,
    split_files,
    standardize_data_mtp,
)
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import maybe_destroy_distributed, maybe_setup_distributed


VERIFIER_NAME_OR_PATH = "/data/.cache_claude/huggingface/hub/models--moonshotai--Kimi-K2.5/snapshots/54383e83fa343a1331754112fb9e3410c55efa2f"


def parse_args():
    parser = argparse.ArgumentParser(description="K2.5 MTP Phase A Training")
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Directory with .pt files from mtp_eval_generate.py",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/mtp_k25_phase_a",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--total-seq-len", type=int, default=8192,
        help="Maximum sequence length for multipack batching",
    )
    parser.add_argument(
        "--freeze-decoder", action=argparse.BooleanOptionalAction, default=True,
        help="Freeze decoder layer (Phase A). Use --no-freeze-decoder for Phase B.",
    )
    parser.add_argument(
        "--verifier-layer-idx", type=int, default=-1,
        help="Which K2.5 layer to copy for decoder init (-1 = last layer = layer 60)",
    )
    parser.add_argument(
        "--loss-type", type=str, default="ce", choices=["ce", "kl"],
        help="Loss function: cross-entropy or KL-divergence",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--logger", type=str, default="tensorboard")
    parser.add_argument("--no-resume-from-checkpoint", action="store_true")
    parser.add_argument("--scheduler-type", type=str, default="cosine")
    parser.add_argument("--scheduler-warmup-steps", type=int, default=100)
    return parser.parse_args()


def setup_dataloader(
    file_list: list[str],
    world_size: int,
    local_rank: int,
    total_seq_len: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = Eagle3SampleFileDataset(
        file_list=file_list,
        max_len=total_seq_len,
        standardize_fn=standardize_data_mtp,
        hidden_states_dtype=torch.bfloat16,
    )
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=world_size,
        rank=local_rank,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        collate_fn=create_collate_fn(total_seq_len),
        persistent_workers=(num_workers > 0),
    )


def main(args):
    # Distributed setup
    local_rank, world_size, rank, is_distributed = maybe_setup_distributed()

    # Logging
    setup_root_logger()
    import logging
    log = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    save_path = str(output_dir / "checkpoints")
    log_dir = str(output_dir / "logs")

    # Load verifier config
    log.info(f"Loading verifier config from {VERIFIER_NAME_OR_PATH}...")
    verifier_config = AutoConfig.from_pretrained(VERIFIER_NAME_OR_PATH, trust_remote_code=True)
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config

    log.info(
        f"Verifier: {verifier_config.model_type}, "
        f"hidden={verifier_config.hidden_size}, "
        f"layers={verifier_config.num_hidden_layers}, "
        f"vocab={verifier_config.vocab_size}"
    )

    # Build MTP model
    log.info(
        f"Building MTPDraftModel "
        f"(freeze_decoder={args.freeze_decoder}, "
        f"layer_idx={args.verifier_layer_idx})..."
    )
    draft_model = MTPDraftModel.from_training_args(
        verifier_config=verifier_config,
        verifier_name_or_path=VERIFIER_NAME_OR_PATH,
        freeze_decoder=args.freeze_decoder,
        verifier_layer_idx=args.verifier_layer_idx,
    )

    # Cast model to BF16 to match data dtype
    draft_model = draft_model.bfloat16()

    # Print parameter counts
    total_params = sum(p.numel() for p in draft_model.parameters())
    trainable_params = sum(
        p.numel() for p in draft_model.parameters() if p.requires_grad
    )
    log.info(
        f"Parameters: total={total_params/1e9:.2f}B, "
        f"trainable={trainable_params/1e9:.2f}B "
        f"({trainable_params/total_params*100:.1f}%)"
    )

    # Data
    train_files, val_files = split_files(args.data_path, ratio=0.9)
    log.info(f"Dataset: {len(train_files)} train, {len(val_files)} val files")

    train_loader = setup_dataloader(
        train_files, world_size, local_rank,
        args.total_seq_len, args.num_workers, args.prefetch_factor,
    )
    val_loader = setup_dataloader(
        val_files, world_size, local_rank,
        args.total_seq_len, args.num_workers, args.prefetch_factor,
    )

    # Trainer kwargs
    train_call_kwargs, val_call_kwargs = MTPDraftModel.get_trainer_kwargs(
        loss_type=args.loss_type,
    )

    trainer_config = TrainerConfig(
        num_epochs=args.epochs,
        save_path=save_path,
        lr=args.lr,
        resume_from_checkpoint=not args.no_resume_from_checkpoint,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs=train_call_kwargs,
        val_call_kwargs=val_call_kwargs,
        scheduler_type=args.scheduler_type,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
    )

    # Logger
    if local_rank == 0:
        setup_metric_logger(args.logger, run_name="mtp_k25_smoke", output_dir=log_dir)

    trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)
    trainer.run_training()

    maybe_destroy_distributed()


if __name__ == "__main__":
    args = parse_args()
    main(args)
