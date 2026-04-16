"""
EAGLE3 Producer-Consumer Phase 5 — Consumer (host-10-83-115-14)

Key improvements over Phase 4:
  - DP=8: 8 consumer processes (one per GPU) via torchrun --nproc_per_node=8
  - DDP via DistributedDataParallel + NVLink AllReduce (~5ms per step)
  - Variable-size recv: receives actual seq_len tensors (not padded to 4096)
  - world_size=9: global rank = local_rank + 1 (producer is rank 0)
  - Only local_rank=0 (global rank 1) saves checkpoints and sends loss back

Process group setup:
  P2P group  (world_size=9): all communication with producer
  DDP group  (ranks 1..8):   gradient AllReduce within consumer node (NVLink)
"""

import os
import sys
import time
import json
import socket
import logging
import traceback
from pathlib import Path
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [consumer] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---- Config ----
CKPT_IN_DIR        = os.environ.get("EAGLE3_CKPT_DIR",      "/data/training/eagle3_v2_apilog/7")
VOCAB_DIR          = os.environ.get("EAGLE3_VOCAB_DIR",     "/data/training/eagle3_v2_apilog/7")
CKPT_OUT_DIR       = os.environ.get("EAGLE3_OUT_DIR",       "/data/training/eagle3_v5_online")
K2_5_PATH          = os.environ.get("K2_5_MODEL_PATH",      "/data/models/Kimi-K2.5")
GLOBAL_STEP_OFFSET = int(os.environ.get("GLOBAL_STEP_OFFSET", "0"))
CONSUMER_DDP_SIZE  = int(os.environ.get("CONSUMER_DDP_SIZE", "8"))

H             = 7168
MAX_SEQ_LEN   = 4096
LR            = 1e-5
WARMUP_STEPS  = 100
LR_MIN_FACTOR = 0.1
CKPT_INTERVAL = 500
LOG_INTERVAL  = 50
GRAD_CLIP     = 1.0
TTT_STEPS     = 3
TTT_DECAY     = 1.0

SYNC_PORT = 29501
P2P_PORT  = int(os.environ.get("P2P_NCCL_PORT", "29503"))


# ---- Checkpoint loading (same as Phase 4) ----

def patch_config(ckpt_dir: str, k2_5_path: str):
    config_path = Path(ckpt_dir) / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    verifier = cfg["speculators_config"]["verifier"]
    old_path = verifier.get("name_or_path", "")
    if old_path != k2_5_path:
        verifier["name_or_path"] = k2_5_path
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        log.info(f"Patched verifier.name_or_path: {old_path!r} → {k2_5_path!r}")
    else:
        log.info(f"Config already points to {k2_5_path!r}, no patch needed")


def load_eagle3(ckpt_dir: str, vocab_dir: str, device: torch.device):
    import numpy as np
    from speculators.models.eagle3 import Eagle3SpeculatorConfig
    from speculators.models.eagle3.core import Eagle3DraftModel
    from safetensors.torch import load_file as load_safetensors

    ckpt_path = Path(ckpt_dir)
    d2t = torch.from_numpy(np.load(str(Path(vocab_dir) / "d2t.npy")))
    t2d = torch.from_numpy(np.load(str(Path(vocab_dir) / "t2d.npy")))
    log.info(f"Vocab mappings: d2t={d2t.shape} t2d={t2d.shape}")

    config = Eagle3SpeculatorConfig.from_pretrained(str(ckpt_path))
    log.info(f"Eagle3 config: draft_vocab={config.draft_vocab_size}  "
             f"verifier={config.speculators_config.verifier.name_or_path}")

    log.info("Building Eagle3DraftModel (loading K2.5 embeddings) ...")
    model = Eagle3DraftModel(config, t2d=t2d, d2t=d2t)

    shard_files = sorted(ckpt_path.glob("model-*.safetensors"))
    if not shard_files:
        shard_files = [ckpt_path / "model.safetensors"]

    log.info(f"Loading {len(shard_files)} safetensors shard(s) ...")
    state_dict: dict = {}
    for shard in shard_files:
        state_dict.update(load_safetensors(str(shard), device="cpu"))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.info(f"  Missing keys: {len(missing)} (expected: verifier weights not saved)")
    if unexpected:
        log.warning(f"  Unexpected keys: {unexpected[:5]}")

    model = model.to(device=device, dtype=torch.float32)
    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Eagle3DraftModel loaded: {n_params:,} trainable params  dtype=float32  device={device}")
    return model, d2t, t2d


# ---- TCP sync ----

def wait_for_producer_ready(producer_addr: str, sync_port: int = SYNC_PORT, poll_interval: float = 5.0):
    log.info(f"TCP sync: polling {producer_addr}:{sync_port} every {poll_interval:.0f}s ...")
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((producer_addr, sync_port))
            msg = s.recv(5)
            s.close()
            if msg == b"ready":
                log.info("TCP sync: received 'ready' — producer K2.5 loaded")
                return
            log.warning(f"TCP sync: unexpected message {msg!r}, retrying ...")
        except (ConnectionRefusedError, socket.timeout, OSError):
            pass
        time.sleep(poll_interval)


# ---- Checkpoint saving (only called by local_rank==0) ----

def save_checkpoint(model, optimizer, scheduler, step: int, out_dir: str):
    from safetensors.torch import save_file as save_safetensors
    ckpt_dir = Path(out_dir) / str(step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP to get the underlying module
    raw_model = model.module if isinstance(model, DDP) else model
    state_dict = {k: v for k, v in raw_model.state_dict().items()
                  if k not in (raw_model._keys_to_ignore_on_save or [])}
    save_safetensors(state_dict, str(ckpt_dir / "model.safetensors"))

    torch.save(optimizer.state_dict(), str(ckpt_dir / "optimizer_state_dict.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), str(ckpt_dir / "scheduler_state_dict.pt"))

    import shutil
    for fname in ("config.json", "config.py"):
        src = Path(CKPT_IN_DIR) / fname
        if src.exists():
            shutil.copy2(src, ckpt_dir / fname)

    log.info(f"Checkpoint saved → {ckpt_dir}")


# ---- Main ----

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    global_rank = local_rank + 1  # global rank in P2P group (producer=0)
    is_chief = (local_rank == 0)  # chief: saves checkpoint, sends loss

    master_addr = os.environ.get("MASTER_ADDR", "10.83.115.10")

    # Only chief patches config (file write race condition otherwise)
    if is_chief:
        patch_config(CKPT_IN_DIR, K2_5_PATH)
    # Brief barrier-like wait so non-chief ranks see patched config
    time.sleep(2 if not is_chief else 0)

    log.info(f"local_rank={local_rank} global_rank={global_rank} is_chief={is_chief}")

    speculators_src = os.environ.get("SPECULATORS_PATH", "")
    if speculators_src:
        sys.path.insert(0, speculators_src)
    model, d2t, t2d = load_eagle3(CKPT_IN_DIR, VOCAB_DIR, device)

    if model.d2t is not None:
        model.d2t = model.d2t.to(device)
    if model.t2d is not None:
        model.t2d = model.t2d.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.999), weight_decay=0.01,
    )

    import math

    def lr_lambda_fn(step: int) -> float:
        if step < WARMUP_STEPS:
            return (step + 1) / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(1, 1_000_000 - WARMUP_STEPS)
        progress = min(progress, 1.0)
        return LR_MIN_FACTOR + (1.0 - LR_MIN_FACTOR) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fn)

    # Resume optimizer/scheduler (only chief loads, others share via DDP param sync)
    if GLOBAL_STEP_OFFSET > 0 and is_chief:
        opt_path = Path(CKPT_IN_DIR) / "optimizer_state_dict.pt"
        sch_path = Path(CKPT_IN_DIR) / "scheduler_state_dict.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(str(opt_path), map_location=device))
            log.info(f"Optimizer state loaded from {opt_path}")
        else:
            log.warning(f"No optimizer state at {opt_path}, fast-forwarding LR schedule")
            for _ in range(GLOBAL_STEP_OFFSET):
                scheduler.step()
        if sch_path.exists() and opt_path.exists():
            scheduler.load_state_dict(torch.load(str(sch_path), map_location="cpu"))
            log.info(f"Scheduler state loaded from {sch_path}")
        log.info(f"Resuming from global_step={GLOBAL_STEP_OFFSET}")

    # TCP sync: all ranks independently connect (producer accepts CONSUMER_DDP_SIZE conns)
    wait_for_producer_ready(master_addr, sync_port=SYNC_PORT)

    # Init P2P dist group (world_size = CONSUMER_DDP_SIZE + 1)
    world_size = CONSUMER_DDP_SIZE + 1
    log.info(f"Init P2P dist group: global_rank={global_rank} world={world_size} "
             f"addr={master_addr}:{P2P_PORT}")
    dist.init_process_group(
        backend="nccl", rank=global_rank, world_size=world_size,
        init_method=f"tcp://{master_addr}:{P2P_PORT}",
        timeout=timedelta(hours=2),
    )
    log.info("P2P dist group initialized")

    # DDP group: consumer-internal AllReduce via NVLink
    consumer_ranks = list(range(1, CONSUMER_DDP_SIZE + 1))
    ddp_group = dist.new_group(ranks=consumer_ranks, timeout=timedelta(hours=2))
    model = DDP(model, process_group=ddp_group, device_ids=[local_rank])
    log.info(f"DDP group initialized (ranks={consumer_ranks})")

    # Warmup
    warmup_t = torch.zeros(1, dtype=torch.float32, device=device)
    dist.recv(warmup_t, src=0)
    dist.send(warmup_t, dst=0)
    log.info("P2P NCCL warm-up done — starting training loop")

    loss_send = torch.zeros(1, dtype=torch.float32, device=device)
    Path(CKPT_OUT_DIR).mkdir(parents=True, exist_ok=True)

    losses = []
    global_step = GLOBAL_STEP_OFFSET

    try:
        while True:
            t0 = time.perf_counter()

            # Receive seq_len first
            meta_recv = torch.zeros(1, dtype=torch.int64, device=device)
            dist.recv(meta_recv, src=0)
            seq_len = int(meta_recv[0].item())

            # Skip sentinel — coordinate across ALL DDP ranks to avoid AllReduce deadlock
            skip_flag = torch.zeros(1, dtype=torch.int32, device=device)
            if seq_len == -1:
                skip_flag[0] = 1
            dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX, group=ddp_group)
            if skip_flag.item() > 0:
                log.info(f'step {global_step}: skip sentinel received (any rank), coordinating skip')
                if is_chief:
                    loss_send[0] = float('nan')
                    dist.send(loss_send, dst=0)
                global_step += 1
                continue

            # Allocate exact-size buffers (variable per step)
            aux_buf  = torch.empty(1, seq_len, 3 * H, dtype=torch.bfloat16, device=device)
            last_buf = torch.empty(1, seq_len, H,     dtype=torch.bfloat16, device=device)
            ids_buf  = torch.empty(1, seq_len,        dtype=torch.int64,    device=device)
            mask_buf = torch.empty(1, seq_len,        dtype=torch.float32,  device=device)

            dist.recv(aux_buf,  src=0)
            dist.recv(last_buf, src=0)
            dist.recv(ids_buf,  src=0)
            dist.recv(mask_buf, src=0)
            recv_t = time.perf_counter() - t0

            nb = aux_buf.nbytes + last_buf.nbytes + ids_buf.nbytes + mask_buf.nbytes

            # NaN guard — coordinate skip across ALL DDP ranks to avoid AllReduce deadlock
            nan_flag = torch.zeros(1, dtype=torch.int32, device=device)
            if not (torch.isfinite(aux_buf).all() and torch.isfinite(last_buf).all()):
                nan_flag[0] = 1
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX, group=ddp_group)
            if nan_flag.item() > 0:
                log.warning(f"step {global_step}: NaN/Inf in hidden states (any rank), skipping")
                if is_chief:
                    loss_send[0] = float('nan')
                    dist.send(loss_send, dst=0)
                global_step += 1
                continue

            lengths = torch.tensor([seq_len], dtype=torch.long, device=device)

            optimizer.zero_grad()
            _draft_tokens, loss, metrics = model(
                hidden_states=aux_buf.float(),
                input_ids=ids_buf,
                lengths=lengths,
                loss_mask=mask_buf,
                verifier_last_hidden_states=last_buf.float(),
                ttt_steps=TTT_STEPS,
                ttt_step_loss_decay=TTT_DECAY,
            )
            # DDP backward: auto AllReduce gradients across consumer ranks via NVLink

            loss_val = loss.item()
            # Coordinate non-finite loss skip across ALL DDP ranks to avoid AllReduce deadlock
            nonfinite_flag = torch.zeros(1, dtype=torch.int32, device=device)
            if not torch.isfinite(loss):
                nonfinite_flag[0] = 1
            dist.all_reduce(nonfinite_flag, op=dist.ReduceOp.MAX, group=ddp_group)
            if nonfinite_flag.item() > 0:
                log.warning(f"step {global_step}: loss={loss_val} non-finite (any rank), skipping backward")
                if is_chief:
                    loss_send[0] = loss_val
                    dist.send(loss_send, dst=0)
                global_step += 1
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if not torch.isfinite(grad_norm):
                log.warning(f"step {global_step}: grad_norm non-finite, skipping optimizer step")
                optimizer.zero_grad()
            else:
                optimizer.step()
            scheduler.step()
            train_t = time.perf_counter() - t0

            if is_chief:
                losses.append(loss_val)
                loss_send[0] = loss_val
                dist.send(loss_send, dst=0)

            if global_step % LOG_INTERVAL == 0 and is_chief:
                lr_now = optimizer.param_groups[0]["lr"]
                full_acc = metrics.get("full_acc_0", torch.tensor(0.0))
                full_acc = full_acc.item() if isinstance(full_acc, torch.Tensor) else full_acc
                log.info(
                    f"step {global_step:06d}: seq_len={seq_len}  "
                    f"recv={recv_t*1000:.1f}ms  train={train_t*1000:.0f}ms  "
                    f"bw={nb/recv_t/1e9:.2f}GB/s  "
                    f"loss={loss_val:.4f}  full_acc_0={full_acc:.4f}  lr={lr_now:.2e}"
                )

            if (global_step + 1) % CKPT_INTERVAL == 0 and is_chief:
                save_checkpoint(model, optimizer, scheduler, global_step + 1, CKPT_OUT_DIR)

            global_step += 1

    except Exception as e:
        log.error(f"EXCEPTION at step {global_step} rank {global_rank}: {type(e).__name__}: {e}")
        log.error(traceback.format_exc())
        raise

    finally:
        if losses and is_chief:
            log.info(f"\n=== CONSUMER SUMMARY local_rank={local_rank} ({global_step} steps) ===")
            log.info(
                f"  loss: {losses[0]:.4f} → {losses[-1]:.4f} "
                f"({'↓ decreasing' if losses[-1] < losses[0] else '↑ not decreasing'})"
            )
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        log.info(f"local_rank={local_rank} DONE")


if __name__ == "__main__":
    main()
