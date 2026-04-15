"""
EAGLE3 Producer-Consumer Phase 3 — Consumer (host-10-83-115-14)

Receives hidden states + input_ids + loss_mask from producer via NCCL P2P.
Loads real Eagle3DraftModel from training checkpoint.
Trains with KL divergence loss (ttt_steps=3).
Saves checkpoints every CKPT_INTERVAL steps.

Setup (run once before deploying LWS, or handled in startup script):
  On consumer node, rsync from producer:
    rsync -avP host-10-83-115-10:/data/training/eagle3_v2_apilog/7/ /tmp/eagle3_ckpt_p3/
    rsync -avP host-10-83-115-10:/data/datasets/apilog_k25_eagle3/vocab_mapping/ /tmp/vocab_p3/
  Patch verifier path in config.json:
    sed -i 's|/data/.cache_claude/huggingface/hub/models--moonshotai--Kimi-K2.5/snapshots/[^"]*|/data/models/Kimi-K2.5|g' /tmp/eagle3_ckpt_p3/config.json

Transfer protocol (producer → consumer, fixed MAX_SEQ_LEN=2048):
  meta     [1]                   int64   actual seq_len
  aux_hs   [1, MAX_SEQ_LEN, 3H]  bf16    aux hidden states (padded)
  last_hs  [1, MAX_SEQ_LEN, H]   bf16    verifier last hidden state (padded)
  input_ids [1, MAX_SEQ_LEN]     int64   (padded with 0)
  loss_mask [1, MAX_SEQ_LEN]     float32 (padded with 0.0)
  loss      [1]                   float32 consumer → producer
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [consumer] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---- Config ----
CKPT_IN_DIR    = os.environ.get("EAGLE3_CKPT_DIR",   "/tmp/eagle3_ckpt_p3")
VOCAB_DIR      = os.environ.get("EAGLE3_VOCAB_DIR",  "/tmp/vocab_p3")
CKPT_OUT_DIR   = os.environ.get("EAGLE3_OUT_DIR",    "/data/training/eagle3_v3_online")
K2_5_PATH      = os.environ.get("K2_5_MODEL_PATH",   "/data/models/Kimi-K2.5")

H              = 7168
MAX_SEQ_LEN    = 2048
LR             = 1e-5              # lower than Phase 2 since we start from pretrained weights
WARMUP_STEPS   = 100
LR_MIN_FACTOR  = 0.1               # cosine decay floor: LR * LR_MIN_FACTOR
CKPT_INTERVAL  = 500               # save checkpoint every N steps
LOG_INTERVAL   = 50
GRAD_CLIP      = 1.0
TTT_STEPS      = 3
TTT_DECAY      = 1.0

SYNC_PORT  = 29501
P2P_PORT   = int(os.environ.get("P2P_NCCL_PORT", "29503"))


# ---- Checkpoint loading ----

def patch_config(ckpt_dir: str, k2_5_path: str):
    """Patch config.json to use local K2.5 path."""
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
    """Load Eagle3DraftModel from speculators-format checkpoint."""
    import numpy as np
    from speculators.models.eagle3 import Eagle3SpeculatorConfig
    from speculators.models.eagle3.core import Eagle3DraftModel
    from safetensors.torch import load_file as load_safetensors

    ckpt_path = Path(ckpt_dir)

    # Load vocab mappings
    d2t = torch.from_numpy(np.load(str(Path(vocab_dir) / "d2t.npy")))
    t2d = torch.from_numpy(np.load(str(Path(vocab_dir) / "t2d.npy")))
    log.info(f"Vocab mappings: d2t={d2t.shape} t2d={t2d.shape}")

    # Load config (verifier path must be patched before this call)
    config = Eagle3SpeculatorConfig.from_pretrained(str(ckpt_path))
    log.info(
        f"Eagle3 config: draft_vocab={config.draft_vocab_size}  "
        f"verifier={config.speculators_config.verifier.name_or_path}"
    )

    # Instantiate model on CPU first
    log.info("Building Eagle3DraftModel (loading K2.5 embeddings) ...")
    model = Eagle3DraftModel(config, t2d=t2d, d2t=d2t)

    # Load safetensors checkpoint
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

    model = model.to(device=device, dtype=torch.bfloat16)
    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Eagle3DraftModel loaded: {n_params:,} trainable params  dtype=bfloat16  device={device}")
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
                log.info("TCP sync: received 'ready' — producer K2.5 loaded, init NCCL now")
                return
            log.warning(f"TCP sync: unexpected message {msg!r}, retrying ...")
        except (ConnectionRefusedError, socket.timeout, OSError):
            pass
        time.sleep(poll_interval)


# ---- Checkpoint saving ----

def save_checkpoint(model, optimizer, scheduler, step: int, out_dir: str):
    from safetensors.torch import save_file as save_safetensors
    ckpt_dir = Path(out_dir) / str(step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    state_dict = {k: v for k, v in model.state_dict().items()
                  if k not in (model._keys_to_ignore_on_save or [])}
    save_safetensors(state_dict, str(ckpt_dir / "model.safetensors"))

    # Save optimizer + scheduler state
    torch.save(optimizer.state_dict(), str(ckpt_dir / "optimizer_state_dict.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), str(ckpt_dir / "scheduler_state_dict.pt"))

    # Copy config files
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

    master_addr = os.environ.get("MASTER_ADDR", "10.83.115.10")

    # 1. Patch config to use local K2.5 path (idempotent)
    patch_config(CKPT_IN_DIR, K2_5_PATH)

    # 2. Load Eagle3 from checkpoint (while producer loads K2.5)
    speculators_src = os.environ.get("SPECULATORS_PATH", "/workspace/speculators/src")
    sys.path.insert(0, speculators_src)
    log.info(f"speculators path: {speculators_src}")
    model, d2t, t2d = load_eagle3(CKPT_IN_DIR, VOCAB_DIR, device)

    # Move vocab mappings to device
    if model.d2t is not None:
        model.d2t = model.d2t.to(device)
    if model.t2d is not None:
        model.t2d = model.t2d.to(device)

    # 3. Init optimizer + LR scheduler
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

    # 4. Wait for producer TCP sync
    wait_for_producer_ready(master_addr, sync_port=SYNC_PORT)

    # 5. Init P2P NCCL group on port 29503
    log.info(f"Init P2P dist group: rank=1 world=2 addr={master_addr}:{P2P_PORT}")
    dist.init_process_group(
        backend="nccl", rank=1, world_size=2,
        init_method=f"tcp://{master_addr}:{P2P_PORT}",
        timeout=timedelta(minutes=5),
    )
    assert dist.get_rank() == 1 and dist.get_world_size() == 2
    log.info("P2P dist group initialized")

    # 5b. Warm-up NCCL P2P communicator
    warmup_t = torch.zeros(1, dtype=torch.float32, device=device)
    dist.recv(warmup_t, src=0)
    dist.send(warmup_t, dst=0)
    log.info("P2P NCCL communicator warm-up done — starting training loop")

    # 6. Pre-allocate receive buffers
    meta_recv  = torch.zeros(1, dtype=torch.int64,    device=device)
    aux_buf    = torch.empty(1, MAX_SEQ_LEN, 3 * H,   dtype=torch.bfloat16, device=device)
    last_buf   = torch.empty(1, MAX_SEQ_LEN, H,       dtype=torch.bfloat16, device=device)
    ids_buf    = torch.empty(1, MAX_SEQ_LEN,           dtype=torch.int64,    device=device)
    mask_buf   = torch.empty(1, MAX_SEQ_LEN,           dtype=torch.float32,  device=device)
    loss_send  = torch.zeros(1, dtype=torch.float32,  device=device)

    nb = aux_buf.nbytes + last_buf.nbytes + ids_buf.nbytes + mask_buf.nbytes
    log.info(f"Receive buffers: {nb/1e6:.0f} MB/step  (MAX_SEQ_LEN={MAX_SEQ_LEN})")

    Path(CKPT_OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 7. Training loop
    losses = []
    global_step = 0

    try:
        while True:   # Producer controls loop termination
            t0 = time.perf_counter()

            # Receive metadata (seq_len) + tensors
            dist.recv(meta_recv, src=0)
            seq_len = int(meta_recv[0].item())

            dist.recv(aux_buf, src=0)
            dist.recv(last_buf, src=0)
            dist.recv(ids_buf, src=0)
            dist.recv(mask_buf, src=0)
            recv_t = time.perf_counter() - t0

            # Slice to actual seq_len (remove padding for efficiency)
            aux_hs   = aux_buf[:, :seq_len, :].contiguous()    # [1, S, 3H]
            last_hs  = last_buf[:, :seq_len, :].contiguous()   # [1, S, H]
            input_ids = ids_buf[:, :seq_len].contiguous()      # [1, S]
            loss_mask = mask_buf[:, :seq_len].contiguous()     # [1, S]
            lengths   = torch.tensor([seq_len], dtype=torch.long, device=device)

            # Forward + loss + backward
            optimizer.zero_grad()
            _draft_tokens, loss, metrics = model(
                hidden_states=aux_hs,
                input_ids=input_ids,
                lengths=lengths,
                loss_mask=loss_mask,
                verifier_last_hidden_states=last_hs,   # triggers KL loss
                ttt_steps=TTT_STEPS,
                ttt_step_loss_decay=TTT_DECAY,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            train_t = time.perf_counter() - t0

            loss_val = loss.item()
            losses.append(loss_val)

            # Send loss back to producer
            loss_send[0] = loss_val
            dist.send(loss_send, dst=0)

            if global_step % LOG_INTERVAL == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                full_acc = metrics.get("full_acc_0", torch.tensor(0.0)).item() if isinstance(metrics.get("full_acc_0"), torch.Tensor) else metrics.get("full_acc_0", 0.0)
                log.info(
                    f"step {global_step:06d}: seq_len={seq_len}  "
                    f"recv={recv_t*1000:.1f}ms  train={train_t*1000:.0f}ms  "
                    f"bw={nb/recv_t/1e9:.2f}GB/s  "
                    f"loss={loss_val:.4f}  full_acc_0={full_acc:.4f}  lr={lr_now:.2e}"
                )

            if (global_step + 1) % CKPT_INTERVAL == 0:
                save_checkpoint(model, optimizer, scheduler, global_step + 1, CKPT_OUT_DIR)

            global_step += 1

    except Exception as e:
        log.error(f"EXCEPTION in consumer loop at step {global_step}: {type(e).__name__}: {e}")
        log.error(traceback.format_exc())
        raise

    finally:
        # Final summary
        if losses:
            log.info(f"\n=== CONSUMER SUMMARY ({global_step} steps) ===")
            log.info(
                f"  loss: {losses[0]:.4f} → {losses[-1]:.4f} "
                f"({'↓ decreasing' if losses[-1] < losses[0] else '↑ not decreasing'})"
            )

        try:
            dist.destroy_process_group()
        except Exception:
            pass
        log.info("DONE")


if __name__ == "__main__":
    main()
