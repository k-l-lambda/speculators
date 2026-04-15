"""
EAGLE3 Producer-Consumer Phase 4 — Producer (youyun.37 / host-10-83-115-10)

Key improvements over Phase 3:
  - Dataset: apilog_k25_eagle3/train/ (5761 files, 4096 seq_len) — same as offline training
  - MAX_SEQ_LEN: 4096 (was 2048 in Phase 3) — no more sequence truncation
  - NaN scrubbing: nan_to_num applied before send (eliminates consumer NaN skips)
  - Length mismatch tolerance relaxed: allow up to ±2 tokens

Transfer per step (MAX_SEQ_LEN=4096, padded):
  meta      [1]               int64    actual seq_len after shift + truncate
  aux_hs    [1, 4096, 21504]  bf16     ~168 MB
  last_hs   [1, 4096,  7168]  bf16     ~56 MB
  input_ids [1, 4096]         int64    ~32 KB
  loss_mask [1, 4096]         float32  ~16 KB
  Total: ~224 MB/step

Port map (unchanged):
  29500: torchrun rendezvous (C10d TCPStore, job lifetime)
  29501: TCP sync signal (producer → consumer "K2.5 ready")
  29502: vLLM TP group (VLLM_MASTER_PORT)
  29503: P2P NCCL group (P2P_NCCL_PORT)
"""

import os
import sys
import time
import socket
import logging
from pathlib import Path

import torch
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [producer] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---- Config ----
MODEL_PATH   = "/data/models/Kimi-K2.5"
LAYER_IDS    = [2, 30, 58, 60]
DATA_DIR     = "/data/datasets/apilog_k25_eagle3/train"   # 5761 files, 4096 seq_len (matches offline)
MAX_SEQ_LEN  = 4096
H            = 7168
NUM_EPOCHS   = 3
LOG_INTERVAL = 50
WARMUP_STEPS = 2

VLLM_PORT = int(os.environ.get("VLLM_MASTER_PORT", "29502"))
SYNC_PORT = 29501
P2P_PORT  = int(os.environ.get("P2P_NCCL_PORT", "29503"))


def tcp_signal_consumer(sync_port: int = SYNC_PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", sync_port))
    s.listen(1)
    log.info(f"TCP sync: listening on port {sync_port}, waiting for consumer ...")
    conn, addr = s.accept()
    log.info(f"TCP sync: consumer connected from {addr}")
    conn.sendall(b"ready")
    conn.close()
    s.close()
    log.info("TCP sync: 'ready' sent — both sides will init NCCL now")


def prepare_batch(result: dict, loss_mask: torch.Tensor, max_len: int, device: torch.device):
    """
    Convert raw vLLM result + loss_mask into padded fixed-size transfer tensors.

    Uses speculators' standardize_data_v1 + shift_batch (same as offline training).
    Applies nan_to_num scrubbing before returning (prevents consumer NaN skips).

    Returns:
        seq_len      int      actual sequence length after shift + truncate
        aux_hs       [1, max_len, 3H]  bf16  (padded, NaN-scrubbed)
        last_hs      [1, max_len, H]   bf16  (padded, NaN-scrubbed)
        input_ids    [1, max_len]      int64 (padded with 0)
        loss_mask_t  [1, max_len]      float32 (padded with 0.0)
    """
    from speculators.train.data import standardize_data_v1, shift_batch

    seq_len_raw = len(result["input_ids"])
    result_input_ids = torch.tensor(result["input_ids"], dtype=torch.long)

    raw = {
        "input_ids": result_input_ids,
        "hidden_states": result["hidden_states"],
        "loss_mask": loss_mask[:seq_len_raw],
    }
    data = standardize_data_v1(raw)
    data["lengths"] = torch.tensor([data["input_ids"].shape[0]], dtype=torch.long)
    data["position_ids"] = torch.arange(data["input_ids"].shape[0], dtype=torch.long)
    data = shift_batch(data)

    seq_len = min(data["input_ids"].shape[0], max_len)

    def _pad(t: torch.Tensor, pad_val: float = 0.0) -> torch.Tensor:
        t = t[:seq_len]
        deficit = max_len - t.shape[0]
        if deficit > 0:
            pad_shape = list(t.shape)
            pad_shape[0] = deficit
            t = torch.cat([t, torch.full(pad_shape, pad_val, dtype=t.dtype)], dim=0)
        return t.unsqueeze(0)

    aux_hs  = _pad(data["hidden_states"]).to(device=device, dtype=torch.bfloat16)
    last_hs = _pad(data["verifier_last_hidden_states"]).to(device=device, dtype=torch.bfloat16)

    # NaN scrubbing: replace NaN/Inf with 0 on producer side to avoid consumer skips
    nan_aux  = (~torch.isfinite(aux_hs)).sum().item()
    nan_last = (~torch.isfinite(last_hs)).sum().item()
    if nan_aux + nan_last > 0:
        log.warning(f"  NaN/Inf scrubbed: aux={nan_aux} last={nan_last} — replacing with 0")
        aux_hs  = torch.nan_to_num(aux_hs,  nan=0.0, posinf=0.0, neginf=0.0)
        last_hs = torch.nan_to_num(last_hs, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        seq_len,
        aux_hs,
        last_hs,
        _pad(data["input_ids"], pad_val=0).to(device=device, dtype=torch.int64),
        _pad(data["loss_mask"].float(), pad_val=0.0).to(device=device),
    )


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    master_addr = os.environ.get("MASTER_ADDR", "10.83.115.10")

    # 1. Clear torchrun env vars + redirect MASTER_PORT before vLLM spawn
    os.environ["MASTER_PORT"] = str(VLLM_PORT)
    log.info(f"Redirected MASTER_PORT to {VLLM_PORT} for vLLM TP workers")
    _clear = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE",
        "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
    ]
    cleared = {k: os.environ.pop(k) for k in _clear if k in os.environ}
    if cleared:
        log.info(f"Cleared {len(cleared)} env vars: {list(cleared.keys())}")

    # 2. Load K2.5 via VllmHiddenStatesGenerator
    sys.path.insert(0, "/workspace/speculators/src")
    from speculators.data_generation.vllm_hidden_states_generator import (
        VllmHiddenStatesGenerator,
    )

    log.info(f"Loading K2.5 TP=8 max_model_len={MAX_SEQ_LEN} layers={LAYER_IDS}")
    t_load = time.time()
    generator = VllmHiddenStatesGenerator(
        model_path=MODEL_PATH,
        layer_ids=LAYER_IDS,
        max_model_len=MAX_SEQ_LEN,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=8,
        enforce_eager=True,
    )
    log.info(f"K2.5 loaded in {time.time() - t_load:.1f}s")

    # 3. TCP sync → both init NCCL P2P
    tcp_signal_consumer()

    from datetime import timedelta
    log.info(f"Init P2P dist group: rank=0 world=2 addr={master_addr}:{P2P_PORT}")
    dist.init_process_group(
        backend="nccl", rank=0, world_size=2,
        init_method=f"tcp://{master_addr}:{P2P_PORT}",
        timeout=timedelta(minutes=5),
    )
    assert dist.get_rank() == 0 and dist.get_world_size() == 2
    log.info("P2P dist group initialized")

    warmup = torch.zeros(1, dtype=torch.float32, device=device)
    dist.send(warmup, dst=1)
    dist.recv(warmup, src=1)
    log.info("P2P NCCL warm-up done")

    # 4. Pre-allocate fixed-size send buffers
    aux_buf   = torch.empty(1, MAX_SEQ_LEN, 3 * H, dtype=torch.bfloat16, device=device)
    last_buf  = torch.empty(1, MAX_SEQ_LEN, H,     dtype=torch.bfloat16, device=device)
    ids_buf   = torch.empty(1, MAX_SEQ_LEN,        dtype=torch.int64,    device=device)
    mask_buf  = torch.empty(1, MAX_SEQ_LEN,        dtype=torch.float32,  device=device)
    meta_send = torch.zeros(1, dtype=torch.int64,  device=device)
    loss_recv = torch.zeros(1, dtype=torch.float32, device=device)

    nb = aux_buf.nbytes + last_buf.nbytes + ids_buf.nbytes + mask_buf.nbytes
    log.info(
        f"Transfer/step: aux={aux_buf.nbytes//1024//1024}MB "
        f"last={last_buf.nbytes//1024//1024}MB "
        f"ids+mask={(ids_buf.nbytes + mask_buf.nbytes)//1024}KB "
        f"total={nb//1024//1024}MB"
    )

    # 5. Dataset
    pt_files = sorted(
        Path(DATA_DIR).glob("data_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    log.info(f"Dataset: {len(pt_files)} files  epochs={NUM_EPOCHS}")
    if not pt_files:
        log.error(f"No .pt files in {DATA_DIR}!")
        dist.destroy_process_group()
        return

    # 6. Training loop
    global_step = 0
    latencies = []

    for epoch in range(NUM_EPOCHS):
        log.info(f"=== Epoch {epoch}/{NUM_EPOCHS} ({len(pt_files)} steps) ===")

        for file_idx, pt_file in enumerate(pt_files):
            t0 = time.perf_counter()

            raw = torch.load(str(pt_file), map_location="cpu", weights_only=True)
            input_ids_disk = raw["input_ids"]
            loss_mask_disk = raw["loss_mask"]

            results = generator.generate([input_ids_disk.tolist()])
            result = results[0]
            gen_t = time.perf_counter() - t0

            # Relaxed length mismatch tolerance (±2) for 4096-token sequences
            result_len = len(result["input_ids"])
            disk_len   = len(input_ids_disk)
            if abs(result_len - disk_len) > 2:
                log.warning(
                    f"step {global_step}: length mismatch disk={disk_len} vllm={result_len}, "
                    f"skipping {pt_file.name}"
                )
                continue

            try:
                seq_len, aux_hs, last_hs, input_ids_t, loss_mask_t = prepare_batch(
                    result, loss_mask_disk, MAX_SEQ_LEN, device
                )
            except Exception as e:
                log.warning(f"step {global_step}: prepare_batch failed: {e}, skipping")
                continue

            aux_buf[:] = aux_hs
            last_buf[:] = last_hs
            ids_buf[:] = input_ids_t
            mask_buf[:] = loss_mask_t
            meta_send[0] = seq_len

            dist.send(meta_send, dst=1)
            dist.send(aux_buf, dst=1)
            dist.send(last_buf, dst=1)
            dist.send(ids_buf, dst=1)
            dist.send(mask_buf, dst=1)
            send_t = time.perf_counter() - t0

            dist.recv(loss_recv, src=1)
            total_t = time.perf_counter() - t0

            if global_step >= WARMUP_STEPS:
                latencies.append(total_t)

            if global_step % LOG_INTERVAL == 0:
                log.info(
                    f"epoch={epoch} file={file_idx:05d} global={global_step:06d} "
                    f"seq_len={seq_len}  gen={gen_t*1000:.0f}ms  "
                    f"send+recv={(total_t - gen_t)*1000:.0f}ms  "
                    f"total={total_t*1000:.0f}ms  "
                    f"consumer_loss={loss_recv.item():.4f}"
                )

            global_step += 1

        if latencies:
            avg = sum(latencies) / len(latencies)
            log.info(
                f"\n=== PRODUCER EPOCH {epoch} SUMMARY ({len(latencies)} steps) ===\n"
                f"  avg total latency: {avg*1000:.0f} ms/step\n"
                f"  transfer: {nb//1024//1024} MB/step"
            )

    dist.destroy_process_group()
    log.info("DONE")


if __name__ == "__main__":
    main()
