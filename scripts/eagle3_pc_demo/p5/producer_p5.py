"""
EAGLE3 Producer-Consumer Phase 5 — Producer (host-10-83-115-10)

Key improvements over Phase 4:
  - Batch=8: K2.5 generates hidden states for 8 sequences per round
  - Variable-size transfer: sends actual seq_len tensors (not padded to 4096)
    avg 8×25 MB = 200 MB/round vs 224 MB/step in Phase 4
  - Sends to 8 consumer ranks (DP=8 DDP on .14)
  - world_size=9: rank 0 (producer) + ranks 1-8 (consumers)

Transfer per step per consumer rank (actual seq_len, avg ~591 tok):
  meta      [1]               int64    actual seq_len (-1 = skip)
  aux_hs    [1, seq_len, 3H]  bf16     ~25 MB avg (was 168 MB padded)
  last_hs   [1, seq_len, H]   bf16     ~8.5 MB avg (was 56 MB padded)
  input_ids [1, seq_len]      int64    ~4.7 KB avg
  loss_mask [1, seq_len]      float32  ~2.4 KB avg
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
MODEL_PATH        = "/data/models/Kimi-K2.5"
LAYER_IDS         = [2, 30, 58, 60]
DATA_DIR          = "/data/datasets/apilog_k25_eagle3/train_40k_greedy_v2"
MAX_SEQ_LEN       = 4096
H                 = 7168
NUM_EPOCHS        = 3
LOG_INTERVAL      = 50
WARMUP_STEPS      = 2
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE", "8"))
CONSUMER_DDP_SIZE = int(os.environ.get("CONSUMER_DDP_SIZE", "8"))

VLLM_PORT = int(os.environ.get("VLLM_MASTER_PORT", "29502"))
SYNC_PORT = 29501
P2P_PORT  = int(os.environ.get("P2P_NCCL_PORT", "29503"))


def tcp_signal_consumers(sync_port: int = SYNC_PORT, n_consumers: int = CONSUMER_DDP_SIZE):
    """Accept n_consumers TCP connections and send 'ready' to each."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", sync_port))
    s.listen(n_consumers)
    log.info(f"TCP sync: listening on port {sync_port}, waiting for {n_consumers} consumers ...")
    for i in range(n_consumers):
        conn, addr = s.accept()
        conn.sendall(b"ready")
        conn.close()
        log.info(f"TCP sync: consumer {i+1}/{n_consumers} connected from {addr}")
    s.close()
    log.info("TCP sync: all consumers notified — both sides will init NCCL now")


def prepare_batch(result: dict, loss_mask: torch.Tensor, max_len: int, device: torch.device):
    """
    Same as Phase 4 prepare_batch but returns actual-size tensors (no padding).
    Returns (seq_len, aux_hs, last_hs, input_ids, loss_mask) where tensors are
    [1, seq_len, ...] — caller slices to seq_len, no MAX_SEQ_LEN padding.
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

    # Slice to actual seq_len (no padding)
    aux_hs  = data["hidden_states"][:seq_len].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    last_hs = data["verifier_last_hidden_states"][:seq_len].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    ids_t   = data["input_ids"][:seq_len].unsqueeze(0).to(device=device, dtype=torch.int64)
    mask_t  = data["loss_mask"].float()[:seq_len].unsqueeze(0).to(device=device)

    # NaN scrubbing
    nan_aux  = (~torch.isfinite(aux_hs)).sum().item()
    nan_last = (~torch.isfinite(last_hs)).sum().item()
    if nan_aux + nan_last > 0:
        log.warning(f"  NaN/Inf scrubbed: aux={nan_aux} last={nan_last}")
        aux_hs  = torch.nan_to_num(aux_hs,  nan=0.0, posinf=0.0, neginf=0.0)
        last_hs = torch.nan_to_num(last_hs, nan=0.0, posinf=0.0, neginf=0.0)

    return seq_len, aux_hs, last_hs, ids_t, mask_t


def send_skip(rank: int, device: torch.device):
    """Send skip sentinel (seq_len=-1) to a consumer rank."""
    dist.send(torch.tensor([-1], dtype=torch.int64, device=device), dst=rank)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    master_addr = os.environ.get("MASTER_ADDR", "10.83.115.10")

    # 1. Redirect MASTER_PORT before vLLM spawn
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

    # 2. Load K2.5
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
        gpu_memory_utilization=0.75,   # slightly lower for batch=8 KV cache headroom
        tensor_parallel_size=8,
        enforce_eager=True,
    )
    log.info(f"K2.5 loaded in {time.time() - t_load:.1f}s")

    # 3. TCP sync → signal all CONSUMER_DDP_SIZE consumers
    tcp_signal_consumers(SYNC_PORT, CONSUMER_DDP_SIZE)

    # 4. Init P2P dist group (world_size = 1 + CONSUMER_DDP_SIZE)
    from datetime import timedelta
    world_size = CONSUMER_DDP_SIZE + 1
    log.info(f"Init P2P dist group: rank=0 world={world_size} addr={master_addr}:{P2P_PORT}")
    dist.init_process_group(
        backend="nccl", rank=0, world_size=world_size,
        init_method=f"tcp://{master_addr}:{P2P_PORT}",
        timeout=timedelta(hours=2),
    )
    log.info("P2P dist group initialized")

    # 5. Warmup with each consumer rank
    warmup = torch.zeros(1, dtype=torch.float32, device=device)
    for i in range(1, world_size):
        dist.send(warmup, dst=i)
        dist.recv(warmup, src=i)
    log.info("P2P NCCL warm-up done (all consumer ranks)")

    # 6. Dataset
    pt_files = sorted(
        Path(DATA_DIR).glob("data_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    log.info(f"Dataset: {len(pt_files)} files  epochs={NUM_EPOCHS}  batch={BATCH_SIZE}")
    if not pt_files:
        log.error(f"No .pt files in {DATA_DIR}!")
        dist.destroy_process_group()
        return

    loss_recv = torch.zeros(1, dtype=torch.float32, device=device)
    meta_send = torch.zeros(1, dtype=torch.int64, device=device)

    global_round = 0  # counts batches of BATCH_SIZE
    latencies = []

    # 7. Training loop
    for epoch in range(NUM_EPOCHS):
        log.info(f"=== Epoch {epoch}/{NUM_EPOCHS} ({len(pt_files)} files, "
                 f"{len(pt_files) // BATCH_SIZE} full batches) ===")

        for batch_start in range(0, len(pt_files) - BATCH_SIZE + 1, BATCH_SIZE):
            t0 = time.perf_counter()
            batch_files = pt_files[batch_start:batch_start + BATCH_SIZE]

            # Load all BATCH_SIZE files
            batch_raw = []
            for f in batch_files:
                batch_raw.append(torch.load(str(f), map_location="cpu", weights_only=True))

            # Generate hidden states for full batch
            all_input_ids = [d["input_ids"].tolist() for d in batch_raw]
            results = generator.generate(all_input_ids)
            gen_t = time.perf_counter() - t0

            # Prepare all consumer payloads first (all-or-nothing: if any rank bad, skip ALL)
            skip_batch = False
            payloads = []  # (consumer_rank, seq_len, aux_hs, last_hs, ids_t, mask_t)
            for i, (result, raw, f) in enumerate(zip(results, batch_raw, batch_files)):
                consumer_rank = i + 1
                result_len = len(result["input_ids"])
                disk_len = len(raw["input_ids"])

                if abs(result_len - disk_len) > 2:
                    log.warning(
                        f"round {global_round} rank {consumer_rank}: "
                        f"length mismatch disk={disk_len} vllm={result_len}, skipping whole batch"
                    )
                    skip_batch = True
                    break

                try:
                    seq_len, aux_hs, last_hs, ids_t, mask_t = prepare_batch(
                        result, raw["loss_mask"], MAX_SEQ_LEN, device
                    )
                    payloads.append((consumer_rank, seq_len, aux_hs, last_hs, ids_t, mask_t))
                except Exception as e:
                    log.warning(f"round {global_round} rank {consumer_rank}: "
                                f"prepare_batch failed: {e}, skipping whole batch")
                    skip_batch = True
                    break

            if skip_batch:
                # Send skip sentinel to ALL consumer ranks so DDP stays in sync
                for r in range(1, CONSUMER_DDP_SIZE + 1):
                    send_skip(r, device)
                dist.recv(loss_recv, src=1)  # chief sends NaN loss on skip
                global_round += 1
                continue

            # Phase 1: Send ALL metas first so all consumer ranks can do skip_flag AllReduce
            # (consumer AllReduces between meta-recv and data-recv; without this split,
            #  producer sends data to rank 1 while rank 1 is in AllReduce waiting for
            #  ranks 2-8 who haven't gotten meta yet -> deadlock)
            for consumer_rank, seq_len, aux_hs, last_hs, ids_t, mask_t in payloads:
                meta_send[0] = seq_len
                dist.send(meta_send, dst=consumer_rank)

            # Phase 2: Send data tensors (all consumers now past skip_flag AllReduce)
            total_sent_bytes = 0
            for consumer_rank, seq_len, aux_hs, last_hs, ids_t, mask_t in payloads:
                dist.send(aux_hs,    dst=consumer_rank)   # [1, seq_len, 3H]
                dist.send(last_hs,   dst=consumer_rank)   # [1, seq_len, H]
                dist.send(ids_t,     dst=consumer_rank)
                dist.send(mask_t,    dst=consumer_rank)

                total_sent_bytes += (aux_hs.nbytes + last_hs.nbytes +
                                     ids_t.nbytes + mask_t.nbytes)

            send_t = time.perf_counter() - t0

            # Receive loss from consumer rank 1 (DDP rank 0)
            dist.recv(loss_recv, src=1)
            total_t = time.perf_counter() - t0

            if global_round >= WARMUP_STEPS:
                latencies.append(total_t)

            if global_round % LOG_INTERVAL == 0:
                log.info(
                    f"epoch={epoch} batch={batch_start//BATCH_SIZE:05d} "
                    f"round={global_round:06d}  "
                    f"gen={gen_t*1000:.0f}ms  "
                    f"send={( send_t - gen_t)*1000:.0f}ms  "
                    f"total={total_t*1000:.0f}ms  "
                    f"sent={total_sent_bytes//1024//1024}MB  "
                    f"consumer_loss={loss_recv.item():.4f}"
                )

            global_round += 1

        if latencies:
            avg = sum(latencies) / len(latencies)
            log.info(
                f"\n=== PRODUCER EPOCH {epoch} SUMMARY ({len(latencies)} rounds) ===\n"
                f"  avg total latency: {avg*1000:.0f} ms/round\n"
                f"  effective throughput: {BATCH_SIZE / avg:.1f} seq/s"
            )

    dist.destroy_process_group()
    log.info("DONE")


if __name__ == "__main__":
    main()
