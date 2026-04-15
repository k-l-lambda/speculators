"""
EAGLE3 Producer-Consumer Demo Phase 2 — Producer (youyun.37 / host-10-83-115-10)

Loads K2.5 (TP=8) via VllmHiddenStatesGenerator.
Sends hidden states [B, S, 3H] + [B, S, H] to consumer via NCCL P2P.
Receives MSE loss scalar back each step.

NCCL port separation:
  - P2P group (rank 0 ↔ rank 1):  MASTER_PORT=29500  (set by torchrun)
  - vLLM TP-group (rank 0..7):     MASTER_PORT=29502  (overridden before vLLM init)

Sync protocol (Phase 2b fix — deferred NCCL P2P init):
  - Producer loads K2.5 (8-9 min), then opens TCP server on SYNC_PORT=29501
  - Consumer polls SYNC_PORT until "ready" arrives, then both init NCCL P2P
  - Avoids NCCL P2P idle timeout during long K2.5 loading window

Tensor sizes (B=2, S=128, H=7168):
  aux_hs   [2, 128, 21504] bf16  = 11 MB
  last_hs  [2, 128,  7168] bf16  =  3.7 MB
  Total: ~14.7 MB/step
"""

import os
import sys
import time
import socket
import logging

import torch
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [producer] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---- Config ----
MODEL_PATH  = "/data/models/Kimi-K2.5"
LAYER_IDS   = [2, 30, 58, 60]   # K2.5 Eagle3 v2 layers: 3 aux + 1 last
B, S        = 2, 128             # batch=2, seq=128
NUM_STEPS   = 10
WARMUP      = 2
VLLM_PORT   = int(os.environ.get("VLLM_MASTER_PORT", "29502"))
SYNC_PORT   = 29501              # TCP sync port: producer signals consumer when K2.5 ready
P2P_PORT    = int(os.environ.get("P2P_NCCL_PORT", "29503"))  # fresh port for P2P NCCL group (avoids torchrun rendezvous on 29500)


def tcp_signal_consumer(sync_port: int = SYNC_PORT):
    """
    Open a TCP server, wait for consumer to connect, reply 'ready', then close.

    This simple handshake lets producer signal consumer that K2.5 has loaded,
    so both sides can call dist.init_process_group() at nearly the same time.
    """
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


def pack_hidden_states(results: list[dict], device: torch.device):
    """
    Stack per-sample hidden states from generator.generate() into batched tensors.

    Args:
        results: list of dicts from VllmHiddenStatesGenerator.generate()
                 each dict has "hidden_states": list[Tensor[S, H]] (CPU, one per layer)
        device: target CUDA device

    Returns:
        aux_hs:  [B, S, 3*H] bfloat16  (concat of layers [2,30,58])
        last_hs: [B, S, H]   bfloat16  (layer 60, post-norm)
    """
    n_aux = len(LAYER_IDS) - 1   # 3 aux layers
    aux_list, last_list = [], []

    for r in results:
        hs = r["hidden_states"]   # list of [S, H] CPU tensors
        aux = torch.cat([hs[j] for j in range(n_aux)], dim=-1)  # [S, 3H]
        aux_list.append(aux.to(device=device, dtype=torch.bfloat16))
        last_list.append(hs[n_aux].to(device=device, dtype=torch.bfloat16))  # [S, H]

    return (
        torch.stack(aux_list),    # [B, S, 3H]
        torch.stack(last_list),   # [B, S, H]
    )


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    master_addr = os.environ.get("MASTER_ADDR", "10.83.115.10")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))

    # 1. Clear ALL torchrun / torch.distributed env vars before spawning vLLM workers.
    #    torchrun injects RANK=0, WORLD_SIZE=2, etc. into the producer process.
    #    vLLM workers inherit these via multiprocessing.spawn → TP group mis-configures
    #    (TCPStore server waits for 2 workers instead of 8 → deadlock).
    #    Also redirect MASTER_PORT to the vLLM TP port (29502) before clearing.
    os.environ["MASTER_PORT"] = str(VLLM_PORT)
    log.info(f"Redirected MASTER_PORT to {VLLM_PORT} for vLLM TP workers")

    _clear_before_vllm = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "GROUP_RANK", "GROUP_WORLD_SIZE",
        "ROLE_RANK", "ROLE_WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE",
        "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
    ]
    cleared = {k: os.environ.pop(k) for k in _clear_before_vllm if k in os.environ}
    if cleared:
        log.info(f"Cleared {len(cleared)} env vars before vLLM spawn: {list(cleared.keys())}")

    # 2. Load K2.5 via VllmHiddenStatesGenerator (spawns 8 TP worker subprocesses)
    sys.path.insert(0, "/workspace/speculators/src")
    from speculators.data_generation.vllm_hidden_states_generator import (
        VllmHiddenStatesGenerator,
    )

    log.info(f"Loading K2.5 via VllmHiddenStatesGenerator: TP=8, layers={LAYER_IDS}")
    t_load = time.time()
    generator = VllmHiddenStatesGenerator(
        model_path=MODEL_PATH,
        layer_ids=LAYER_IDS,
        max_model_len=512,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=8,
        enforce_eager=True,
    )
    log.info(f"K2.5 loaded in {time.time() - t_load:.1f}s")

    # 3. TCP sync: signal consumer that K2.5 is loaded, then BOTH init NCCL P2P.
    #    This avoids NCCL P2P idle timeout during the 8+ min K2.5 loading window.
    #    (Previous approach: init NCCL P2P before K2.5 load → NCCL QP idle timeout.)
    tcp_signal_consumer()

    from datetime import timedelta
    log.info(f"Init P2P dist group: rank=0 world_size=2 addr={master_addr}:{P2P_PORT}")
    dist.init_process_group(
        backend="nccl",
        rank=0,
        world_size=2,
        init_method=f"tcp://{master_addr}:{P2P_PORT}",  # P2P_PORT=29503 avoids torchrun rendezvous on 29500
        timeout=timedelta(minutes=5),  # both sides ready now, fast init
    )
    assert dist.get_rank() == 0 and dist.get_world_size() == 2
    log.info("P2P dist group initialized")

    # 3b. Warm-up: force NCCL 2-rank P2P communicator creation before first send
    warmup = torch.zeros(1, dtype=torch.float32, device=device)
    dist.send(warmup, dst=1)
    dist.recv(warmup, src=1)
    log.info("P2P NCCL communicator warm-up done")

    # 4. Pre-allocate buffers (same shape every step)
    H = 7168
    aux_buf   = torch.empty(B, S, 3 * H, dtype=torch.bfloat16, device=device)
    last_buf  = torch.empty(B, S, H,     dtype=torch.bfloat16, device=device)
    loss_recv = torch.zeros(1, dtype=torch.float32, device=device)
    transfer_bytes = aux_buf.nbytes + last_buf.nbytes

    log.info(
        f"Transfer per step: aux_hs={aux_buf.nbytes/1e6:.1f}MB "
        f"last_hs={last_buf.nbytes/1e6:.1f}MB "
        f"total={transfer_bytes/1e6:.1f}MB"
    )

    # 5. Fixed synthetic batch (same tokens every step for reproducibility)
    token_ids = [[i % 9000 + 1000 for i in range(S)] for _ in range(B)]
    log.info(f"Starting loop: B={B} S={S} steps={NUM_STEPS}")

    latencies = []
    for step in range(NUM_STEPS):
        t0 = time.perf_counter()

        # Generate hidden states from K2.5 (this is the slow part)
        results = generator.generate(token_ids)
        gen_t = time.perf_counter() - t0

        # Pack into [B, S, 3H] and [B, S, H]
        aux_hs, last_hs = pack_hidden_states(results, device)
        aux_buf[:] = aux_hs
        last_buf[:] = last_hs

        # Send to consumer (rank 1)
        dist.send(aux_buf, dst=1)
        dist.send(last_buf, dst=1)
        send_t = time.perf_counter() - t0

        # Recv loss scalar from consumer
        dist.recv(loss_recv, src=1)
        total_t = time.perf_counter() - t0

        if step >= WARMUP:
            latencies.append(total_t)

        log.info(
            f"step {step:02d}: gen={gen_t*1000:.0f}ms  "
            f"send={send_t*1000:.1f}ms  "
            f"total={total_t*1000:.0f}ms  "
            f"consumer_loss={loss_recv.item():.4f}"
        )

    if latencies:
        avg = sum(latencies) / len(latencies)
        log.info(f"\n=== PRODUCER SUMMARY ({len(latencies)} measured steps) ===")
        log.info(f"  avg total latency:  {avg*1000:.0f} ms/step")
        log.info(f"  transfer per step:  {transfer_bytes/1e6:.1f} MB")

    dist.destroy_process_group()
    log.info("DONE")


if __name__ == "__main__":
    main()
