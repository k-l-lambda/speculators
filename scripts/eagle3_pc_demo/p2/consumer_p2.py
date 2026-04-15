"""
EAGLE3 Producer-Consumer Demo Phase 2 — Consumer (host-10-83-115-14)

Receives hidden states from producer via NCCL P2P.
Trains Eagle3HeadPOC (FC + MLP residual) with MSE loss vs target hidden state.
Sends loss scalar back to producer each step.

No speculators dependency — uses standard PyTorch only.

Sync protocol (Phase 2b fix — deferred NCCL P2P init):
  - Producer loads K2.5 (8-9 min), then opens TCP server on SYNC_PORT=29501
  - Consumer polls SYNC_PORT until "ready" arrives, then both init NCCL P2P
  - Avoids NCCL P2P idle timeout during long K2.5 loading window

Eagle3HeadPOC architecture:
  FC(3*H → H) + LayerNorm + SiLU-MLP(H → 2H → H) residual
  Loss: MSE(eagle3_output, last_hs_from_verifier)
  ~360M parameters for H=7168
"""

import os
import time
import socket
import logging
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [consumer] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---- Config ----
H         = 7168    # K2.5 hidden dim
N_AUX     = 3       # aux layer count (LAYER_IDS[0:3] = [2, 30, 58])
B, S      = 2, 128
LR        = 1e-4
NUM_STEPS = 10
SYNC_PORT = 29501   # TCP sync port: wait for producer's "ready" before init NCCL
P2P_PORT  = int(os.environ.get("P2P_NCCL_PORT", "29503"))  # fresh port for P2P NCCL group


class Eagle3HeadPOC(nn.Module):
    """
    Minimal Eagle3 head for Phase 2 POC.

    Input:  aux_hs [B, S, N_AUX*H]  (concat of 3 aux layer hidden states, bfloat16)
    Target: last_hs [B, S, H]        (verifier's last hidden state, bfloat16)
    Output: [B, S, H] bfloat16       (draft representation, MSE loss vs target)

    Architecture:
        FC(3H → H) → LayerNorm → SiLU-MLP residual(H → 2H → H)
    """

    def __init__(self, H: int = 7168, n_aux: int = 3):
        super().__init__()
        self.fc   = nn.Linear(n_aux * H, H, bias=False)
        self.norm = nn.LayerNorm(H)
        self.w1   = nn.Linear(H, H * 2, bias=False)
        self.w2   = nn.Linear(H * 2, H, bias=False)
        self.act  = nn.SiLU()

    def forward(self, aux_hs: torch.Tensor) -> torch.Tensor:
        # aux_hs: [B, S, n_aux*H] float32 (caller casts from bfloat16)
        x = self.fc(aux_hs)                    # [B, S, H] float32
        x = self.norm(x)                       # LayerNorm in fp32 (all params float32)
        residual = x
        x = self.w2(self.act(self.w1(x)))      # [B, S, H] MLP
        return x + residual                    # [B, S, H] float32


def wait_for_producer_ready(producer_addr: str, sync_port: int = SYNC_PORT, poll_interval: float = 5.0):
    """
    Poll producer's TCP sync port until "ready" signal received.

    Producer opens this server only after K2.5 has finished loading (~8-9 min).
    Consumer polls until it can connect, then both call dist.init_process_group()
    at nearly the same time — avoiding NCCL P2P idle timeout.
    """
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


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    master_addr = os.environ.get("MASTER_ADDR", "10.83.115.10")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))

    # 1. Init Eagle3 head + optimizer early (while producer loads K2.5)
    model = Eagle3HeadPOC(H=H, n_aux=N_AUX).to(device=device)  # float32 — avoids LayerNorm dtype mismatch with bfloat16 model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Eagle3HeadPOC: {n_params:,} parameters  device={device}  dtype=float32")

    # 2. Wait for producer to finish loading K2.5 via TCP sync.
    #    No NCCL P2P calls during this phase — avoids NCCL QP idle timeout.
    wait_for_producer_ready(master_addr, sync_port=SYNC_PORT)

    # 3. Init P2P dist group — both sides ready now, init completes quickly
    from datetime import timedelta
    log.info(f"Init P2P dist group: rank=1 world_size=2 addr={master_addr}:{P2P_PORT}")
    dist.init_process_group(
        backend="nccl",
        rank=1,
        world_size=2,
        init_method=f"tcp://{master_addr}:{P2P_PORT}",  # P2P_PORT=29503 avoids torchrun rendezvous on 29500
        timeout=timedelta(minutes=5),  # both sides ready, fast init
    )
    assert dist.get_rank() == 1 and dist.get_world_size() == 2
    log.info("P2P dist group initialized")

    # 3b. Warm-up: force NCCL 2-rank P2P communicator creation
    warmup = torch.zeros(1, dtype=torch.float32, device=device)
    dist.recv(warmup, src=0)
    dist.send(warmup, dst=0)
    log.info("P2P NCCL communicator warm-up done")

    # 4. Pre-allocate receive buffers (same shape every step)
    aux_buf  = torch.empty(B, S, N_AUX * H, dtype=torch.bfloat16, device=device)
    last_buf = torch.empty(B, S, H,         dtype=torch.bfloat16, device=device)
    loss_send = torch.zeros(1, dtype=torch.float32, device=device)
    transfer_bytes = aux_buf.nbytes + last_buf.nbytes

    log.info(
        f"Buffers: aux_hs={aux_buf.nbytes/1e6:.1f}MB  "
        f"last_hs={last_buf.nbytes/1e6:.1f}MB  "
        f"total={transfer_bytes/1e6:.1f}MB/step"
    )
    log.info(f"Starting main loop (B={B} S={S} H={H}) ...")

    losses = []
    try:
        for step in range(NUM_STEPS):
            t0 = time.perf_counter()

            # Receive hidden states from producer (rank 0)
            dist.recv(aux_buf, src=0)
            dist.recv(last_buf, src=0)
            recv_t = time.perf_counter() - t0

            # Forward + loss + backward (cast bfloat16 network inputs to float32)
            optimizer.zero_grad()
            draft_hs = model(aux_buf.float())                           # [B, S, H] float32
            loss = F.mse_loss(draft_hs, last_buf.float())               # MSE vs verifier last hs
            loss.backward()
            optimizer.step()
            train_t = time.perf_counter() - t0

            loss_val = loss.item()
            losses.append(loss_val)

            # Send loss scalar back to producer
            loss_send[0] = loss_val
            dist.send(loss_send, dst=0)

            log.info(
                f"step {step:02d}: recv={recv_t*1000:.1f}ms  "
                f"train={train_t*1000:.1f}ms  "
                f"bw={transfer_bytes/recv_t/1e9:.2f}GB/s  "
                f"loss={loss_val:.4f}"
            )
    except Exception as e:
        log.error(f"EXCEPTION in consumer loop: {type(e).__name__}: {e}")
        log.error(traceback.format_exc())
        raise

    if losses:
        log.info(f"\n=== CONSUMER SUMMARY ({NUM_STEPS} steps) ===")
        log.info(
            f"  loss: {losses[0]:.4f} → {losses[-1]:.4f} "
            f"({'↓ decreasing ✓' if losses[-1] < losses[0] else '↑ not decreasing'})"
        )
        log.info(f"  Eagle3HeadPOC params: {n_params:,}")

    dist.destroy_process_group()
    log.info("DONE")


if __name__ == "__main__":
    main()
