"""
EAGLE3 Producer-Consumer Demo Phase 2 — Consumer (host-10-83-115-14)

Receives hidden states from producer via NCCL P2P.
Trains Eagle3HeadPOC (FC + MLP residual) with MSE loss vs target hidden state.
Sends loss scalar back to producer each step.

No speculators dependency — uses standard PyTorch only.

Eagle3HeadPOC architecture:
  FC(3*H → H) + LayerNorm + SiLU-MLP(H → 2H → H) residual
  Loss: MSE(eagle3_output, last_hs_from_verifier)
  ~126M parameters for H=7168
"""

import os
import time
import logging

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
        # aux_hs: [B, S, n_aux*H] bfloat16
        x = self.fc(aux_hs)                    # [B, S, H] bfloat16
        x = self.norm(x.float()).bfloat16()    # LayerNorm in fp32 for stability
        residual = x
        x = self.w2(self.act(self.w1(x)))      # [B, S, H] MLP
        return x + residual                    # [B, S, H]


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 1. Init P2P dist group
    from datetime import timedelta
    master_addr = os.environ.get("MASTER_ADDR", "10.83.115.10")
    master_port = os.environ.get("MASTER_PORT", "29500")
    log.info(f"Init P2P dist group: rank=1 world_size=2 addr={master_addr}:{master_port}")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    assert dist.get_rank() == 1 and dist.get_world_size() == 2
    log.info("P2P dist group initialized")

    # 1b. P2P warm-up: force NCCL 2-rank communicator creation while both sides are alive.
    #     Without this, lazy communicator setup on first dist.recv() can time out if
    #     rank 0 (producer) is busy loading K2.5 and never triggers its side.
    warmup = torch.zeros(1, dtype=torch.float32, device=device)
    dist.recv(warmup, src=0)
    dist.send(warmup, dst=0)
    log.info("P2P NCCL communicator warm-up done — waiting for producer to load K2.5 ...")

    # 2. Init Eagle3 head + optimizer
    model = Eagle3HeadPOC(H=H, n_aux=N_AUX).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Eagle3HeadPOC: {n_params:,} parameters  device={device}")

    # 3. Pre-allocate receive buffers (same shape every step)
    aux_buf  = torch.empty(B, S, N_AUX * H, dtype=torch.bfloat16, device=device)
    last_buf = torch.empty(B, S, H,         dtype=torch.bfloat16, device=device)
    loss_send = torch.zeros(1, dtype=torch.float32, device=device)
    transfer_bytes = aux_buf.nbytes + last_buf.nbytes

    log.info(
        f"Buffers: aux_hs={aux_buf.nbytes/1e6:.1f}MB  "
        f"last_hs={last_buf.nbytes/1e6:.1f}MB  "
        f"total={transfer_bytes/1e6:.1f}MB/step"
    )
    log.info(f"Waiting for producer (B={B} S={S} H={H}) ...")

    losses = []
    try:
        for step in range(NUM_STEPS):
            t0 = time.perf_counter()

            # Receive hidden states from producer (rank 0)
            dist.recv(aux_buf, src=0)
            dist.recv(last_buf, src=0)
            recv_t = time.perf_counter() - t0

            # Forward + loss + backward
            optimizer.zero_grad()
            draft_hs = model(aux_buf)                                   # [B, S, H] bf16
            loss = F.mse_loss(draft_hs.float(), last_buf.float())       # MSE vs verifier last hs
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
        avg_bw = transfer_bytes / (sum(step_t for step_t in []) + 1e-9)
        log.info(f"  Eagle3HeadPOC params: {n_params:,}")

    dist.destroy_process_group()
    log.info("DONE")


if __name__ == "__main__":
    main()
