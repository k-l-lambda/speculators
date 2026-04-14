"""
EAGLE3 Producer-Consumer POC — Phase 1
Cross-node NCCL hidden state transfer benchmark

Topology:
  rank 0 (producer, youyun.37 / host-10-83-115-10): generate random [B,S,3H] tensors, send via NCCL P2P
  rank 1 (consumer, host-10-83-115-14): recv tensors, run fake FC+backward, log metrics

Tensor sizes (K2.5 dims, B=4, S=512):
  hidden_states: [4, 512, 21504] bf16 = 88.1 MB   (3 * H=7168)
  topk_vals:     [4, 512, 100]   bf16 = 0.4 MB
  topk_idx:      [4, 512, 100]   int32 = 0.4 MB
  Total per step: ~89 MB

Launch (handled by LWS yaml via torchrun):
  torchrun --nnodes=2 --nproc_per_node=1
           --node_rank=<0|1>
           --master_addr=<youyun.37-IP>
           --master_port=29500
           demo_p1_transfer.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 2))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[rank {rank}] init_process_group start  world_size={world_size} device={device}", flush=True)
    dist.init_process_group(backend="nccl")
    print(f"[rank {rank}] init_process_group DONE", flush=True)

    # ---- Config ----
    B    = 4       # batch size
    S    = 512     # sequence length
    H    = 7168    # K2.5 hidden dim (4096 for smaller proxy models)
    TOPK = 100     # top-K logits (avoids full 163840-vocab transfer)
    NUM_STEPS = 25
    WARMUP    = 5

    # ---- Tensors ----
    hidden    = torch.randn(B, S, 3 * H, dtype=torch.bfloat16, device=device)
    topk_vals = torch.randn(B, S, TOPK,  dtype=torch.bfloat16, device=device)
    topk_idx  = torch.randint(0, 32000, (B, S, TOPK), dtype=torch.int32, device=device)

    total_bytes = hidden.nbytes + topk_vals.nbytes + topk_idx.nbytes

    # ---- Consumer-side model (tiny FC simulating Eagle3 head) ----
    if rank == 1:
        fc        = nn.Linear(3 * H, H, bias=False, dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(fc.parameters(), lr=1e-4)

    if rank == 0:
        print(
            f"[producer] tensor sizes per step:\n"
            f"  hidden_states  [{B},{S},{3*H}] bf16 = {hidden.nbytes/1e6:.1f} MB\n"
            f"  topk_vals      [{B},{S},{TOPK}] bf16 = {topk_vals.nbytes/1e6:.1f} MB\n"
            f"  topk_idx       [{B},{S},{TOPK}] i32  = {topk_idx.nbytes/1e6:.1f} MB\n"
            f"  TOTAL: {total_bytes/1e6:.1f} MB",
            flush=True,
        )

    latencies = []

    for step in range(NUM_STEPS):
        dist.barrier()
        t0 = time.perf_counter()

        if rank == 0:
            # ---- Producer ----
            dist.send(hidden,    dst=1)
            dist.send(topk_vals, dst=1)
            dist.send(topk_idx,  dst=1)
            elapsed = time.perf_counter() - t0
            if step >= WARMUP:
                latencies.append(elapsed)
                bw = total_bytes / elapsed / 1e9
                print(
                    f"[producer] step {step:02d}: send {elapsed*1000:.1f} ms  bw={bw:.2f} GB/s",
                    flush=True,
                )

        else:
            # ---- Consumer ----
            recv_hidden    = torch.empty_like(hidden)
            recv_topk_vals = torch.empty_like(topk_vals)
            recv_topk_idx  = torch.empty_like(topk_idx)

            dist.recv(recv_hidden,    src=0)
            dist.recv(recv_topk_vals, src=0)
            dist.recv(recv_topk_idx,  src=0)
            recv_elapsed = time.perf_counter() - t0

            # Simulate Eagle3 draft head training step
            optimizer.zero_grad()
            out  = fc(recv_hidden.float())   # [B, S, H]
            loss = out.mean()
            loss.backward()
            optimizer.step()
            train_elapsed = time.perf_counter() - t0

            if step >= WARMUP:
                latencies.append(recv_elapsed)
                bw = total_bytes / recv_elapsed / 1e9
                print(
                    f"[consumer] step {step:02d}: recv={recv_elapsed*1000:.1f}ms"
                    f"  train={train_elapsed*1000:.1f}ms"
                    f"  bw={bw:.2f} GB/s  loss={loss.item():.6f}",
                    flush=True,
                )

    dist.barrier()

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        avg_bw  = total_bytes / avg_lat / 1e9
        p50     = sorted(latencies)[len(latencies) // 2]
        p95     = sorted(latencies)[int(len(latencies) * 0.95)]
        print(
            f"\n[rank {rank}] === SUMMARY ({len(latencies)} measured steps) ===\n"
            f"  avg latency : {avg_lat*1000:.1f} ms\n"
            f"  p50 latency : {p50*1000:.1f} ms\n"
            f"  p95 latency : {p95*1000:.1f} ms\n"
            f"  avg bw      : {avg_bw:.2f} GB/s\n"
            f"  tensor size : {total_bytes/1e6:.1f} MB\n",
            flush=True,
        )

    dist.destroy_process_group()
    print(f"[rank {rank}] DONE.", flush=True)


if __name__ == "__main__":
    main()
