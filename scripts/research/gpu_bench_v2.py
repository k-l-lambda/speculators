#!/usr/bin/env python3
"""H200 GPU performance benchmark v2 — improved per GPT-5.4 review.
Official H200 specs: BF16 Tensor Core (dense) = 989.4 TFLOPS, HBM3e = 4.8 TB/s.

Fixes from v1: preallocated output, CUDA events, multi-size GEMM, telemetry capture.
"""
import torch, time, json, os, subprocess


def get_gpu_telemetry():
    """Capture nvidia-smi telemetry for all GPUs."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,pstate,temperature.gpu,clocks.sm,clocks.max.sm,"
            "clocks.mem,power.draw,power.limit,clocks_throttle_reasons.sw_power_cap,"
            "clocks_throttle_reasons.hw_thermal_slowdown,memory.used,memory.total",
            "--format=csv,noheader",
        ], text=True).strip()
        return out
    except Exception as e:
        return "Error: " + str(e)


def bench_matmul(device, dtype=torch.bfloat16, M=8192, N=8192, K=8192,
                 warmup_sec=3, bench_sec=5):
    """GEMM benchmark with preallocated output and CUDA events."""
    torch.cuda.set_device(device)
    with torch.no_grad():
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        C = torch.empty(M, N, dtype=dtype, device=device)

        # Warmup for fixed duration
        torch.cuda.synchronize(device)
        t_warmup = time.perf_counter()
        while time.perf_counter() - t_warmup < warmup_sec:
            torch.mm(A, B, out=C)
        torch.cuda.synchronize(device)

        # Calibrate iteration count
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        test_iters = 10
        for _ in range(test_iters):
            torch.mm(A, B, out=C)
        torch.cuda.synchronize(device)
        per_iter = (time.perf_counter() - t0) / test_iters
        iters = max(20, int(bench_sec / per_iter))

        # Timed region with CUDA events
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(iters):
            torch.mm(A, B, out=C)
        end_evt.record()
        torch.cuda.synchronize(device)

        elapsed_ms = start_evt.elapsed_time(end_evt)
        elapsed_s = elapsed_ms / 1e3
        flops = 2.0 * M * N * K * iters
        tflops = flops / elapsed_s / 1e12
        return tflops, elapsed_s / iters, iters


def bench_bandwidth(device, size_mb=1024):
    """Memory bandwidth: preallocated copy with CUDA events."""
    torch.cuda.set_device(device)
    with torch.no_grad():
        n = size_mb * 1024 * 1024 // 2
        A = torch.randn(n, dtype=torch.bfloat16, device=device)
        B = torch.empty_like(A)
        for _ in range(5):
            B.copy_(A)
        torch.cuda.synchronize(device)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        iters = 20
        start_evt.record()
        for _ in range(iters):
            B.copy_(A)
        end_evt.record()
        torch.cuda.synchronize(device)

        elapsed_s = start_evt.elapsed_time(end_evt) / 1e3
        bytes_total = 2 * n * 2 * iters  # read + write
        return bytes_total / elapsed_s / 1e9


def main():
    ngpu = torch.cuda.device_count()
    hostname = os.popen("hostname").read().strip()
    print("Host: %s, GPUs: %d" % (hostname, ngpu))
    print("PyTorch: %s, CUDA: %s" % (torch.__version__, torch.version.cuda))
    print("H200 official: BF16=989.4 TFLOPS, HBM3e BW=4.8 TB/s")
    print("TF32 matmul: %s" % torch.backends.cuda.matmul.allow_tf32)
    print()

    # Telemetry before
    print("=== GPU telemetry (before) ===")
    print(get_gpu_telemetry())
    print()

    # Multi-size GEMM
    sizes = [4096, 8192, 16384]
    results = []
    for i in range(ngpu):
        dev = "cuda:%d" % i
        name = torch.cuda.get_device_name(i)
        row = {"gpu": i, "name": name}
        parts = []
        for sz in sizes:
            tflops, lat, iters = bench_matmul(dev, M=sz, N=sz, K=sz)
            pct = tflops / 989.4 * 100
            key_t = "%d_tflops" % sz
            key_p = "%d_pct" % sz
            key_l = "%d_lat_ms" % sz
            row[key_t] = round(tflops, 1)
            row[key_p] = round(pct, 1)
            row[key_l] = round(lat * 1000, 2)
            parts.append("%d=%.0f(%.0f%%)" % (sz, tflops, pct))
        bw = bench_bandwidth(dev)
        row["bw_gbps"] = round(bw)
        print("GPU %d (%s): %s, BW=%d GB/s" % (i, name, ", ".join(parts), bw))
        results.append(row)

    # P2P bandwidth
    if ngpu > 1:
        print()
        size_mb = 256
        n = size_mb * 1024 * 1024 // 2
        with torch.no_grad():
            src = torch.randn(n, dtype=torch.bfloat16, device="cuda:0")
            dsts = [torch.empty(n, dtype=torch.bfloat16, device="cuda:%d" % j)
                    for j in range(1, ngpu)]
            for _ in range(5):
                for dst in dsts:
                    dst.copy_(src)
            for j in range(ngpu):
                torch.cuda.synchronize("cuda:%d" % j)

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            iters = 20
            start_evt.record()
            for _ in range(iters):
                for dst in dsts:
                    dst.copy_(src)
            end_evt.record()
            for j in range(ngpu):
                torch.cuda.synchronize("cuda:%d" % j)
            elapsed_s = start_evt.elapsed_time(end_evt) / 1e3
            total_bytes = size_mb * 1024 * 1024 * (ngpu - 1) * iters
            print("P2P BW (GPU0->all): %.1f GB/s" % (total_bytes / elapsed_s / 1e9))

    # Summary
    print()
    for sz in sizes:
        key_t = "%d_tflops" % sz
        avg = sum(r[key_t] for r in results) / len(results)
        print("AVG %dx%d: %.1f TFLOPS (%.1f%%)" % (sz, sz, avg, avg / 989.4 * 100))

    # Telemetry after
    print()
    print("=== GPU telemetry (after) ===")
    print(get_gpu_telemetry())

    with open("/tmp/gpu_bench_result.json", "w") as f:
        json.dump({"host": hostname, "results": results}, f)


if __name__ == "__main__":
    main()
