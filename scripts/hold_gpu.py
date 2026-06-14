#!/usr/bin/env python
"""Temporarily keep GPU(s) busy so a node stays reserved while you debug.

Allocates a chunk of GPU memory and runs continuous matmuls to hold both
memory and compute utilization high. Ctrl-C (or the --minutes timeout) frees
everything and exits cleanly.

Examples:
    python hold_gpu.py                      # all visible GPUs, ~90% mem, until Ctrl-C
    python hold_gpu.py --gpus 0,3           # only cuda:0 and cuda:3
    python hold_gpu.py --mem-frac 0.5       # leave half the memory free
    python hold_gpu.py --minutes 30         # auto-release after 30 minutes
    python hold_gpu.py --util low           # keep mem held but light on compute
"""
import argparse
import time

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default=None,
                   help="comma-separated GPU indices, e.g. '0,3'. Default: all visible.")
    p.add_argument("--mem-frac", type=float, default=0.9,
                   help="fraction of each GPU's free memory to grab (default 0.9).")
    p.add_argument("--util", choices=["high", "low"], default="high",
                   help="'high' = constant matmuls (~100%% util), "
                        "'low' = mostly idle, just hold memory.")
    p.add_argument("--minutes", type=float, default=0,
                   help="auto-release after this many minutes (0 = run until Ctrl-C).")
    p.add_argument("--matmul-size", type=int, default=8192,
                   help="square matrix size for the busy-loop matmul (default 8192).")
    return p.parse_args()


def grab_memory(device, mem_frac):
    """Allocate a big tensor occupying ~mem_frac of free memory on `device`."""
    free, _ = torch.cuda.mem_get_info(device)
    # leave headroom for the matmul workspace; reserve mem_frac of *free* mem.
    n_bytes = int(free * mem_frac)
    n_floats = max(n_bytes // 4, 1)  # float32 = 4 bytes
    try:
        blob = torch.empty(n_floats, dtype=torch.float32, device=device)
        blob.fill_(0)
        gb = blob.numel() * 4 / 1024 ** 3
        print(f"[cuda:{device}] held {gb:.1f} GiB")
        return blob
    except RuntimeError as e:
        print(f"[cuda:{device}] could not allocate {n_bytes/1024**3:.1f} GiB: {e}")
        return None


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device available.")

    if args.gpus:
        devices = [int(x) for x in args.gpus.split(",")]
    else:
        devices = list(range(torch.cuda.device_count()))
    print(f"Holding GPUs: {devices}  (util={args.util}, mem_frac={args.mem_frac})")

    blobs = {}
    work = {}
    for d in devices:
        torch.cuda.set_device(d)
        blobs[d] = grab_memory(d, args.mem_frac)
        if args.util == "high":
            n = args.matmul_size
            work[d] = (torch.randn(n, n, device=d), torch.randn(n, n, device=d))

    deadline = time.time() + args.minutes * 60 if args.minutes > 0 else None
    print("Running. Press Ctrl-C to release." if deadline is None
          else f"Running for {args.minutes} min. Ctrl-C to release early.")

    try:
        while True:
            if deadline and time.time() >= deadline:
                print("Timeout reached, releasing.")
                break
            if args.util == "high":
                for d in devices:
                    torch.cuda.set_device(d)
                    a, b = work[d]
                    c = a @ b
                    a.copy_(c)  # keep the result alive, prevent dead-code elim
                torch.cuda.synchronize()
            else:
                time.sleep(5)
    except KeyboardInterrupt:
        print("\nInterrupted, releasing GPU memory.")
    finally:
        blobs.clear()
        work.clear()
        for d in devices:
            torch.cuda.set_device(d)
            torch.cuda.empty_cache()
        print("Done.")


if __name__ == "__main__":
    main()
