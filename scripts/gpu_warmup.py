#!/usr/bin/env python
"""GPU warmup / soak loop: keep device(s) warm with a steady matmul workload.

Allocates a chunk of GPU memory and runs a duty-cycled matmul loop to keep the
device(s) warm and resident. Ctrl-C (or the --minutes timeout) frees everything
and exits cleanly.

Examples:
    python gpu_warmup.py                     # all visible GPUs, ~60% util, until Ctrl-C
    python gpu_warmup.py --gpus 0,3          # only cuda:0 and cuda:3
    python gpu_warmup.py --mem-frac 0.5      # leave half the memory free
    python gpu_warmup.py --minutes 30        # auto-release after 30 minutes
    python gpu_warmup.py --util high         # constant matmuls (~100% util)
    python gpu_warmup.py --util low          # keep mem resident but light on compute
"""
import argparse
import time

import torch

# target compute duty cycle per util level (fraction of wall-clock spent in matmul)
DUTY = {"high": 1.0, "medium": 0.6, "low": 0.0}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default=None,
                   help="comma-separated GPU indices, e.g. '0,3'. Default: all visible.")
    p.add_argument("--mem-frac", type=float, default=0.9,
                   help="fraction of each GPU's free memory to grab (default 0.9).")
    p.add_argument("--util", choices=["high", "medium", "low"], default="medium",
                   help="'high' = constant matmuls (~100%% util), "
                        "'medium' = duty-cycled to ~60%% util (default), "
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
        print(f"[cuda:{device}] reserved {gb:.1f} GiB")
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
    duty = DUTY[args.util]
    print(f"Warming GPUs: {devices}  (util={args.util}, mem_frac={args.mem_frac})")

    blobs = {}
    work = {}
    for d in devices:
        torch.cuda.set_device(d)
        blobs[d] = grab_memory(d, args.mem_frac)
        if duty > 0:
            # allocate the matmul pair out of the headroom; shrink on OOM so a
            # tight GPU still holds memory (blob) even if it can't run matmuls.
            n = args.matmul_size
            while n >= 512:
                try:
                    work[d] = (torch.randn(n, n, device=d), torch.randn(n, n, device=d))
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    n //= 2
            if d not in work:
                print(f"[cuda:{d}] no room for matmul workspace; holding memory only.")

    deadline = time.time() + args.minutes * 60 if args.minutes > 0 else None
    print("Running. Press Ctrl-C to release." if deadline is None
          else f"Running for {args.minutes} min. Ctrl-C to release early.")

    try:
        while True:
            if deadline and time.time() >= deadline:
                print("Timeout reached, releasing.")
                break
            if duty <= 0 or not work:
                time.sleep(5)
                continue
            # one matmul burst across all devices, timed so we can throttle to `duty`
            t0 = time.time()
            for d in devices:
                if d not in work:
                    continue
                torch.cuda.set_device(d)
                a, b = work[d]
                try:
                    c = a @ b
                    a.copy_(c)  # keep the result alive, prevent dead-code elim
                except torch.cuda.OutOfMemoryError:
                    # never crash: drop this device to hold-only, keep its blob alive
                    print(f"[cuda:{d}] OOM during matmul; dropping to hold-only.")
                    work.pop(d, None)
                    torch.cuda.empty_cache()
            torch.cuda.synchronize()
            busy = time.time() - t0
            # sleep so that busy / (busy + sleep) ~= duty  (skip when duty == 1.0)
            if duty < 1.0:
                time.sleep(busy * (1.0 / duty - 1.0))
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
