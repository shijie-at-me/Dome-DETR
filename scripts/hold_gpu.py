"""
Hold a card's spare memory while one of our runs trains on it, without touching its compute: one
allocation, then sleep, and exit as soon as the run's process is gone so the next run in the queue
(an L, which needs the memory) is not blocked.

    CUDA_VISIBLE_DEVICES=0 python scripts/hold_gpu.py <gigabytes> <trainer pid>

The shared server's convention is that a card whose memory is full is left alone; the S runs use
under half of an L40, so the rest is held here.
"""

import os
import sys
import time

import torch


def main():
    gigabytes, pid = float(sys.argv[1]), int(sys.argv[2])
    block = torch.empty(int(gigabytes * 2**30), dtype=torch.uint8, device="cuda")  # noqa: F841
    torch.cuda.synchronize()
    print(f"holding {gigabytes} GB on cuda:{torch.cuda.current_device()} while pid {pid} runs", flush=True)
    while os.path.exists(f"/proc/{pid}"):
        time.sleep(30)
    print("run finished, releasing", flush=True)


if __name__ == "__main__":
    main()
