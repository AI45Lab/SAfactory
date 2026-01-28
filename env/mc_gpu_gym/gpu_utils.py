import argparse
import os
import glob
import subprocess
import sys


# ---------------- GPU 资源获取（pynvml） ---------------- #

def get_gpus_by_pynvml():
    try:
        import pynvml
    except ImportError:
        return None

    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()

    gpus = []
    for i in range(count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)

        gpus.append({
            "index": i,
            "mem_used": mem.used,
            "mem_total": mem.total,
            "gpu_util": util.gpu,
        })

    return gpus


# ---------------- GPU 资源获取（nvidia-smi 兜底） ---------------- #

def get_gpus_by_smi():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    gpus = []
    for i, line in enumerate(out.decode().strip().splitlines()):
        used, total, util = map(int, line.split(","))
        gpus.append({
            "index": i,
            "mem_used": used * 1024 * 1024,
            "mem_total": total * 1024 * 1024,
            "gpu_util": util,
        })

    return gpus


# ---------------- 选择最空闲 GPU ---------------- #

def gpu_score(gpu):
    mem_ratio = gpu["mem_used"] / gpu["mem_total"]
    util_ratio = gpu["gpu_util"] / 100.0
    return 0.7 * mem_ratio + 0.3 * util_ratio


def select_least_used_gpu():
    gpus = get_gpus_by_pynvml()
    print(gpus)
    if gpus is None:
        gpus = get_gpus_by_smi()
    if not gpus:
        return None
    return min(gpus, key=gpu_score)["index"]


# ---------------- /dev/dri 映射 ---------------- #

def select_dri_device(cuda_index):
    if cuda_index is None:
        return "cpu"

    candidate = f"/dev/dri/card{cuda_index}"
    if os.path.exists(candidate):
        return candidate

    cards = sorted(glob.glob("/dev/dri/card*"))
    if cards:
        return cards[0]

    return "cpu"


# ---------------- main ---------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    cuda_index = select_least_used_gpu()
    device = select_dri_device(cuda_index)

    print(device)
