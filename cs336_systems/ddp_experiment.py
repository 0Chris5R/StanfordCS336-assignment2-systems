
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import timeit


# gloo for cpu development, nccl for GPU

# always do at least 5 warmup iters for benchmarking and call torch.cuda.synchronize()

# we can set the device with torch.cuda.set_device(rank) or  device = f"cuda:{rank}")
# and then tensor.to(f"cuda:{rank}" or tensor.to(device)


# common operations include dist.all_gather and dist.all_reduce

# Vary:  Backend + device type: Gloo + CPU, NCCL + GPU.
# all-reduce data size: float32 data tensors ranging over 1MB, 10MB, 100MB, 1GB.

# Number of processes: 2, 4, or 6 processes.
# Resource requirements: Up to 6 GPUs. Each benchmarking run should take less than 5 minute


def setup(rank, world_size, device):
    if device.type == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def distributed_demo(rank, world_size, tensor_size, device):
    setup(rank, world_size, device)

    if device.type == "cuda":
        torch.cuda.set_device(rank)
        data_device = "cuda"

    else:
        data_device = device

    data = torch.randint(0, 10, (tensor_size//4,),
                         dtype=torch.float32, device=data_device)
    # print(f"rank {rank} data (before all-reduce): {data}")

    dist.all_reduce(data, async_op=False)

    # print(f"rank {rank} data (after all-reduce): {data}")


if __name__ == "__main__":

    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def device_synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    world_sizes = [2, 4, 6]

    tensor_sizes = [1024**2, 10 * 1024**2, 100 * 1024**2, 1024**3]

    for world_size, tensor_size in [(world_size, tensor_size) for world_size in world_sizes for tensor_size in tensor_sizes]:

        # warmup

        for i in range(5):

            mp.spawn(
                fn=distributed_demo,
                args=(world_size, tensor_size, device),
                nprocs=world_size,
                join=True,
            )

        device_synchronize()

        # benchmark

        times = np.empty(5)

        for i in range(5):

            start = timeit.default_timer()

            mp.spawn(
                fn=distributed_demo,
                args=(world_size, tensor_size, device),
                nprocs=world_size,
                join=True,
            )
            device_synchronize()
            end = timeit.default_timer()
            times[i] = end-start

        print(
            "\n-- Benchmark results --\n"
            "----------------------\n\n"
            f"Time for World Size:{world_size} and tensor size: {tensor_size/1024**2:.2f} MB:\n"
            f"  Average: {np.mean(times):.6e} s\n"
            f"  Std:     {np.std(times):.6e} s\n\n"
        )
