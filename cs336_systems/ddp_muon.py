
from cs336_systems.ddp_parameter import DDPParameter, DDPBucket
from cs336_systems.shard_optimizer import ShardOptimizer
from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)
import os
import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import timeit
import gc
from cs336_basics.model import Transformer
from cs336_basics.train import AdamW, Muon, cross_entropy, gradient_clipping
import warnings
warnings.filterwarnings(
    "ignore", message=".*'repr'.*Field.*|.*'frozen'.*Field.*")


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


def get_batch_sharded(dataset: np.ndarray | str, batch_size: int, context_length: int, device: str, rank, world_size, step) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dataset, str):
        dataset = np.load(dataset, mmap_mode='r')
    random = np.random.default_rng(seed=step)
    starting_indices = random.integers(
        0, len(dataset)-context_length, batch_size)
    batch_size_per_rank = batch_size//world_size
    starting_indices = starting_indices[batch_size_per_rank *
                                        rank: batch_size_per_rank * rank + batch_size_per_rank]
    inputs = [dataset[i:i+context_length] for i in starting_indices]
    targets = [dataset[i+1:i+1+context_length] for i in starting_indices]
    inputs = torch.tensor(np.stack(inputs), device=device, dtype=torch.int32)
    targets = torch.tensor(np.stack(targets), device=device, dtype=torch.int32)
    return inputs, targets


def ddp_training(rank, world_size, device, training_steps, dataset, model_params, optimizer_params, batch_size, state_dict, ddp_type="naive", bucket_size=None, optimizer_sharding=False, shard_gradient=False):

    setup(rank, world_size, device)
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        data_device = "cuda"
    else:
        print("No GPU available")
        data_device = device.type
    model_ddp = Transformer(**model_params, device=data_device,
                            dtype=torch.float32)
    model_ddp.load_state_dict(state_dict)

    if ddp_type == "parameter" and shard_gradient:
        model_ddp = DDPParameter(model_ddp, sharded=True)

    if rank == 0:
        print("With sharded gradients")
        print(
            f"Memory at model initialization: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

    parameters_adam = []
    parameters_muon = []

    for param in model_ddp.parameters():
        if param.ndim >= 2:
            parameters_muon.append(param)
        else:
            parameters_adam.append(param)

    optimizer_adam_ddp = ShardOptimizer(
        parameters_adam, AdamW, **optimizer_params)
    optimizer_muon_ddp = ShardOptimizer(
        parameters_muon, Muon, **optimizer_params)

    if rank == 0:
        print("With sharded optimizer: ")

    if rank == 0:
        times = np.empty(5)
        times_grad_transfer = np.empty(5)

    for step in range(training_steps):
        dist.barrier(device_ids=[rank])
        if step >= 5 and rank == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_start = timeit.default_timer()
        inputs, targets = get_batch_sharded(
            dataset, batch_size=batch_size, context_length=model_params["context_length"], device=data_device, rank=rank, world_size=world_size, step=step)
        logits = model_ddp(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)),
                             targets.view(-1))
        optimizer_adam_ddp.zero_grad()
        optimizer_muon_ddp.zero_grad()
        loss.backward()
        if step >= 5 and rank == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
        if ddp_type == "parameter":
            model_ddp.finish_gradient_synchronization()

            local_grad_norm = 0
            for parameter in model_ddp.parameters():
                if parameter.grad is None:
                    continue

                local_grad_norm += torch.sum(parameter.grad ** 2)

            dist.all_reduce(local_grad_norm, op=dist.ReduceOp.SUM)

            grad_norm = torch.sqrt(local_grad_norm).item()

            if grad_norm > 1.0:
                scale = 1.0/grad_norm
                for parameter in model_ddp.parameters():
                    if parameter.grad is None:
                        continue
                    parameter.grad.mul_(scale)

        if step == 0 and rank == 0:
            print(
                f"Memory before optimizer step: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

        optimizer_adam_ddp.step()
        optimizer_muon_ddp.step()

        if step == 0 and rank == 0:
            print(
                f"Memory after optimizer step: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

        if step >= 5 and rank == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_end = timeit.default_timer()
            times[step-5] = time_end - time_start

    if rank == 0:
        torch.save(model_ddp.module.state_dict(),
                   "ddp_parameter_weights.pt")
        print(
            f"Avg time per training step with parameter overlap DDP: {np.mean(times):.6e} s")

    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Running on GPU(s)")
    world_size = 2
    model_params = {
        "vocab_size": 10000,
        "context_length": 256,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "rope_theta": 10000.0,
        "weights": None,
    }
    optimizer_params = {
        "lr": 1e-3,
        "weight_decay": 0.01,
    }
    dataset = np.random.randint(0, 100, size=(1000,))
    batch_size = 16
    training_steps = 10

    model = Transformer(**model_params, device="cpu", dtype=torch.float32)
    state_dict = model.state_dict()
    del model
    gc.collect()

    # sharded optimizer and gradients
    mp.spawn(
        fn=ddp_training,
        args=(world_size, device, training_steps, dataset,
              model_params, optimizer_params, batch_size, state_dict, "parameter", None, True,
              True),
        nprocs=world_size,
        join=True,
    )

    gc.collect()
    torch.cuda.empty_cache()

    model = Transformer(**model_params, device=device, dtype=torch.float32)
    model.load_state_dict(state_dict)

    parameters_adam = []
    parameters_muon = []

    for param in model.parameters():
        if param.ndim >= 2:
            parameters_muon.append(param)
        else:
            parameters_adam.append(param)

    optimizer_adam = AdamW(parameters_adam, **optimizer_params)
    optimizer_muon = Muon(parameters_muon, **optimizer_params)

    times = np.empty(5)
    for step in range(training_steps):
        if step >= 5:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_start = timeit.default_timer()
        inputs, targets = get_batch_sharded(
            dataset, batch_size=batch_size, context_length=model_params["context_length"], device=device, rank=0, world_size=1, step=step)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)),
                             targets.view(-1))
        optimizer_adam.zero_grad()
        optimizer_muon.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        optimizer_adam.step()
        optimizer_muon.step()
        if step >= 5:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_end = timeit.default_timer()
            times[step-5] = time_end - time_start
    print(
        f"Avg time per training step on single process: {np.mean(times):.6e} s")

    single_process_weights = model.state_dict()

    ddp_parameter_weights = torch.load("ddp_parameter_weights.pt")

    for key in ddp_parameter_weights:
        torch.testing.assert_close(
            ddp_parameter_weights[key], single_process_weights[key], atol=1e-4, rtol=1e-4)

    print("Distributed, sharded and single proccess weights match!")
