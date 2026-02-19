
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
from cs336_basics.train import AdamW, cross_entropy, gradient_clipping
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


def ddp_training(rank, world_size, device, training_steps, dataset, model_params, optimizer_params, batch_size, state_dict, ddp_type="naive", bucket_size=None, optimizer_sharding=False):

    setup(rank, world_size, device)
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        data_device = "cuda"
    else:
        data_device = device.type
    model_ddp = Transformer(**model_params, device=data_device,
                            dtype=torch.float32)
    model_ddp.load_state_dict(state_dict)
    if ddp_type == "parameter":
        model_ddp = DDPParameter(model_ddp)

    if ddp_type == "bucket":
        model_ddp = DDPBucket(model_ddp, bucket_size)

    if rank == 0:
        print(
            f"Memory at model initialization: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

    if optimizer_sharding is True:
        optimizer_ddp = ShardOptimizer(
            model_ddp.parameters(), AdamW, **optimizer_params)

        if rank == 0:
            print("With sharded optimizer: ")

    else:
        optimizer_ddp = AdamW(model_ddp.parameters(), **optimizer_params)

    if rank == 0:
        times = np.empty(5)
        times_grad_transfer = np.empty(5)

    # NVTX range for the entire training loop - visible in Nsight
    range_name = f"DDP_{ddp_type}_rank{rank}"
    nvtx.range_push(range_name)

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
        optimizer_ddp.zero_grad()
        loss.backward()
        if step >= 5 and rank == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_grad_transfer_start = timeit.default_timer()
        if ddp_type == "naive":
            gradients = []
            for param in model_ddp.parameters():
                gradients.append(param.grad)
            flat_grads = _flatten_dense_tensors(gradients)
            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
            unflatted_gradients = _unflatten_dense_tensors(
                flat_grads, gradients)
            for i, param in enumerate(model_ddp.parameters()):
                param.grad.copy_(unflatted_gradients[i] / world_size)
            if step >= 5 and rank == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                time_grad_transfer_stop = timeit.default_timer()
                times_grad_transfer[step-5] = time_grad_transfer_stop - \
                    time_grad_transfer_start
        if ddp_type == "parameter" or ddp_type == "bucket":
            model_ddp.finish_gradient_synchronization()
        gradient_clipping(model_ddp.parameters(), 1.0)
        if step == 0 and rank == 0:
            print(
                f"Memory before optimizer step: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

        optimizer_ddp.step()

        if step == 0 and rank == 0:
            print(
                f"Memory after optimizer step: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

        if step >= 5 and rank == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_end = timeit.default_timer()
            times[step-5] = time_end - time_start

    nvtx.range_pop()  # End NVTX range

    if rank == 0:
        if ddp_type == "naive":
            torch.save(model_ddp.state_dict(), "ddp_naive_weights.pt")
            print(
                f"Avg time per training step with naive DDP: {np.mean(times):.6e} s")
            print(
                f"Time spent transfering gradients: {np.mean(times_grad_transfer):.6e} s . This equals {100*times_grad_transfer.mean()/times.mean():.2f} % of total time")
        elif ddp_type == "parameter":
            torch.save(model_ddp.module.state_dict(),
                       "ddp_parameter_weights.pt")
            print(
                f"Avg time per training step with parameter overlap DDP: {np.mean(times):.6e} s")

        elif ddp_type == "bucket":
            torch.save(model_ddp.module.state_dict(),
                       "ddp_bucket_weights.pt")
            print(
                f"Avg time per training step with bucket size {bucket_size}MB overlap DDP : {np.mean(times):.6e} s")

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

    mp.spawn(
        fn=ddp_training,
        args=(world_size, device, training_steps, dataset,
              model_params, optimizer_params, batch_size, state_dict, "naive"),
        nprocs=world_size,
        join=True,
    )

    gc.collect()
    torch.cuda.empty_cache()

    mp.spawn(
        fn=ddp_training,
        args=(world_size, device, training_steps, dataset,
              model_params, optimizer_params, batch_size, state_dict, "parameter"),
        nprocs=world_size,
        join=True,
    )

    gc.collect()
    torch.cuda.empty_cache()

    for bucket_sz in [1, 10, 100, 1000]:

        mp.spawn(
            fn=ddp_training,
            args=(world_size, device, training_steps, dataset,
                  model_params, optimizer_params, batch_size, state_dict, "bucket", bucket_sz),
            nprocs=world_size,
            join=True,
        )

        gc.collect()
        torch.cuda.empty_cache()

    for shard_optimizer in [False, True]:
        mp.spawn(
            fn=ddp_training,
            args=(world_size, device, training_steps, dataset,
                  model_params, optimizer_params, batch_size, state_dict, "parameter", None, shard_optimizer),
            nprocs=world_size,
            join=True,
        )

        gc.collect()
        torch.cuda.empty_cache()

    model = Transformer(**model_params, device=device, dtype=torch.float32)
    model.load_state_dict(state_dict)
    optimizer = AdamW(model.parameters(), **optimizer_params)

    times = np.empty(5)
    nvtx.range_push("Single_process")
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
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        optimizer.step()
        if step >= 5:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_end = timeit.default_timer()
            times[step-5] = time_end - time_start
    nvtx.range_pop()
    print(
        f"Avg time per training step on single process: {np.mean(times):.6e} s")

    ddp_parameter_weights = torch.load("ddp_naive_weights.pt")

    single_process_weights = model.state_dict()
    for key in ddp_parameter_weights:
        torch.testing.assert_close(
            ddp_parameter_weights[key], single_process_weights[key], atol=1e-4, rtol=1e-4)

    print("Naive and single proccess weights match!")

    ddp_parameter_weights = torch.load("ddp_parameter_weights.pt")

    for key in ddp_parameter_weights:
        torch.testing.assert_close(
            ddp_parameter_weights[key], single_process_weights[key], atol=1e-4, rtol=1e-4)

    print("Parameter overlap and single proccess weights match!")

    ddp_bucket_weights = torch.load("ddp_bucket_weights.pt")
    for key in ddp_bucket_weights:
        torch.testing.assert_close(
            ddp_bucket_weights[key], single_process_weights[key], atol=1e-4, rtol=1e-4)

    print("Bucket overlap and single process weights match!")
