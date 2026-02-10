
from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import timeit

from cs336_basics.model import Transformer
from cs336_basics.train import AdamW, cross_entropy, gradient_clipping
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


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


def naive_ddp_training(rank, world_size, device, training_steps, dataset, model_params, optimizer_params, batch_size, state_dict):
    setup(rank, world_size, device)

    if device.type == "cuda":
        torch.cuda.set_device(rank)
        data_device = "cuda"

    else:
        data_device = device.type

    model_ddp = Transformer(**model_params, device=data_device,
                            dtype=torch.float32)

    model_ddp.load_state_dict(state_dict)

    optimizer_ddp = AdamW(model_ddp.parameters(), **optimizer_params)

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

        optimizer_ddp.zero_grad()

        loss.backward()

        if step >= 5 and rank == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_grad_transfer_start = timeit.default_timer()

        # Super naive transfer: 1 broadcast per parameter
        # for param in model_ddp.parameters():
        #     dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        #     param.grad /= world_size

        # Slightly less naive: Flatten the params to one tensor, broadcast and then unflatten
        gradients = []
        for param in model_ddp.parameters():
            gradients.append(param.grad)

        flat_grads = _flatten_dense_tensors(gradients)

        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)

        unflatted_gradients = _unflatten_dense_tensors(flat_grads, gradients)

        for i, param in enumerate(model_ddp.parameters()):
            param.grad.copy_(unflatted_gradients[i] / world_size)

        if step >= 5 and rank == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_grad_transfer_stop = timeit.default_timer()
            times_grad_transfer[step-5] = time_grad_transfer_stop - \
                time_grad_transfer_start

        gradient_clipping(model_ddp.parameters(), 1.0)

        optimizer_ddp.step()

        if step >= 5 and rank == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_end = timeit.default_timer()
            times[step-5] = time_end - time_start

    if rank == 0:
        torch.save(model_ddp.state_dict(), "ddp_weights.pt")
        print(
            f"Avg time per training step with naive DDP: {np.mean(times):.6e} s")
        print(
            f"Time spent transfering gradients: {np.mean(times_grad_transfer):.6e} s . This equals {100*times_grad_transfer.mean()/times.mean():.2f} % of total time")

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
        "d_model": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "d_ff": 5120,
        "rope_theta": 10000.0,
        "weights": None,
    }

    optimizer_params = {
        "lr": 1e-3,
        "weight_decay": 0.01,
    }

    model = Transformer(**model_params, device=device, dtype=torch.float32)

    optimizer = AdamW(model.parameters(), **optimizer_params)

    dataset = np.random.randint(0, 100, size=(1000,))
    batch_size = 4
    # First 5 steps warmup
    training_steps = 10

    mp.spawn(
        fn=naive_ddp_training,
        args=(world_size, device, training_steps, dataset,
              model_params, optimizer_params, batch_size, model.state_dict()),
        nprocs=world_size,
        join=True,
    )

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

        optimizer.zero_grad()

        loss.backward()

        gradient_clipping(model.parameters(), 1.0)

        optimizer.step()

        if step >= 5:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_end = timeit.default_timer()
            times[step-5] = time_end - time_start

    print(
        f"Avg time per training step on single process: {np.mean(times):.6e} s")

    ddp_weights = torch.load("ddp_weights.pt")
    single_process_weights = model.state_dict()

    for key in ddp_weights:
        torch.testing.assert_close(
            ddp_weights[key], single_process_weights[key], atol=1e-4, rtol=1e-4)

    print("All weights match!")
