
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

from cs336_basics.model import Transformer
from cs336_basics.train import AdamW, cross_entropy, gradient_clipping


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

    for step in range(training_steps):

        inputs, targets = get_batch_sharded(
            dataset, batch_size=batch_size, context_length=model_params["context_length"], device=data_device, rank=rank, world_size=world_size, step=step)

        logits = model_ddp(inputs)

        loss = cross_entropy(logits.view(-1, logits.size(-1)),
                             targets.view(-1))

        optimizer_ddp.zero_grad()

        loss.backward()

        for param in model_ddp.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

        gradient_clipping(model_ddp.parameters(), 1.0)

        optimizer_ddp.step()

    if rank == 0:
        torch.save(model_ddp.state_dict(), "ddp_weights.pt")


if __name__ == "__main__":

    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print("Running on GPU(s)")

    world_size = 2

    model_params = {
        "vocab_size": 100,
        "context_length": 32,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 2,
        "d_ff": 128,
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
    batch_size = 8
    training_steps = 5

    mp.spawn(
        fn=naive_ddp_training,
        args=(world_size, device, training_steps, dataset,
              model_params, optimizer_params, batch_size, model.state_dict()),
        nprocs=world_size,
        join=True,
    )

    for step in range(training_steps):

        inputs, targets = get_batch_sharded(
            dataset, batch_size=batch_size, context_length=model_params["context_length"], device=device, rank=0, world_size=1, step=step)

        logits = model(inputs)

        loss = cross_entropy(logits.view(-1, logits.size(-1)),
                             targets.view(-1))

        optimizer.zero_grad()

        loss.backward()

        gradient_clipping(model.parameters(), 1.0)

        optimizer.step()

    ddp_weights = torch.load("ddp_weights.pt")
    single_process_weights = model.state_dict()

    for key in ddp_weights:
        torch.testing.assert_close(
            ddp_weights[key], single_process_weights[key])

    print("All weights match!")
