from cs336_basics.model import Transformer
from cs336_basics.train import AdamW, Muon, get_batch, cross_entropy, gradient_clipping
import torch
import numpy as np
import timeit
import argparse


def benchmark(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, device, dtype, warmup_steps, batch_size, n_steps=10, compile_on=False, use_muon=False):

    model = Transformer(vocab_size, context_length, d_model, num_layers,
                        num_heads, d_ff, 10000, None, device, dtype, True, True)

    if compile_on and device.type == "cuda":
        model = torch.compile(model)

    if use_muon:
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if len(param.shape) == 2 and name not in ("output.W", "embedding.E"):
                muon_params.append(param)
            else:
                adamw_params.append(param)
        muon = Muon(muon_params, 1e-3, 0.1,
                    0.95, True)
        adamw = AdamW(adamw_params, 1e-3, 0.1,
                      (0.95, 0.9), eps=1e-7, cautious_weight_decay=True)
        optimizers = (muon, adamw)
    else:
        optimizer = AdamW(model.parameters(), 1e-3, 0.1,
                          (0.95, 0.9), eps=1e-7, cautious_weight_decay=True)
        optimizers = (optimizer,)

    # random data as we only care about measuring speed and memory
    dataset = np.random.rand(context_length * batch_size * 2)
    inputs, targets = get_batch(dataset, batch_size, context_length, device)

    # Warmup
    if warmup_steps > 0:
        for _ in range(warmup_steps):

            logits = model(inputs, True)
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1))

            for optimizer in optimizers:
                optimizer.zero_grad()

            loss.backward()

            gradient_clipping(model.parameters(), 1)

            for optimizer in optimizers:
                optimizer.step()

    def device_synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    # actual benchmarking:

    times_forward = np.empty(n_steps)
    times_backward = np.empty(n_steps)

    inputs, targets = get_batch(
        dataset, batch_size, context_length, device)

    for i in range(n_steps):

        start_forward = timeit.default_timer()
        logits = model(inputs, True)

        device_synchronize()
        end_forward = timeit.default_timer()
        times_forward[i] = end_forward-start_forward

        start_backward = timeit.default_timer()
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1))

        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()

        gradient_clipping(model.parameters(), 1)

        for optimizer in optimizers:
            optimizer.step()

        device_synchronize()

        end_backward = timeit.default_timer()
        times_backward[i] = end_backward-start_backward

    print(
        "\n-- Benchmark results --\n"
        "----------------------\n\n"
        "Forward pass:\n"
        f"  Average: {np.mean(times_forward):.6e} s\n"
        f"  Std:     {np.std(times_forward):.6e} s\n\n"
        "Backward pass:\n"
        f"  Average: {np.mean(times_backward):.6e} s\n"
        f"  Std:     {np.std(times_backward):.6e} s"
    )


# make parameters configurable through cli arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end benchmarking for Transformer forward/backward passes"
    )

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=10)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu",
        choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"]
    )

    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--use-muon", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("\nRunning benchmark with configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    benchmark(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
        dtype=dtype,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        compile_on=args.compile,
        use_muon=args.use_muon,
    )


if __name__ == "__main__":
    main()
