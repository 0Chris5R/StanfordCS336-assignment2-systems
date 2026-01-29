

import torch
import numpy as np
import timeit
import itertools

from cs336_systems.benchmarking import annotated_scaled_dot_product_attention


def benchmark_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 8
    d_model_values = [16, 32, 64, 128]
    seq_len_values = [256, 1024, 4096, 8192, 16384]
    warmup_steps = 5
    n_runs = 100

    print(f"{'d_model':>8} | {'seq_len':>8} | {'Forward (ms)':>20} | {'Backward (ms)':>20} | {'Memory (MB)':>12}")
    print("-" * 85)

    for d_model, seq_len in itertools.product(d_model_values, seq_len_values):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            Q = torch.randn(batch_size, seq_len, d_model,
                            device=device, requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model,
                            device=device, requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model,
                            device=device, requires_grad=True)

            # Warmup
            for _ in range(warmup_steps):
                out = annotated_scaled_dot_product_attention(Q, K, V)
                out.sum().backward()
                Q.grad, K.grad, V.grad = None, None, None
                torch.cuda.synchronize()

            # Time forward passes
            times_forward = []
            times_backward = []
            for _ in range(n_runs):
                Q.grad, K.grad, V.grad = None, None, None
                start = timeit.default_timer()
                out = annotated_scaled_dot_product_attention(Q, K, V)
                torch.cuda.synchronize()
                times_forward.append(timeit.default_timer() - start)

                memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

                start_backward = timeit.default_timer()
                out.sum().backward()
                torch.cuda.synchronize()
                times_backward.append(timeit.default_timer() - start_backward)

            fwd = f"{np.mean(times_forward)*1000:.6e} +/- {np.std(times_forward)*1000:.6e}"
            bwd = f"{np.mean(times_backward)*1000:.6e} +/- {np.std(times_backward)*1000:.6e}"
            print(
                f"{d_model:>8} | {seq_len:>8} | {fwd:>20} | {bwd:>20} | {memory_mb:>12.1f}")

        except torch.cuda.OutOfMemoryError:
            print(
                f"{d_model:>8} | {seq_len:>8} | {'OOM':>20} | {'OOM':>20} | {'OOM':>12}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark_attention()
