import torch
import triton
import itertools
from cs336_systems.flash_attention_triton import FlashAttention
from cs336_systems.benchmarking import annotated_scaled_dot_product_attention


def benchmark_attention():
    seq_lengths = [2**i for i in range(7, 17)]  # 128 to 65536
    embed_dims = [2**i for i in range(4, 8)]     # 16 to 128
    # T4 supports float16 but no bf16 - so using float16 here.
    precisions = [torch.float32, torch.float16]

    pytorch_attn_compiled = torch.compile(
        annotated_scaled_dot_product_attention)

    results = []

    for seq_len, d_head, dtype in itertools.product(seq_lengths, embed_dims, precisions):
        try:
            q = torch.randn(1, seq_len, d_head, device='cuda',
                            dtype=dtype, requires_grad=True)
            k = torch.randn(1, seq_len, d_head, device='cuda',
                            dtype=dtype, requires_grad=True)
            v = torch.randn(1, seq_len, d_head, device='cuda',
                            dtype=dtype, requires_grad=True)
            do = torch.randn(1, seq_len, d_head, device='cuda', dtype=dtype)

            # Triton FlashAttention forward
            flash_fwd_ms = triton.testing.do_bench(
                lambda: FlashAttention.apply(q, k, v, True)
            )

            # Triton FlashAttention forward+backward
            def flash_bwd():
                q.grad = k.grad = v.grad = None
                o = FlashAttention.apply(q, k, v, True)
                o.backward(do)
            flash_fwd_bwd_ms = triton.testing.do_bench(flash_bwd)
            flash_bwd_ms = flash_fwd_bwd_ms - flash_fwd_ms

            # PyTorch attention (uncompiled) forward
            pytorch_fwd_ms = triton.testing.do_bench(
                lambda: annotated_scaled_dot_product_attention(
                    q, k, v, is_causal=True)
            )

            # PyTorch attention (uncompiled) forward+backward
            def pytorch_bwd():
                q.grad = k.grad = v.grad = None
                o = annotated_scaled_dot_product_attention(
                    q, k, v, is_causal=True)
                o.backward(do)
            pytorch_fwd_bwd_ms = triton.testing.do_bench(pytorch_bwd)
            pytorch_bwd_ms = pytorch_fwd_bwd_ms - pytorch_fwd_ms

            # PyTorch attention (compiled) forward
            pytorch_compiled_fwd_ms = triton.testing.do_bench(
                lambda: pytorch_attn_compiled(q, k, v, is_causal=True)
            )

            # PyTorch attention (compiled) forward+backward
            def pytorch_compiled_bwd():
                q.grad = k.grad = v.grad = None
                o = pytorch_attn_compiled(q, k, v, is_causal=True)
                o.backward(do)
            pytorch_compiled_fwd_bwd_ms = triton.testing.do_bench(
                pytorch_compiled_bwd)
            pytorch_compiled_bwd_ms = pytorch_compiled_fwd_bwd_ms - pytorch_compiled_fwd_ms

            results.append({
                'seq_len': seq_len,
                'd_head': d_head,
                'dtype': str(dtype).split('.')[-1],
                'flash_fwd_ms': flash_fwd_ms,
                'flash_bwd_ms': flash_bwd_ms,
                'flash_fwd_bwd_ms': flash_fwd_bwd_ms,
                'pytorch_fwd_ms': pytorch_fwd_ms,
                'pytorch_bwd_ms': pytorch_bwd_ms,
                'pytorch_fwd_bwd_ms': pytorch_fwd_bwd_ms,
                'pytorch_compiled_fwd_ms': pytorch_compiled_fwd_ms,
                'pytorch_compiled_bwd_ms': pytorch_compiled_bwd_ms,
                'pytorch_compiled_fwd_bwd_ms': pytorch_compiled_fwd_bwd_ms,
            })

            print(
                f"seq_len={seq_len:5d}, d_head={d_head:3d}, dtype={str(dtype).split('.')[-1]:8s}")
            print(
                f"  Flash:            fwd={flash_fwd_ms:8.3f}ms  bwd={flash_bwd_ms:8.3f}ms  total={flash_fwd_bwd_ms:8.3f}ms")
            print(
                f"  PyTorch:          fwd={pytorch_fwd_ms:8.3f}ms  bwd={pytorch_bwd_ms:8.3f}ms  total={pytorch_fwd_bwd_ms:8.3f}ms")
            print(
                f"  PyTorch compiled: fwd={pytorch_compiled_fwd_ms:8.3f}ms  bwd={pytorch_compiled_bwd_ms:8.3f}ms  total={pytorch_compiled_fwd_bwd_ms:8.3f}ms")
            print()

        except Exception as e:
            print(
                f"seq_len={seq_len:5d}, d_head={d_head:3d}, dtype={str(dtype).split('.')[-1]:8s} | ERROR: {e}")
            print()

        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    results = benchmark_attention()
