import torch
import triton
import itertools
from cs336_systems.flash_attention_triton import FlashAttention
from cs336_basics.model import MultiHeadSelfAttention


def benchmark_attention():
    seq_lengths = [2**i for i in range(7, 17)]  # 128 to 65536
    embed_dims = [2**i for i in range(4, 8)]     # 16 to 128
    # T4 supports float16 but no bf16 - so using float16 here.
    precisions = [torch.float32, torch.float16]
    pytorch_attn_compiled = torch.compile(
        MultiHeadSelfAttention.scaled_dot_product_attention)
    results = []
    for seq_len, d_head, dtype in itertools.product(seq_lengths, embed_dims, precisions):
        q = torch.randn(1, seq_len, d_head, device='cuda',
                        dtype=dtype, requires_grad=True)
        k = torch.randn(1, seq_len, d_head, device='cuda',
                        dtype=dtype, requires_grad=True)
        v = torch.randn(1, seq_len, d_head, device='cuda',
                        dtype=dtype, requires_grad=True)
        do = torch.randn(1, seq_len, d_head, device='cuda', dtype=dtype)

        print(
            f"seq_len={seq_len:5d}, d_head={d_head:3d}, dtype={str(dtype).split('.')[-1]:8s}")

        # Triton FlashAttention
        try:
            flash_fwd_ms = triton.testing.do_bench(
                lambda: FlashAttention.apply(q, k, v, True)
            )

            def flash_bwd():
                q.grad = k.grad = v.grad = None
                o = FlashAttention.apply(q, k, v, True)
                o.backward(do)
            flash_fwd_bwd_ms = triton.testing.do_bench(flash_bwd)
            flash_bwd_ms = flash_fwd_bwd_ms - flash_fwd_ms
            print(
                f"  Flash:            fwd={flash_fwd_ms:8.3f}ms  bwd={flash_bwd_ms:8.3f}ms  total={flash_fwd_bwd_ms:8.3f}ms")
        except Exception as e:
            flash_fwd_ms = flash_bwd_ms = flash_fwd_bwd_ms = None
            print(f"  Flash:            ERROR: {e}")

        torch.cuda.empty_cache()

        # PyTorch attention (compiled)
        try:
            causal_mask = torch.triu(torch.ones(
                seq_len, seq_len, device='cuda'), diagonal=1).bool()
            pytorch_compiled_fwd_ms = triton.testing.do_bench(
                lambda: pytorch_attn_compiled(q, k, v, causal_mask)
            )

            def pytorch_compiled_bwd():
                q.grad = k.grad = v.grad = None
                o = pytorch_attn_compiled(q, k, v, causal_mask)
                o.backward(do)
            pytorch_compiled_fwd_bwd_ms = triton.testing.do_bench(
                pytorch_compiled_bwd)
            pytorch_compiled_bwd_ms = pytorch_compiled_fwd_bwd_ms - pytorch_compiled_fwd_ms
            print(
                f"  PyTorch compiled: fwd={pytorch_compiled_fwd_ms:8.3f}ms  bwd={pytorch_compiled_bwd_ms:8.3f}ms  total={pytorch_compiled_fwd_bwd_ms:8.3f}ms")
        except Exception as e:
            pytorch_compiled_fwd_ms = pytorch_compiled_bwd_ms = pytorch_compiled_fwd_bwd_ms = None
            print(f"  PyTorch compiled: ERROR: OOM")

        torch.cuda.empty_cache()

        results.append({
            'seq_len': seq_len,
            'd_head': d_head,
            'dtype': str(dtype).split('.')[-1],
            'flash_fwd_ms': flash_fwd_ms,
            'flash_bwd_ms': flash_bwd_ms,
            'flash_fwd_bwd_ms': flash_fwd_bwd_ms,
            'pytorch_compiled_fwd_ms': pytorch_compiled_fwd_ms,
            'pytorch_compiled_bwd_ms': pytorch_compiled_bwd_ms,
            'pytorch_compiled_fwd_bwd_ms': pytorch_compiled_fwd_bwd_ms,
        })
        print()
    return results


if __name__ == "__main__":
    results = benchmark_attention()
