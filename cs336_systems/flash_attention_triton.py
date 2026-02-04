import torch
import math
import triton
import triton.language as tl


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        D = Q.shape[-1]
        scale = 1/math.sqrt(D)

        O = torch.empty_like(V)
        L = torch.empty_like(Q[..., 0])

        flash_fwd_kernel(
            Q, K, V,
            O, L,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *L.stride(),
            Q.shape[-2], K.shape[-2],
            scale,
            D=D,
            Q_TILE_SIZE=64,
            K_TILE_SIZE=64,
        )

        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, grad_out):
        pass


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_qb,
        shape=(N_QUERIES),
        strides=(stride_lb, stride_lq),
        offsets=(query_tile_index * Q_TILE_SIZE),
        block_shape=(Q_TILE_SIZE),

    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):

        Kj = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        Sj = tl.dot(Qi, Kj) * scale

        mj = tl.max(m_i, tl.max(Sj, axis=0))

        Pj = tl.exp(Sj - mj[:, None])

        l_i = tl.exp(m_i - mj) * l_i + tl.sum(Pj, axis=0)

        O_i = tl.exp(m_i - mj)[:, None] * O_i + tl.dot(Pj.to(Vj.dtype), Vj)

        m_i = mj

        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i /= l_i

    l_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i,
             boundary_check=(0,))

    tl.store(L_block_ptr, l_i, boundary_check=(0,))
