import torch
import math
import triton
import triton.language as tl


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        D = Q.shape[-1]
        N_QUERIES = Q.shape[-2]
        scale = 1/math.sqrt(D)
        Q_TILE_SIZE = min(64, Q.shape[-2])
        K_TILE_SIZE = min(64, K.shape[-2])
        NUM_Q_TILES = (N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        BATCH_SIZE = Q.shape[0]

        O = torch.empty_like(Q)
        L = torch.empty_like(Q[..., 0])

        flash_fwd_kernel[(NUM_Q_TILES, BATCH_SIZE)](
            Q, K, V,
            O, L,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *L.stride(),
            N_QUERIES, K.shape[-2],
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_out):

        L, Q, K, V, O = ctx.saved_tensors

        is_causal = ctx.is_causal

        dQ = torch.empty_like(Q)

        dK = torch.empty_like(K)

        dV = torch.empty_like(V)

        return dQ, dK, dV, None


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
    is_causal: tl.constexpr,
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
        order=(0, 1),
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
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),

    )

    Qi = tl.load(Q_block_ptr, boundary_check=(
        0,), padding_option="zero").to(tl.float16)

    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    keys_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        q_pos_global = tl.arange(0, Q_TILE_SIZE)
        q_pos_local = query_tile_index * Q_TILE_SIZE
        q_pos = q_pos_global + q_pos_local
        k_pos_global = tl.arange(0, K_TILE_SIZE)
        q_max = q_pos_local + Q_TILE_SIZE - 1
        keys_tiles = tl.cdiv(q_max+1, K_TILE_SIZE)

    for j in range(keys_tiles):
        if is_causal:
            k_pos_local = j*K_TILE_SIZE
            k_pos = k_pos_global + k_pos_local

        Kj = tl.load(K_block_ptr, boundary_check=(
            1,), padding_option="zero").to(tl.float16)
        Vj = tl.load(V_block_ptr, boundary_check=(
            0,), padding_option="zero").to(tl.float16)

        Sj = tl.dot(Qi, Kj).to(tl.float32) * scale

        if is_causal:
            mask = q_pos[:, None] >= k_pos[None, :]
            Sj = tl.where(mask, Sj, float('-inf'))

        mj = tl.maximum(m_i, tl.max(Sj, axis=1))

        Pj = tl.exp(Sj - mj[:, None])

        l_i = tl.exp(m_i - mj) * l_i + tl.sum(Pj, axis=1)

        O_i = tl.exp(m_i - mj)[:, None] * O_i + \
            tl.dot(Pj.to(tl.float16), Vj).to(tl.float32)

        m_i = mj

        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i /= l_i[:, None]

    l_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty),
             boundary_check=(0,))

    tl.store(L_block_ptr, l_i.to(
        L_block_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Ki = tl.load(K_block_ptr, boundary_check=(
        0,), padding_option="zero").to(tl.float16)

    Vi = tl.load(V_block_ptr, boundary_check=(0,),
                 padding_option="zero").to(tl.float16)

    dVi
    dKi

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    if is_causal:
        k_pos_local = key_tile_index * K_TILE_SIZE
        k_pos = k_pos_global + k_pos_local
        k_pos_global = tl.arange(0, K_TILE_SIZE)
        q_pos_global = tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):

        Qj = tl.load(Q_block_ptr, boundary_check=(
            0,), padding_option="zero").to(tl.float16)

        Sj = tl.dot(Qj, tl.trans(Ki)).to(tl.float32) * scale

        if is_causal:
            q_pos_local = j*Q_TILE_SIZE
            q_pos = q_pos_global + q_pos_local

            mask = q_pos[:, None] >= k_pos[None, :]
            Sj = tl.where(mask, Sj, Sj - 1e6)

        # mj = tl.maximum(m_i, tl.max(Sj, axis=1))

        # Pj = tl.exp(Sj - mj[:, None])

        # l_i = tl.exp(m_i - mj) * l_i + tl.sum(Pj, axis=1)

        # O_i = tl.exp(m_i - mj)[:, None] * O_i + \
        #     tl.dot(Pj.to(tl.float16), Vj).to(tl.float32)

        # m_i = mj

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))

    # O_i /= l_i[:, None]

    # l_i = m_i + tl.log(l_i)

    # tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty),
    #          boundary_check=(0,))

    # tl.store(L_block_ptr, l_i.to(
    #     L_block_ptr.type.element_ty), boundary_check=(0,))
