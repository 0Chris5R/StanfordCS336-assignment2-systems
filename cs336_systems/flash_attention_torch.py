import torch
import math


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        seq_len = Q.shape[-2]

        # we assume inputs are by default clean powers of 2 and at least 16.

        MAX_TILE_SIZE = 64
        num_tiles = (seq_len + MAX_TILE_SIZE - 1) // MAX_TILE_SIZE

        O = torch.empty_like(V)
        L = torch.empty(*Q.shape[:-2], seq_len)

        for i in range(num_tiles):
            idx_q = i * MAX_TILE_SIZE
            tile_q = min(MAX_TILE_SIZE, seq_len - idx_q)

            m_prev = torch.full(
                (*Q.shape[:-2], tile_q), fill_value=-torch.inf)
            l_prev = torch.zeros((*Q.shape[:-2], tile_q))
            o_prev = torch.zeros((*V.shape[:-2], tile_q, V.shape[-1]))

            kv_tile_end = num_tiles
            if is_causal:
                pos_q = torch.arange(idx_q, idx_q + tile_q)
                q_min = idx_q
                q_max = idx_q + tile_q - 1
                kv_tile_end = (q_max + 1 + MAX_TILE_SIZE - 1) // MAX_TILE_SIZE

            for j in range(kv_tile_end):

                idx_kv = j * MAX_TILE_SIZE
                tile_kv = min(MAX_TILE_SIZE, seq_len - idx_kv)

                scores = (Q[..., idx_q:idx_q+tile_q, :] @ K[..., idx_kv:idx_kv +
                                                            tile_kv, :].transpose(-1, -2)) / math.sqrt(K.shape[-1])

                if is_causal:

                    k_max = idx_kv + tile_kv - 1

                    if not (q_min >= k_max):

                        pos_k = torch.arange(idx_kv, idx_kv + tile_kv)

                        mask = pos_q[:, None] >= pos_k[None, :]

                        scores = scores.masked_fill(~mask, -1e6)

                m = torch.maximum(m_prev, torch.max(
                    scores, dim=-1).values)

                p = torch.exp(scores - m[..., None])

                l = torch.exp(m_prev - m) * l_prev + torch.sum(p, dim=-1)

                o_i = torch.exp(m_prev-m)[..., None] * o_prev + p @ V[..., idx_kv:idx_kv +
                                                                      tile_kv, :]

                o_prev = o_i
                m_prev = m
                l_prev = l

            O[..., idx_q:idx_q+tile_q, :] = o_i/l[..., None]
            L[..., idx_q:idx_q+tile_q] = m + torch.log(l)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_out):

        # written like it would make sense on GPU with KV outer and Q inner loop to have less HBM reads and writes - on triton we would split this into 2 kernels to avoid atomic writes

        MAX_TILE_SIZE = 64
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        seq_len = Q.shape[-2]
        num_tiles = (seq_len + MAX_TILE_SIZE - 1) // MAX_TILE_SIZE

        dV = torch.zeros_like(V)
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)

        for i in range(num_tiles):
            idx_kv = i * MAX_TILE_SIZE
            tile_kv = min(MAX_TILE_SIZE, seq_len - idx_kv)

            dK_i = torch.zeros_like(K[..., idx_kv:idx_kv+tile_kv, :])
            dV_i = torch.zeros_like(V[..., idx_kv:idx_kv+tile_kv, :])

            q_tiles_start = 0
            if is_causal:
                pos_k = torch.arange(idx_kv, idx_kv + tile_kv)
                k_min = idx_kv
                k_max = idx_kv + tile_kv - 1

                q_tiles_start = k_min // MAX_TILE_SIZE

            for j in range(q_tiles_start, num_tiles):
                idx_q = j * MAX_TILE_SIZE
                tile_q = min(MAX_TILE_SIZE, seq_len - idx_q)

                S = (Q[..., idx_q:idx_q+tile_q, :] @ K[..., idx_kv:idx_kv +
                                                       tile_kv, :].transpose(-1, -2)) / math.sqrt(K.shape[-1])

                if is_causal:
                    q_min = idx_q

                    if not (q_min >= k_max):

                        pos_q = torch.arange(idx_q, idx_q + tile_q)
                        mask = pos_q[:, None] >= pos_k[None, :]

                        S = S.masked_fill(~mask, value=-1e6)

                P = torch.exp(S-L[..., idx_q:idx_q+tile_q, None])
                dV_j = P.transpose(-1, -
                                   2) @ grad_out[..., idx_q:idx_q+tile_q, :]
                dP = grad_out[..., idx_q:idx_q+tile_q, :] @ V[..., idx_kv:idx_kv +
                                                              tile_kv, :].transpose(-1, -2)
                dS = P * (dP - torch.sum(O[..., idx_q:idx_q+tile_q, :] *
                                         grad_out[..., idx_q:idx_q+tile_q, :], dim=-1)[..., None])
                dQ_j = dS @ K[..., idx_kv:idx_kv+tile_kv, :] / \
                    math.sqrt(K.shape[-1])

                dK_j = dS.transpose(-1, -2) @ Q[..., idx_q:idx_q +
                                                tile_q, :] / math.sqrt(K.shape[-1])

                dQ[..., idx_q:idx_q+tile_q, :] += dQ_j
                dV_i += dV_j
                dK_i += dK_j

            dV[..., idx_kv:idx_kv+tile_kv, :] = dV_i
            dK[..., idx_kv:idx_kv+tile_kv, :] = dK_i

        return dQ, dK, dV, None
