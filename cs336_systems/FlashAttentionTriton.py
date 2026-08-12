import triton
import triton.language as tl


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        B, Nq, d = Q.shape
        Nk = K.shape[1]
        Bq = Bk = 16

        O = torch.empty_like(Q, device=Q.device)
        L = torch.empty(B, Nq, device=Q.device, dtype=Q.dtype)

        flash_kernel_fwd[(triton.cdiv(Nq, Bq), B)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk, 1.0 / (d ** 0.5),
            d, Bq, Bk, is_causal
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("Backward pass is not implemented for FlashAttentionTriton.")

@triton.jit
def flash_kernel_fwd(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
                     stride_qb, stride_qq, stride_qd, 
                     stride_kb, stride_kk, stride_kd,
                     stride_vb, stride_vk, stride_vd,
                     stride_ob, stride_oq, stride_od,
                     stride_lb, stride_lq,
                     N_QUERIES, N_KEYS, scale, 
                     D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, 
                     K_TILE_SIZE: tl.constexpr, is_causal: tl.constexpr):


    query_tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_id * stride_qb,
                                    shape = (N_QUERIES, D),
                                    strides = (stride_qq, stride_qd),
                                    offsets = (query_tile_id * Q_TILE_SIZE, 0),
                                    block_shape = (Q_TILE_SIZE, D),
                                    order = (1, 0)
                                    )

    K_block_ptr = tl.make_block_ptr(K_ptr + batch_id * stride_kb,
                                        shape = (N_KEYS, D),
                                        strides = (stride_kk, stride_kd),
                                        offsets = (0, 0),
                                        block_shape = (K_TILE_SIZE, D),
                                        order = (1, 0)
                                        )

    V_block_ptr = tl.make_block_ptr(V_ptr + batch_id * stride_vb,
                                            shape = (N_KEYS, D),
                                            strides = (stride_vk, stride_vd),
                                            offsets = (0, 0),
                                            block_shape = (K_TILE_SIZE, D),
                                            order = (1, 0)
                                            )

    O_block_ptr = tl.make_block_ptr(O_ptr + batch_id * stride_ob,
                                                shape = (N_QUERIES, D),
                                                strides = (stride_oq, stride_od),
                                                offsets = (query_tile_id * Q_TILE_SIZE, 0),
                                                block_shape = (Q_TILE_SIZE, D),
                                                order = (1, 0)
                                                )

    L_block_ptr = tl.make_block_ptr(L_ptr + batch_id * stride_lb, shape=(N_QUERIES,), 
                                    strides=(stride_lq,),  offsets=(query_tile_id * Q_TILE_SIZE,),
                                    block_shape=(Q_TILE_SIZE,), order=(0,))
        
    
        
    q = tl.load(Q_block_ptr)

    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        s = tl.dot(q, tl.trans(k)) * scale

        if is_causal:
            q_idx = query_tile_id * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s = s + tl.where(q_idx[:, None] >= k_idx[None, :], 0.0, -1e6)
            
        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        corr = tl.exp(m - m_new)

        l = corr * l + tl.sum(p, axis=1)
        o = o * corr[:, None]
        o = tl.dot(p.to(v.dtype), v, acc=o)

        m = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o / l[:, None]
    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, m + tl.log(l))

