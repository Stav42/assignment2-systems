import torch


@torch.compile
def flash_backward(Q, K, V, O, dO, L, is_causal=False):
    d = Q.shape[-1]
    scale = 1.0 / (d ** 0.5)

    # This part is the recompute part
    S = torch.einsum("bqd,bkd->bqk", Q, K) * scale
    if is_causal:
        q_idx = torch.arange(Q.shape[1], device=Q.device)
        k_idx = torch.arange(K.shape[1], device=Q.device)
        S = S + torch.where(q_idx[:, None] >= k_idx[None, :], 0.0, -1e6)

    # Exponentiate in fp32 since L may be fp32, then match the input dtype so
    # the matmuls below see consistent operands.
    P  = torch.exp(S.float() - L.unsqueeze(-1)).to(Q.dtype)
    dV = torch.einsum("bqk,bqd->bkd", P, dO)
    dP = torch.einsum("bqd,bkd->bqk", dO, V)
    D  = (O * dO).sum(dim=-1)
    dS = P * (dP - D.unsqueeze(-1))
    dQ = torch.einsum("bqk,bkd->bqd", dS, K) * scale
    dK = torch.einsum("bqk,bqd->bkd", dS, Q) * scale

    return dQ, dK, dV


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, d = Q.shape
        Nk = K.shape[1]

        Bq = Bk = 16
        scale = 1.0 / (d ** 0.5)

        O = torch.empty_like(Q, device=Q.device)
        L = torch.empty(B, Nq, device=Q.device, dtype=Q.dtype)

        for i in range (Nq // Bq):
            q_i = Q[:, i*Bq:(i+1)*Bq, :]
            o_i = torch.zeros(B, Bq, d, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, Bq, device=Q.device, dtype=Q.dtype)
            m_i = torch.full((B, Bq), float('-inf'), device=Q.device, dtype=Q.dtype)

            for j in range (Nk // Bk):
                k_j = K[:, j*Bk:(j+1)*Bk, :]
                v_j = V[:, j*Bk:(j+1)*Bk, :]

                S_ij = torch.bmm(q_i, k_j.transpose(1, 2)) * scale
                m_new = torch.max(m_i, torch.max(S_ij, dim=2).values)
                p_ij = torch.exp(S_ij - m_new.unsqueeze(2))
                corr = torch.exp(m_i - m_new)
                l_i = corr * l_i + torch.sum(p_ij, dim=2)
                o_i = corr.unsqueeze(2) * o_i + torch.bmm(p_ij, v_j)
                m_i = m_new

            O[:, i*Bq:(i+1)*Bq, :] = o_i / l_i.unsqueeze(2)
            L[:, i*Bq:(i+1)*Bq] = m_i + torch.log(l_i)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
        # raise NotImplementedError("Backward pass is not implemented for FlashAttentionPytorch.")
