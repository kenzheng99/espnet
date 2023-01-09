#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Roshan Sharma (Carnegie Mellon University)
# Apache 2.0

"""X-NOR self-attention layer definition."""


import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn
import math
import logging


class XNorCosAttention(nn.Module):
    """
    XNorcos attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.2,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal
        self.dropout = torch.nn.Dropout(dropout_rate)

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        elif act_fun == "sig":
            return F.sigmoid
        elif act_fun == "swish":
            return F.silu

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.permute(1, 0, 2)  # query.view(tgt_len, bsz, embed_dim)
        key = key.permute(1, 0, 2)  # key.view(src_len, bsz, embed_dim)
        value = value.permute(1, 0, 2)  # value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # n_batch_pos = pos_emb.size(0)
        # p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        # p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        q_ = torch.cat(
            [
                q_ * torch.sin(weight_index[:, :tgt_len, :] / m),
                q_ * torch.cos(weight_index[:, :tgt_len, :] / m),
            ],
            dim=-1,
        )
        # (N * h, S, 2 * d)
        comp_q_ = torch.cat(
            [
                comp_q_ * torch.sin(weight_index[:, :tgt_len, :] / m),
                comp_q_ * torch.cos(weight_index[:, :tgt_len, :] / m),
            ],
            dim=-1,
        )
        k_ = torch.cat(
            [
                k_ * torch.sin(weight_index[:, :src_len, :] / m),
                k_ * torch.cos(weight_index[:, :src_len, :] / m),
            ],
            dim=-1,
        )
        comp_k_ = torch.cat(
            [
                comp_k_ * torch.sin(weight_index[:, :src_len, :] / m),
                comp_k_ * torch.cos(weight_index[:, :src_len, :] / m),
            ],
            dim=-1,
        )

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv + comp_qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", self.dropout(q_), torch.sum(k_, axis=1)), eps
            )
            comp_z_ = 1 / torch.clamp_min(
                torch.einsum(
                    "nld,nd->nl", self.dropout(comp_q_), torch.sum(comp_k_, axis=1)
                ),
                eps,
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output += attn_output2
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class XNorCosPosAttention(nn.Module):
    """
    XNor with  attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        elif act_fun == "sig":
            return F.sigmoid
        elif act_fun == "swish":
            return F.silu

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # q_ = torch.cat(
        #     [
        #         q * torch.sin(weight_index[:, :tgt_len, :] / m),
        #         q * torch.cos(weight_index[:, :tgt_len, :] / m),
        #     ],
        #     dim=-1,
        # )
        q_ = q
        comp_q_ = 1 - q
        k_ = k
        comp_k_ = 1 - k_
        # (N * h, S, 2 * d)
        # comp_q_ = torch.cat(
        #     [
        #         comp_q * torch.sin(weight_index[:, :tgt_len, :] / m),
        #         comp_q * torch.cos(weight_index[:, :tgt_len, :] / m),
        #     ],
        #     dim=-1,
        # )
        # k_ = torch.cat(
        #     [
        #         k * torch.sin(weight_index[:, :src_len, :] / m),
        #         k * torch.cos(weight_index[:, :src_len, :] / m),
        #     ],
        #     dim=-1,
        # )
        # comp_k_ = torch.cat(
        #     [
        #         comp_k * torch.sin(weight_index[:, :src_len, :] / m),
        #         comp_k * torch.cos(weight_index[:, :src_len, :] / m),
        #     ],
        #     dim=-1,
        # )

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            weights = torch.bmm(q_, k_.transpose(1, 2)) + torch.bmm(
                comp_q_, comp_k_.transpose(1, 2)
            )
            # (N * h, L, S) -> (N * h, L, S)
            denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
            # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
            attn_weights = weights / denom
            # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
            attn_output = torch.bmm(attn_weights, v)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )

        # L, N, E
        if self.linear_out:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class XNorCosLogPosAttention(nn.Module):
    """
    XNor with  attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        q_ = q
        comp_q_ = 1 - q_
        k_ = k
        comp_k_ = 1 - k_
        # q_ = torch.cat(
        #     [
        #         q * torch.sin(weight_index[:, :tgt_len, :] / m),
        #         q * torch.cos(weight_index[:, :tgt_len, :] / m),
        #     ],
        #     dim=-1,
        # )
        # (N * h, S, 2 * d)
        # comp_q_ = torch.cat(
        #     [
        #         comp_q * torch.sin(weight_index[:, :tgt_len, :] / m),
        #         comp_q * torch.cos(weight_index[:, :tgt_len, :] / m),
        #     ],
        #     dim=-1,
        # )
        # k_ = torch.cat(
        #     [
        #         k * torch.sin(weight_index[:, :src_len, :] / m),
        #         k * torch.cos(weight_index[:, :src_len, :] / m),
        #     ],
        #     dim=-1,
        # )
        # comp_k_ = torch.cat(
        #     [
        #         comp_k * torch.sin(weight_index[:, :src_len, :] / m),
        #         comp_k * torch.cos(weight_index[:, :src_len, :] / m),
        #     ],
        #     dim=-1,
        # )

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            weights = 0.5 * (
                torch.bmm(q_, k_.transpose(1, 2))
                + torch.bmm(comp_q_, comp_k_.transpose(1, 2))
            )
            # (N * h, L, S) -> (N * h, L, S)
            denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
            # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
            attn_weights = weights / denom

            # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
            attn_output = torch.bmm(attn_weights, v)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
            # L, N, E
            if self.has_outproj:
                attn_output = self.out_proj(attn_output)

        # L, N, E
        if self.linear_out:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class SoftmaxAttention(nn.Module):
    """
    Softmax attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    # def get_index(self, seq_len):
    #     index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

    #     return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        elif act_fun == "sig":
            return F.sigmoid
        elif act_fun == "swish":
            return F.silu

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # n_batch_pos = pos_emb.size(0)
        # p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        # p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)

    def left_product(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        # test for the correctness of the program
        if key == None:
            key = query
        if value == None:
            value = query

        n_head = self.n_head
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // n_head

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.act_fun(q.view(tgt_len, bsz, n_head, head_dim))
        k = self.act_fun(k.view(tgt_len, bsz, n_head, head_dim))

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        q_ = q
        k_ = k
        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            weights = weights.masked_fill(attn_mask == float("-inf"), 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output


class ModifiedSoftmaxAttention(nn.Module):
    """
    Softmax attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        elif act_fun == "sig":
            return F.sigmoid
        elif act_fun == "swish":
            return F.silu

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=0)

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # n_batch_pos = pos_emb.size(0)
        # p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        # p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)

    def left_product(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        # test for the correctness of the program
        if key == None:
            key = query
        if value == None:
            value = query

        n_head = self.n_head
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // n_head

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.act_fun(q.view(tgt_len, bsz, n_head, head_dim))
        k = self.act_fun(k.view(tgt_len, bsz, n_head, head_dim))

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        q_ = q
        k_ = k
        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            weights = weights.masked_fill(attn_mask == float("-inf"), 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output


class XNorAttention(nn.Module):
    """
    XNorm attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        elif act_fun == "sig":
            return F.sigmoid
        elif act_fun == "swish":
            return F.silu

    @staticmethod
    def apply_rotary_position_embeddings(
        sinusoidal_pos, query_layer, key_layer, value_layer=None
    ):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack(
            [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
        ).reshape_as(query_layer)
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack(
            [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
        ).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack(
                [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
            ).reshape_as(value_layer)
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # n_batch_pos = pos_emb.size(0)
        # p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        # p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            comp_z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output += attn_output2
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)

    def left_product(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        # test for the correctness of the program
        if key == None:
            key = query
        if value == None:
            value = query

        n_head = self.n_head
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // n_head

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.act_fun(q.view(tgt_len, bsz, n_head, head_dim))
        k = self.act_fun(k.view(tgt_len, bsz, n_head, head_dim))

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        q_ = q
        k_ = k
        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            weights = weights.masked_fill(attn_mask == float("-inf"), 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output


class XNorNonormAttention(XNorAttention):
    """
    XNorm attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        elif act_fun == "sig":
            return F.sigmoid
        elif act_fun == "swish":
            return F.silu

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # n_batch_pos = pos_emb.size(0)
        # p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        # p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = torch.ones((bsz * n_head, tgt_len)).to(
                kv_.device
            )  # / torch.clamp_min(
            #     torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            # )
            comp_z_ = torch.ones((bsz * n_head, tgt_len)).to(kv_.device)
            #  / torch.clamp_min(
            #     torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            # )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output += attn_output2
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class ModifiedXNorAttention(XNorAttention):
    """
    XNorm attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        elif act_fun == "sig":
            return F.sigmoid
        elif act_fun == "swish":
            return F.silu

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=0)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            ) + torch.clamp_min(
                torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_ + comp_kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            # attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            # attn_output += attn_output2
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class XNorRelPosAttention(XNorAttention):
    """
    XNorm attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(n_head, n_feat // n_head))
        self.pos_bias_v = nn.Parameter(torch.Tensor(n_head, n_feat // n_head))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        nbatch_pos, _, embed_dim = pos_emb.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)
        # logging.info(f"POS EMB Shape {pos_emb.shape} QUERY {query.shape}")
        pos_emb = pos_emb.view(-1, n_batch_pos, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        ## REL POS EMB
        n_batch_pos = pos_emb.size(0)

        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, head_dim)
        p_ = torch.nn.functional(p, dim=-1)
        comp_p_ = 1 - p
        p_ = p_.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)
        comp_p_ = comp_p_.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        q_reshaped_ = q.transpose(0, 1).view(
            bsz, tgt_len, self.n_head, embed_dim // self.n_head
        )  ## (batch,seqlen,n_head,emb_dim)
        q_with_bias_u = (q_reshaped_ + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q_reshaped_ + self.pos_bias_v).transpose(1, 2)

        q_comp_with_bias_u = (1 - q_reshaped_ + self.pos_bias_u).transpose(1, 2)
        q_comp_with_bias_v = (1 - q_reshaped_ + self.pos_bias_v).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_u_ = torch.nn.functional.softmax(q_with_bias_u, dim=-1).view(
            bsz * self.n_head, tgt_len, embed_dim // self.n_head
        )
        # (batch, head, time1, d_k)
        q_with_bias_v_ = torch.nn.functional.softmax(q_with_bias_v, dim=-1).view(
            bsz * self.n_head, tgt_len, embed_dim // self.n_head
        )

        q_comp_with_bias_u_ = torch.nn.functional.softmax(
            q_comp_with_bias_u, dim=-1
        ).view(bsz * self.n_head, tgt_len, embed_dim // self.n_head)
        # (batch, head, time1, d_k)
        q_comp_with_bias_v_ = torch.nn.functional.softmax(
            q_comp_with_bias_v, dim=-1
        ).view(bsz * self.n_head, tgt_len, embed_dim // self.n_head)

        # # (batch, head, time1, 2*time1-1)
        # matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # matrix_bd = self.rel_shift(matrix_bd)

        k = torch.nn.functional.softmax(
            k.contiguous().view(bsz, src_len, self.n_head, embed_dim // self.n_head),
            dim=-1,
        )
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_with_bias_u_ = (
            q_with_bias_u_.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        )
        q_comp_with_bias_u_ = (
            q_comp_with_bias_u_.contiguous()
            .view(-1, bsz * n_head, head_dim)
            .transpose(0, 1)
        )
        q_comp_with_bias_v_ = (
            q_comp_with_bias_v_.contiguous()
            .view(-1, bsz * n_head, head_dim)
            .transpose(0, 1)
        )
        q_with_bias_v_ = (
            q_with_bias_v_.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        )

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)

            # (N * h, S, d) (N * h, 2S-1, d) -> (N * h, 2 * d, d)
            rv_ = torch.einsum("nld,nlm->ndm", k_, p_)
            comp_rv_ = torch.einsum("nld,nlm->ndm", comp_k_, comp_p_)
            logging.info(
                f"P {p_.shape} K {k_.shape} V {v.shape} QU {q_with_bias_u_.shape} QV {q_comp_with_bias_v_}"
            )
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_1_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_with_bias_u_, torch.sum(k_, axis=1)), eps
            )
            comp_z_1_ = 1 / torch.clamp_min(
                torch.einsum(
                    "nld,nd->nl", q_comp_with_bias_u_, torch.sum(comp_k_, axis=1)
                ),
                eps,
            )
            z_2 = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_with_bias_v_, torch.sum(p_, axis=1)), eps
            )
            comp_z_2 = 1 / torch.clamp_min(
                torch.einsum(
                    "nld,nd->nl", q_comp_with_bias_v_, torch.sum(comp_p_, axis=1)
                ),
                eps,
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output1 = torch.einsum("nld,ndm,nl->nlm", q_with_bias_u_, kv_, z_1_)
            comp_attn_output1 = torch.einsum(
                "nld,ndm,nl->nlm", q_comp_with_bias_u_, comp_kv_, comp_z_1_
            )
            attn_output2 = self.rel_shift(
                torch.einsum("nld,ndm,nl->nlm", q_with_bias_v_, rv_, z_2)
            )
            comp_attn_output2 = self.rel_shift(
                torch.einsum("nld,ndm,nl->nlm", q_comp_with_bias_v_, comp_rv_, comp_z_2)
            )

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output = (
                attn_output1 + attn_output2 + comp_attn_output1 + comp_attn_output2
            )
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)

    def left_product(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        # test for the correctness of the program
        if key == None:
            key = query
        if value == None:
            value = query

        n_head = self.n_head
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // n_head

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        q_ = q
        k_ = k
        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            weights = weights.masked_fill(attn_mask == float("-inf"), 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output


class XNorReNormAttention(XNorAttention):
    """
    XNorm attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        elif act_fun == "sig":
            return F.sigmoid
        elif act_fun == "swish":
            return F.silu

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                (
                    torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1))
                    + torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1))
                ),
                eps,
            )

            # comp_z_ = 1 / torch.clamp_min(
            #     torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            # )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, z_)
            attn_output += attn_output2
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class XNorRopeAttention(XNorAttention):
    """
    XNor Rope attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal
        # ROPE Encoding
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            8192, n_feat // n_head
        )

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        sinusoidal_pos = self.embed_positions(
            query.permute(1, 0, 2).view(bsz, -1, n_head, head_dim).shape[:-1]
        )[None, None, :, :]
        # logging.info(f"POSENC {sinusoidal_pos.shape}")
        ## ROPE: Needs (bs,num_heads,seq_len,per_head_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k
        q, k = self.apply_rotary_position_embeddings(
            sinusoidal_pos, q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3)
        )
        comp_q, comp_k = self.apply_rotary_position_embeddings(
            sinusoidal_pos, comp_q.permute(1, 2, 0, 3), comp_k.permute(1, 2, 0, 3)
        )
        q, k = q.permute(2, 0, 1, 3), k.permute(2, 0, 1, 3)
        comp_q, comp_k = comp_q.permute(2, 0, 1, 3), comp_k.permute(2, 0, 1, 3)

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            comp_z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output += attn_output2
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)
