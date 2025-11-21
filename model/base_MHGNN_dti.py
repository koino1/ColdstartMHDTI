import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax


class MAGNN_metapath_specific(nn.Module):
    def __init__(self,
                 etypes,
                 out_dim,
                 num_heads,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 alpha=0.01,
                 use_minibatch=False,
                 attn_switch=False):
        super(MAGNN_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rnn_type = rnn_type
        self.etypes = etypes
        self.r_vec = r_vec
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch
        ##多头注意力的投影
        self.avg_to_heads = nn.Linear(out_dim, num_heads * out_dim, bias=False)
        self.inst_to_heads = nn.Linear(out_dim, num_heads * out_dim, bias=False)
        nn.init.xavier_uniform_(self.avg_to_heads.weight)  # 可选：初始化
        nn.init.xavier_uniform_(self.inst_to_heads.weight)

        self.max_seq = 64
        self.pos_emb = nn.Embedding(self.max_seq, self.out_dim)
        # （可选）轻量 dropout 稳定训练
        self.pos_drop = nn.Dropout(0.1)
        self.self_mha = nn.MultiheadAttention(self.out_dim, self.num_heads, batch_first=False)
        self.cross_mha = nn.MultiheadAttention(self.out_dim, self.num_heads, batch_first=False)

        # 可选：自注意力后做轻量规范化与前馈（更稳）
        self.sa_ln1 = nn.LayerNorm(self.out_dim)  # self-attn 前
        self.sa_ln2 = nn.LayerNorm(self.out_dim)
        self.sa_ff = nn.Sequential(
            nn.Linear(self.out_dim, 4 * self.out_dim),
            nn.GELU(),
            nn.Linear(4 * self.out_dim, self.out_dim),
        )
        self.ca_ln1 = nn.LayerNorm(self.out_dim)  # cross-attn 前（对 Query 流）
        self.ca_ln2 = nn.LayerNorm(self.out_dim)
        # 也可加个小前馈
        self.ca_ff = nn.Sequential(
            nn.Linear(self.out_dim, 4 * self.out_dim),
            nn.GELU(),
            nn.Linear(4 * self.out_dim, self.out_dim),
        )




        # node-level attention
        # attention considers the center node embedding or not

        self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        # weight initialization

        nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, inputs):
        # features: num_all_nodes x out_dim
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx = inputs
        else:
            g, features, type_mask, edge_metapath_indices = inputs

        g = g.to(features.device)

        # Embedding layer
        # use torch.nn.functional.embedding or torch.embedding here
        # do not use torch.nn.embedding
        # edata: E x Seq x out_dim
        # 约定：pad_id 就是追加的这行的行号
        if not hasattr(self, "pad_id"):
            self.pad_id = features.size(0)  # 真实节点总数 = PAD 的新 id

        # 若还没追加过 PAD 行，就在末尾追加一行全 0（只追加一次）
        if features.size(0) == self.pad_id:
            pad_row = torch.zeros(1, features.size(1), device=features.device, dtype=features.dtype)
            features = torch.cat([features, pad_row], dim=0)
        edata = F.embedding(edge_metapath_indices, features)  ## node features on each metapath


        # apply rnn to metapath-based feature sequence
        if self.rnn_type == 'average':
            hidden = torch.mean(edata, dim=1)  # [E, d]
            hidden = self.avg_to_heads(hidden)  # [E, H*d]
            hidden = hidden.unsqueeze(0)  # [1, E, H*d]
        elif self.rnn_type == 'self-attn':
            key_padding_mask = (edge_metapath_indices == self.pad_id)  # True=PAD

            edata = F.embedding(edge_metapath_indices, features)  # [E, Seq, out_dim]
            E, Seq, d = edata.shape
            tokens = edata.permute(1, 0, 2)  # [Seq, E, d]
            pos_ids = torch.arange(Seq, device=tokens.device)
            pos_vec = self.pos_emb(pos_ids)  # [Seq, d]
            pos_vec = pos_vec.unsqueeze(1).expand(Seq, E, d)  # [Seq, E, d]
            # --- 把 PAD 位置的pos置零（与 key_padding_mask 对齐）
            # 现成的 key_padding_mask 是 [E, Seq]，转置到 [Seq, E]
            pad_mask_T = key_padding_mask.transpose(0, 1)  # [Seq, E]
            pos_vec = pos_vec.masked_fill(pad_mask_T.unsqueeze(-1), 0.0)

            # --- 加到 tokens 上
            # ① Self-Attention：先 LN 再 MHA，然后残差
            tokens = self.pos_drop(tokens + pos_vec)  # [Seq, E, d]
            sa_in = self.sa_ln1(tokens)  # [Seq,E,d]
            sa_out, _ = self.self_mha(sa_in, sa_in, sa_in, key_padding_mask=key_padding_mask)
            tokens = tokens + sa_out
            # ② 从 tokens 读出实例向量（掩码平均）
            sa_out = tokens.permute(1, 0, 2)
            valid = (~key_padding_mask).float()
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            inst_emb = (sa_out * valid.unsqueeze(-1)).sum(dim=1) / denom  # [E,d]
            # ③ FFN：先 LN 再 FFN，然后残差（不再跟 LN）
            inst_in = self.sa_ln2(inst_emb)
            inst_emb = inst_emb + self.sa_ff(inst_in)

            # --- 计算每条实例的“最后真实位置”的 target 节点 id
            lengths = (~key_padding_mask).sum(dim=1)  # [E]
            last_pos = (lengths - 1).clamp_min(0)  # [E]
            last_ids = edge_metapath_indices.gather(1, last_pos.view(-1, 1)).squeeze(1)  # [E]

            # target 节点向量（Query）
            target_feat = F.embedding(last_ids, features)
            # Cross-Attention: Q = target_feat (len=1), K/V = sa_out 的序列
            q = self.ca_ln1(target_feat).unsqueeze(0)  # [1, E, d]  先LN(Pre-LN)
            k = sa_out.permute(1, 0, 2)  # [Seq, E, d]
            v = k
            cross_out, _ = self.cross_mha(q, k, v, key_padding_mask=key_padding_mask)
            inst_emb = inst_emb + cross_out.squeeze(0)  # 残差，不再跟LN
            # FFN 前再LN一次（Pre-LN）
            inst_emb = inst_emb + self.ca_ff(self.ca_ln2(inst_emb))

            hidden = self.inst_to_heads(inst_emb).unsqueeze(0)     # [1, E, H*d]






        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim
        a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
        a = self.leaky_relu(a)
        g.edata.update({'eft': eft, 'a': a})
        # compute softmax normalized attention values
        self.edge_softmax(g) # Compute softmax over weights of incoming edges for every node.
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft')) # Send messages along all the edges of the specified type and update all the nodes of the corresponding destination type
        ret = g.ndata['ft']  # E x num_heads x out_dim

        if self.use_minibatch:
            return ret[target_idx]
        else:
            return ret         #得到的是一个batch内，同一种元路径（比如0-1-1-0）的中心节点的特征，其中E是元路径实例的个数，H是头数，dim是特征的维度


class MAGNN_ctr_ntype_specific(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 use_minibatch=False,
                 attn_switch=False,
                 args=None):
        super(MAGNN_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch
        self.args = args

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(MAGNN_metapath_specific(etypes_list[i],
                                                                out_dim,
                                                                num_heads,
                                                                rnn_type,
                                                                r_vec,
                                                                attn_drop=attn_drop,
                                                                use_minibatch=use_minibatch,
                                                                attn_switch=attn_switch))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        if self.args.semantic_fusion == 'attention':
            self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
            self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
            # weight initialization
            nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
            nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, target_idx, metapath_layer in zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
        else:
            g_list, features, type_mask, edge_metapath_indices_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, metapath_layer in zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        if self.args.semantic_fusion == 'attention':
            beta = []
            for metapath_out in metapath_outs:
                fc1 = torch.tanh(self.fc1(metapath_out))
                fc1_mean = torch.mean(fc1, dim=0)  # metapath specific vector
                fc2 = self.fc2(fc1_mean)  # metapath importance
                beta.append(fc2)
            beta = torch.cat(beta, dim=0)
            beta = F.softmax(beta, dim=0)
            beta = torch.unsqueeze(beta, dim=-1)
            beta = torch.unsqueeze(beta, dim=-1)
            metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
            metapath_outs = torch.cat(metapath_outs, dim=0)
            h = torch.sum(beta * metapath_outs, dim=0)
        elif self.args.semantic_fusion == 'average':
            h = torch.mean(torch.stack(metapath_outs, dim=0), dim=0)
        elif self.args.semantic_fusion == 'max-pooling':
            h, _ = torch.max(torch.stack(metapath_outs, dim=0), dim=0)
        elif self.args.semantic_fusion == 'concatenation':
            h = torch.cat(metapath_outs, dim=1)
        return h
