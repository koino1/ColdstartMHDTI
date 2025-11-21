import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from model.base_MHGNN_dti import MAGNN_ctr_ntype_specific


def adj_normalize(adj):
    rowsum = adj.sum(1)
    r_inv = torch.pow(rowsum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    adj_ = r_mat_inv * adj * r_mat_inv
    return adj_


def MinMax_scalar(x):
    min = x.min(1).values
    min = min.view(-1, 1).repeat(1, x.shape[1])
    max = x.max(1).values
    max = max.view(-1, 1).repeat(1, x.shape[1])
    scalar = (x - min) / (max - min)
    return scalar


def normalize(x):
    rowsum = x.sum(1)
    rowsum = rowsum.view(-1, 1).repeat(1, x.shape[1])
    x_norm = x / rowsum
    return x_norm


class TypeTower(nn.Module):
    """
    Linear -> GELU -> Dropout -> Linear -> Dropout
    -> 残差到第一次投影 -> LayerNorm
    """
    def __init__(self, in_dim, hid, p=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid, bias=True)
        self.fc2  = nn.Linear(hid, hid, bias=True)
        self.drop = nn.Dropout(p)
        self.ln   = nn.LayerNorm(hid)
        nn.init.xavier_normal_(self.proj.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight,  gain=1.414)

    def forward(self, x):
        z = self.proj(x)      # 第一次把各类型对齐到 hidden_dim
        h = F.gelu(z)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return self.ln(h + z) # 残差到首投影，再做 LayerNorm（Post-LN）

# for link prediction task
class MAGNN_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.2,
                 attn_switch=False,
                 args=None):
        super(MAGNN_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.user_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[0],
                                                   etypes_lists[0],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True,
                                                   attn_switch=attn_switch,
                                                   args=args)
        self.item_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[1],
                                                   etypes_lists[1],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True,
                                                   attn_switch=attn_switch,
                                                   args=args)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        # self.fc_user = nn.Linear(in_dim * num_heads * num_metapaths_list[0], out_dim * num_heads, bias=True)
        # self.fc_item = nn.Linear(in_dim * num_heads * num_metapaths_list[1], out_dim * num_heads, bias=True)
        # nn.init.xavier_normal_(self.fc_user.weight, gain=1.414)
        # nn.init.xavier_normal_(self.fc_item.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ctr_ntype-specific layers
        h_user = self.user_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0]))
        h_item = self.item_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1]))

        return [h_user, h_item]

        # logits_user = self.fc_user(h_user)
        # logits_item = self.fc_item(h_item)
        # return [logits_user, logits_item], [h_user, h_item]


#
# class GCN_layer(nn.Module):
#     def __init__(self, dim, mp_ls=None):
#         super().__init__()
#         self.gcn1 = nn.Parameter(torch.zeros([dim, 128]), requires_grad=True)
#         self.gcn2 = nn.Parameter(torch.zeros([128, 2]), requires_grad=True)
#         nn.init.xavier_normal_(self.gcn1, gain=1.414)
#         nn.init.xavier_normal_(self.gcn2, gain=1.414)
#
#     # --- 与旧代码完全一致的接口与行为：返回 logits（2 维） ---
#     def forward(self, x, adj):
#
#         # 第一层（到 128 维）
#         adj = F.softmax(torch.matmul(x, x.T), dim=-1)
#         x = F.relu(torch.matmul(torch.matmul(adj, x), self.gcn1))
#         x = torch.matmul(torch.matmul(adj, x), self.gcn2)
#
#         return x
#
#
#     # --- 新增：不改动旧接口，给需要的人取“联合嵌入” ---
#     def forward_with_feat(self, x, adj):
#         # 与 forward 完全一致的邻接与计算
#         A = F.softmax(torch.matmul(x, x.T), dim=-1)
#         h = torch.matmul(torch.matmul(A, x), self.gcn1)  # 128维（第一步线性前ReLU）
#         h = F.relu(h)
#         z = torch.matmul(A, h)                           # 128维（第二次传播，推荐作为联合嵌入）
#         logits = torch.matmul(z, self.gcn2)              # 2维输出
#         return logits, z, h


class linear_module(nn.Module):
    def __init__(self, dim):
        super(linear_module, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim / 2), bias=True)
        self.fc2 = nn.Linear(int(dim / 2), 2, bias=False)
        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, x, x2=None):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def forward_with_feat(self, x, adj):
        # 与 forward 完全一致的邻接与计算
        A = F.softmax(torch.matmul(x, x.T), dim=-1)
        h = torch.matmul(torch.matmul(A, x), self.gcn1)  # 128维（第一步线性前ReLU）
        h = F.relu(h)
        z = torch.matmul(A, h)  # 128维（第二次传播，推荐作为联合嵌入）
        logits = torch.matmul(z, self.gcn2)  # 2维输出
        return logits, z, h


class ProductMLP(nn.Module):
    """
    x = h_user ⊙ h_item -> MLP -> 2 logits
    与现有管线兼容：forward() 返回 2 维 logits，外面再 softmax。
    """

    def __init__(self, dim, hidden=None):
        super().__init__()
        h = hidden or max(128, dim // 2)
        self.fc1 = nn.Linear(dim, h, bias=True)
        self.fc2 = nn.Linear(h, 2, bias=False)
        # init
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, h_user, h_item, adj=None):
        x = h_user * h_item  # [B, dim]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # [B, 2]  logits
        return x

    def forward_with_feat(self, h_user, h_item, adj=None):
        x = h_user * h_item
        h = F.relu(self.fc1(x))  # 倒数第二层特征，可用于可视化
        logits = self.fc2(h)
        return logits, h


class MAGNN_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 dropout_rate=0.2,
                 attn_switch=False,
                 args=None):
        super(MAGNN_lp, self).__init__()
        self.hidden_dim = hidden_dim
        self.args = args

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([
            TypeTower(feats_dim, hidden_dim, p=dropout_rate) for feats_dim in feats_dim_list
        ])
        # <<< ADDED

        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(0.1)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers


        # MAGNN_lp layers
        self.layer1 = MAGNN_lp_layer(num_metapaths_list,
                                     num_edge_type,
                                     etypes_lists,
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     rnn_type,
                                     attn_drop=dropout_rate,
                                     attn_switch=attn_switch,
                                     args=args)
        dim = out_dim * num_heads * 2
        if self.args.semantic_fusion == 'concatenation':
            dim = out_dim * num_heads * (num_metapaths_list[0] + num_metapaths_list[1])
        # predictor
        if self.args.predictor == 'product_mlp':
            feat_dim = out_dim * num_heads  # h_user 的维度
            self.classifier = ProductMLP(dim=feat_dim)
        elif self.args.predictor == 'linear':
            self.classifier = linear_module(dim=dim)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists, adj = inputs

        # ntype-specific transformation
        device = features_list[0].device
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=device)
        for i, tower in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = tower(features_list[i])  # 小塔已含激活/残差/LN/dropout
        # 仍可保留全局特征 dropout（建议稍小，例如 0.1~0.2）
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        # [logits_user, logits_item], [h_user, h_item] = self.layer1(
        #     (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))
        [h_user, h_item] = self.layer1(
            (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))
        x_out = self.classifier(h_user, h_item, adj)

        return F.softmax(x_out, dim=-1)

    ### 这是新增为了保存药物-靶标嵌入的
    def encode_batch(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists, adj = inputs

        device = features_list[0].device
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=device)
        for i, tower in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = tower(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        # 你的 layer1 保持不变
        h_user, h_item = self.layer1(
            (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists)
        )

        logits, z_joint = self.classifier.forward_with_feat(h_user, h_item, adj=None)  # 返回: [B], [B,hidden] 或 [B,k]

        # 2) 兼容你原本的返回签名（第三个位置需要 x_cat）
        x_cat = torch.cat([h_user, h_item], dim=1)

        return (
            h_user.detach(),
            h_item.detach(),
            x_cat.detach(),  # 占原来的 x_cat 位
            z_joint.detach(),  # penultimate/中间特征
            logits.detach()  # logit/prob 用于可视化/阈值
        )