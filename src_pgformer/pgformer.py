"""PGFormer: routing-oriented performance predictor for proximity-graph configuration.

Implements the dual-branch architecture described in "FedTopoTuner: A Federated
Topology-Guided Tuning Framework for Automated Proximity Graph Configuration":

  * Global branch -- scalable all-pair message passing (NodeFormer) that captures
                    global navigability via a Kernelized Gumbel-Softmax operator over
                    positive random features (Sec. 3.2.1).
  * Local branch  -- geometry-preserving message passing (EGNN) over a proxy k-NN
                    graph that captures local connectivity (Sec. 3.2.2).

Both branches read the same aligned node embedding produced by the client-specific
dimension adapter, are read out by mean pooling, and are concatenated with the
configuration embedding before a final MLP regressor predicts the retrieval metrics
(Sec. 3.2.3).

The public interface (``pgformer.__init__`` and ``pgformer.forward``) is kept
compatible with the existing training / evaluation code, which calls::

    model = pgformer(d, hidden, c, d_para, num_layers=..., alpha=..., dropout=...,
                     num_heads=..., kernel=..., use_bn=..., use_residual=...,
                     use_graph=..., use_weight=...)
    out = model(node_feat, edge_index, params)   # out: [num_configs, c]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Global branch: Kernelized Gumbel-Softmax all-pair attention (NodeFormer)
# ---------------------------------------------------------------------------

def positive_random_features(x, W_rf):
    """Positive random features (Eq. 4 of the paper).

    phi(x) = exp(-||x||^2 / 2) * [exp(w_1 . x), ..., exp(w_m . x)] / sqrt(m),
    with w_i ~ N(0, I). The random projection ``W_rf`` is shared between the query
    and the key so that E[phi(q) . phi(k)] = exp(q . k).
    """
    x_sq = (x * x).sum(dim=-1, keepdim=True)          # [..., 1]
    proj = torch.einsum("...d,md->...m", x, W_rf)     # [..., m]
    return torch.exp(proj - 0.5 * x_sq) / math.sqrt(W_rf.shape[0])


def sample_gumbel(shape, device):
    """Sample i.i.d. Gumbel(0, 1) noise, g = -log(-log(u)) with u ~ U(0, 1)."""
    u = torch.rand(shape, device=device).clamp_min(1e-9)
    return -torch.log(-torch.log(u))


def kernelized_gumbel_softmax(q, k, v, W_rf, tau=1.0, num_gumbels=4):
    """Linearized all-pair attention with a Kernelized Gumbel-Softmax operator.

    This factorizes the ``exp(q . k / tau)`` softmax kernel of all-pair attention
    into an inner product of positive random features (Eq. 3-4), so the pairwise
    quadratic cost is reduced to linear cost. Following Algorithm 2, the same
    per-node Gumbel noise is folded into BOTH the query and the key random
    features, and ``T`` such samples are averaged to stabilise the estimator.

    Args:
        q, k, v: [N, H, M] query / key / value tensors (N nodes, H heads, M channels).
        W_rf: [m, M] fixed random projection for the positive random features.
        tau: Gumbel-Softmax temperature.
        num_gumbels: number of Gumbel samples to average (T in Algorithm 2).

    Returns:
        out: [N, H, M] attentive aggregation.
    """
    eps = 1e-8
    M = q.shape[-1]

    # Standard attention scaling keeps q . k bounded before the exponential kernel.
    q = q / math.sqrt(max(M, 1))
    k = k / math.sqrt(max(M, 1))

    outputs = []
    for _ in range(num_gumbels):
        # Per-node Gumbel(0, 1) noise shared by query and key (Algorithm 2, lines
        # 5-7). Since both q(θ) and k(θ) are noisy, the key-side aggregates
        # u(θ), o(θ) are recomputed for every sample.
        g = sample_gumbel((q.shape[0], q.shape[1]), q.device).clamp(-5.0, 5.0)
        noise = torch.exp(g / tau).unsqueeze(-1)              # [N, H, 1]

        phi_q = positive_random_features(q, W_rf) * noise     # [N, H, m]
        phi_k = positive_random_features(k, W_rf) * noise     # [N, H, m]

        u = torch.einsum("nhm,nhd->hmd", phi_k, v)            # [H, m, M]
        o = phi_k.sum(dim=0)                                  # [H, m]

        num = torch.einsum("nhm,hmd->nhd", phi_q, u)          # [N, H, M]
        den = torch.einsum("nhm,hm->nh", phi_q, o)            # [N, H]
        outputs.append(num / den.unsqueeze(-1).clamp_min(eps))

    return torch.stack(outputs, dim=0).mean(dim=0)            # [N, H, M]


class NodeFormerConv(nn.Module):
    """One global-branch layer: scalable all-pair message passing."""

    def __init__(self, in_channels, out_channels, num_heads=1,
                 num_gumbels=4, tau=1.0, num_features=None):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_gumbels = num_gumbels
        self.tau = tau

        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        num_features = num_features or out_channels
        # Fixed random projection shared by query and key (Eq. 4). A fixed seed keeps
        # the map identical across the global model and every client copy.
        gen = torch.Generator().manual_seed(0)
        self.register_buffer(
            "W_rf", torch.randn(num_features, out_channels, generator=gen))

    def reset_parameters(self):
        self.Wq.reset_parameters()
        self.Wk.reset_parameters()
        self.Wv.reset_parameters()

    def forward(self, x):
        N = x.shape[0]
        q = self.Wq(x).reshape(N, self.num_heads, self.out_channels)
        k = self.Wk(x).reshape(N, self.num_heads, self.out_channels)
        v = self.Wv(x).reshape(N, self.num_heads, self.out_channels)
        out = kernelized_gumbel_softmax(q, k, v, self.W_rf,
                                        tau=self.tau, num_gumbels=self.num_gumbels)
        return out.mean(dim=1)   # [N, out_channels]


# ---------------------------------------------------------------------------
# Local branch: geometry-preserving message passing on the proxy k-NN graph
# ---------------------------------------------------------------------------

class EGNNLocalConv(nn.Module):
    """One local-branch layer (E(n)-invariant message passing), Eq. 6-7 of the paper.

        m_ij = MLP_edge(h_i, h_j, ||x_i - x_j||^2)
        h_i' = MLP_node(h_i, sum_j m_ij)

    The relative squared distance between raw vectors is fused into the edge features,
    which keeps the local connectivity invariant to rigid transformations.
    """

    def __init__(self, hidden_channels, dropout=0.5):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + 1, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, h, coord, edge_index, edge_weight=None):
        row, col = edge_index
        diff = coord[row] - coord[col]
        radial = (diff * diff).sum(dim=-1, keepdim=True)     # ||x_i - x_j||^2
        edge_feat = torch.cat([h[row], h[col], radial], dim=-1)
        m = self.edge_mlp(edge_feat)                          # Eq. 6
        if edge_weight is not None:
            m = m * edge_weight.view(-1, 1)

        agg = torch.zeros_like(h)
        agg = agg.index_add_(0, row, m)                       # sum_j m_ij
        return self.node_mlp(torch.cat([h, agg], dim=-1))     # Eq. 7


# ---------------------------------------------------------------------------
# PGFormer
# ---------------------------------------------------------------------------

class pgformer(nn.Module):
    """PGFormer: routing-oriented performance predictor (dual-branch)."""

    def __init__(self, in_channels, hidden_channels, out_channels, d_para,
                 num_layers=2, num_heads=1, kernel='simple',
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_graph=True,
                 num_gumbels=4, tau=1.0):
        super().__init__()

        # Client-specific dimension adapter: maps raw vectors into the aligned latent
        # space shared by the federated backbone. Kept local and not aggregated.
        self.dim_adapter = nn.Linear(in_channels, hidden_channels)
        self.adapter_norm = nn.LayerNorm(hidden_channels)

        # Global branch: scalable all-pair message passing (global navigability).
        self.global_convs = nn.ModuleList([
            NodeFormerConv(hidden_channels, hidden_channels, num_heads=num_heads,
                           num_gumbels=num_gumbels, tau=tau)
            for _ in range(num_layers)
        ])
        self.global_bns = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

        # Local branch: geometry-preserving message passing over the proxy k-NN graph.
        self.use_graph = use_graph
        self.local_convs = nn.ModuleList([
            EGNNLocalConv(hidden_channels, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.local_bns = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

        # Prediction head (Sec. 3.2.3): MLP over concat(H_global, H_local, v).
        repr_channels = 2 * hidden_channels if use_graph else hidden_channels
        self.head = nn.Sequential(
            nn.Linear(repr_channels + d_para, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.alpha = alpha

    def reset_parameters(self):
        self.dim_adapter.reset_parameters()
        self.adapter_norm.reset_parameters()
        for conv in self.global_convs:
            conv.reset_parameters()
        for bn in self.global_bns:
            bn.reset_parameters()
        for conv in self.local_convs:
            conv.reset_parameters()
        for bn in self.local_bns:
            bn.reset_parameters()
        for module in self.head:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _run_branch(self, h, convs, bns):
        """Shared forward for a stack of residual+batch-norm'd conv layers."""
        for i, conv in enumerate(convs):
            h_in = h
            h = conv(h)
            if self.use_residual:
                h = self.alpha * h + (1.0 - self.alpha) * h_in
            if self.use_bn:
                h = bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index, params, edge_weight=None):
        # x: [N, in_channels] raw vectors; edge_index: [2, E] proxy k-NN graph;
        # params: [num_configs, d_para] configuration embeddings.

        # 1. Dimension adapter -> aligned node embedding shared by both branches.
        z = self.dim_adapter(x)
        z = self.adapter_norm(z)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # 2. Global branch: all-pair message passing (global navigability).
        h_global = self._run_branch(z, self.global_convs, self.global_bns)

        # 3. Local branch: EGNN over the proxy k-NN graph (local connectivity).
        if self.use_graph:
            h_local = z
            coord = x  # raw vectors used as spatial coordinates (Eq. 6)
            for i, conv in enumerate(self.local_convs):
                h_in = h_local
                h_local = conv(h_local, coord, edge_index, edge_weight)
                if self.use_residual:
                    h_local = self.alpha * h_local + (1.0 - self.alpha) * h_in
                if self.use_bn:
                    h_local = self.local_bns[i](h_local)
                h_local = F.relu(h_local)
                h_local = F.dropout(h_local, p=self.dropout, training=self.training)

        # 4. Mean-pooling readout and concatenation (Eq. 5 / Sec. 3.2.3).
        H_global = h_global.mean(dim=0)                    # [hidden]
        if self.use_graph:
            H_local = h_local.mean(dim=0)                  # [hidden]
            H = torch.cat([H_global, H_local], dim=-1)     # [2 * hidden]
        else:
            H = H_global                                   # [hidden]

        # 5. Concatenate with the configuration embedding and predict (Eq. 8).
        num_configs = params.shape[0]
        H = H.unsqueeze(0).expand(num_configs, -1)         # [num_configs, repr]
        H = torch.cat([H, params], dim=-1)                 # [num_configs, repr + d_para]
        return self.head(H)
