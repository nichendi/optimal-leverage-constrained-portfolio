import torch
import torch.nn as nn


class LongOnlyNN(nn.Module):
    """
    A two-layer neural network for long-only portfolio optimization.
    Proposed in Ni, C., Li, Y., Forsyth, P., & Carroll, R. (2022).
    Optimal asset allocation for outperforming a stochastic benchmark target.
    Quantitative Finance, 22(9), 1595-1626.
    """

    def __init__(self, D_in, H, D_out, is_random_initial_weight=True):
        """
        D_in: input dimension
        H: dimension of hidden layer
        D_out: output dimension
        """
        super(LongOnlyNN, self).__init__()
        self.linear1 = nn.Linear(D_in, H, bias=False)
        self.linear2 = nn.Linear(H, D_out, bias=False)
        if is_random_initial_weight:
            layer1_weights = torch.rand(H, D_in)
            layer2_weights = torch.rand(D_out, H)
        else:
            # initialize weights to zero
            layer1_weights = torch.zeros(H, D_in)
            layer2_weights = torch.zeros(D_out, H)
        self.linear1.weight = torch.nn.Parameter(layer1_weights)
        self.linear2.weight = torch.nn.Parameter(layer2_weights)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.sigmoid(-self.linear1(x))
        p = self.softmax(self.linear2(hidden))
        return p


class FFN(nn.Module):
    """A simple positionwise feed-forward block with sigmoid activation"""

    def __init__(self, num_hidden_nodes, num_output_nodes):
        super().__init__()
        self.dense1 = nn.LazyLinear(num_hidden_nodes)
        self.sigmoid = nn.Sigmoid()
        self.dense2 = nn.LazyLinear(num_output_nodes)

    def forward(self, X):
        return self.dense2(self.sigmoid(self.dense1(X)))


class RCNN(nn.Module):
    """Relaxed-constraint neural network (RCNN) proposed in the paper.
    The network is designed to approximate the optimal control of a leverage-constrained portfolio.
    """

    def __init__(
        self,
        num_assets: int,
        num_hidden_nodes: int,
        p_max: float,
    ):
        """
        num_assets: number of assets
        p_max: maximum total long position
        num_hidden_nodes: number of hidden nodes in the FFN block
        """
        super(RCNN, self).__init__()
        self.p_max = p_max
        self.blks = nn.Sequential()
        # We find a single FFN is sufficient for our paper. However, you can always increase the depth of the network by adding more FFN blocks.
        self.blks.add_module("FFN", FFN(num_hidden_nodes, num_assets - 1))
        # set up mapping of assets to their categories
        self.num_assets = num_assets
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.clone()
        # FFN
        for blk in self.blks:
            h = blk(h)
        # bounded mapping
        u = (1 - self.p_max) + (2 * self.p_max - 1) * self.sigmoid(h)
        # extension mapping
        p = torch.cat([u, 1 - torch.sum(u, dim=1, keepdim=True)], dim=1)
        # scaling mapping
        v = p.clamp(min=0).sum(dim=1, keepdim=True)
        l1 = (self.p_max / v).clamp(max=1)
        # subtracts an additional 1e-6 to avoid division by zero in computation
        l2 = ((1 - self.p_max) / (1 - v - 1e-6)).clamp(max=1)
        p = p.clamp(min=0) * l1 + p.clamp(max=0) * l2

        return p
