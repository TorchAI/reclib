import torch
from reclib.modules.embedders import Linear_Embedder, Embedding

from reclib.modules.layers import   CrossNetwork, MultiLayerPerceptron


class DeepCrossNetwork(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Parameters
    ----------
    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embedding = Embedding(field_dims, embed_dim)
        self.linear = Linear_Embedder(field_dims)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.cn_output = torch.nn.Linear(self.embed_output_dim, 1)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        cross_term = self.cn(embed_x)
        x = self.linear(x) + self.cn_output(cross_term) + self.mlp(embed_x)
        return torch.sigmoid(x.squeeze(1))
