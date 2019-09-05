import torch

from reclib.modules import FeedForward
from reclib.modules.embedders import LinearEmbedder, Embedding
from reclib.modules.layers import FactorizationMachine


class DeepFactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.
    Parameters
    ----------

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = LinearEmbedder(field_dims, 1)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = Embedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = FeedForward(2,
                               self.embed_output_dim,
                               [mlp_dims, 1],
                               True,
                               ['relu', 'linear'],
                               [dropout, 0])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(
            embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
