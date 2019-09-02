import torch

from reclib.modules import FeedForward
from reclib.modules.embedders import Embedding
from reclib.modules.layers import FactorizationMachine


class NeuralFactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Parameters
    ----------
    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.embedding = Embedding(field_dims, embed_dim)
        self.linear = LogisticRegressionModel(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = FeedForward(2,
                               self.embed_dim,
                               [mlp_dims, 1],
                               True,
                               ['relu', 'linear'],
                               [dropouts[1], 0])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        cross_term = self.fm(self.embedding(x))
        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))
