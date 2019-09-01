import torch

from reclib.modules.layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class FactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Parameters
    ----------
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
