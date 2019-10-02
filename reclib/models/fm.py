import torch

from reclib.modules.embedders import LinearEmbedder, Embedding
from reclib.modules import FactorizationMachine


class FactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine. `"Factorization Machines"<https://ieeexplore.ieee.org/abstract/document/5694074>`_
    by S Rendle., 2010.        
    """
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = Embedding(field_dims, embed_dim)
        self.linear = LinearEmbedder(field_dims, 1)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
