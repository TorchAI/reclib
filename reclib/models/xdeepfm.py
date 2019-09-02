from typing import List, Optional

import torch

from reclib.modules.embedders import LinearEmbedder, Embedding
from reclib.modules.layers import CompressedInteractionNetwork


class ExtremeDeepFactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.

    Parameters
    ----------
    field_dims: ``List``
        List of sizes of each field
    embed_dim: ``int``
        The embedding dimension
    mlp_dim: ``int``
        Output dimension for MLP
    cross_layer_sizes: ``int``
    dropout: ``float``, optional (default=``None``)
    split_half: ``bool``


    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.

    """

    def __init__(self,
                 field_dims: List[int],
                 embed_dim: int,
                 mlp_dims: int,
                 cross_layer_sizes: int,
                 dropout: Optional[float] = None,
                 split_half: bool = True):
        super().__init__()
        self.embedding = Embedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(
            len(field_dims), cross_layer_sizes, split_half)

        self.mlp = FeedForward(2,
                               self.embed_output_dim,
                               [mlp_dims, 1],
                               True,
                               ['relu', 'linear'],
                               [dropouts, 0])

        self.linear = LinearEmbedder(field_dims, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + \
            self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
