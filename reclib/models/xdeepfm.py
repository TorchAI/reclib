from typing import Optional, Union, List

import torch

from reclib.modules import FeedForward
from reclib.modules.embedders import LinearEmbedder, Embedding
from reclib.modules import CompressedInteractionNetwork


class ExtremeDeepFactorizationMachine(torch.nn.Module):
    """
    Parameters
    ----------
    field_dims: ``List``
        List of sizes of each field (size here means the number of unique items in a field)
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
                 cross_layer_sizes: List[int],
                 mlp_dims: Union[int, List[int]],
                 dropout: Optional[Union[float, List[float]]] = 0.0,
                 split_half: bool = True):
        super().__init__()
        self.linear = LinearEmbedder(field_dims, 1)
        self.embedding = Embedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(
            len(field_dims), cross_layer_sizes, split_half)

        if isinstance(mlp_dims, int):
            mlp_layers = 1
        else:
            mlp_layers = len(mlp_dims)

        # + 1 to include the final layer where the output dim is 1
        self.mlp = FeedForward(num_layers=mlp_layers,
                               input_dim=self.embed_output_dim,
                               hidden_dims=mlp_dims,
                               batch_norm=True,
                               activations=torch.nn.ReLU(),
                               dropout=dropout)
        # We need to separate cuz output layer doesn't have batch norm
        self.output_layer = torch.nn.Linear(mlp_dims[-1], 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Long tensor of size ``(batch_size, num_fields)``
        Returns
        ----------
        output: ``(batch_size,)``
        """
        # ``(batch_size, num_fields, embed_dim)
        embed_x = self.embedding(x)
        # ``(batch_size, 1)``
        tmp = self.linear(x) + self.cin(embed_x) + \
              self.output_layer(self.mlp(embed_x.view(-1, self.embed_output_dim)))
        output = torch.sigmoid(tmp.squeeze(1))
        return output
