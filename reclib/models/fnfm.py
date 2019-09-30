import torch

from reclib.models import LogisticRegression
from reclib.modules import FeedForward
from reclib.modules import FieldAwareFactorizationLayer
from reclib.modules.embedders import LinearEmbedder

class FieldAwareNeuralFactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of Field-aware Neural Factorization Machine.
    Parameters
    ----------
    Reference:
        L Zhang, et al. Field-aware Neural Factorization Machine for Click-Through Rate Prediction, 2019.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.linear = LinearEmbedder(field_dims)
        self.ffm = FieldAwareFactorizationLayer(field_dims, embed_dim)
        self.bn = torch.nn.BatchNorm1d(embed_dim),
        self.dp = torch.nn.Dropout(dropouts[0])

        self.ffm_output_dim = len(field_dims) * (len(field_dims) - 1) // 2 * embed_dim
        self.mlp = FeedForward(2,
                               self.ffm_output_dim,
                               [mlp_dims, 1],
                               batch_norm=True,
                               activations=['relu', 'linear'],
                               dropouts=[dropouts[1], 0])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        cross_term = self.ffm(x).view(-1, self.ffm_output_dim)
        cross_term = self.bn(cross_term)
        cross_term = self.dropout(cross_term)
        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))
