import torch
from reclib.modules.embedders import Linear_Embedder, Embedding

from reclib.modules.layers import FieldAwareFactorizationLayer, MultiLayerPerceptron
from reclib.models import LogisticRegression


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
        self.linear = LogisticRegression(field_dims)
        self.ffm = torch.nn.Sequential(
            FieldAwareFactorizationLayer(field_dims, embed_dim),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.ffm_output_dim = len(field_dims) * (len(field_dims) - 1) // 2 * embed_dim
        self.mlp = MultiLayerPerceptron(self.ffm_output_dim, mlp_dims, dropouts[1])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        cross_term = self.ffm(self.embed_layer(x))
        x = self.linear(x) + self.mlp(cross_term.view(-1, self.ffm_output_dim))
        return torch.sigmoid(x.squeeze(1))
