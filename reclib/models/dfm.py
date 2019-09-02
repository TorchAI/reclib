import torch
from reclib.modules.embedders import Linear_Embedder, Embedding

from reclib.modules.layers import FactorizationMachine,   MultiLayerPerceptron


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
        self.linear = Linear_Embedder(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = Embedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims,
                                        dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(
            embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
