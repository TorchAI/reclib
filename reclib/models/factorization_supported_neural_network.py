import torch

from reclib.modules import FeedForward
from reclib.modules.embedders import Embedding


class FactorizationSupportedNeuralNetwork(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Parameters
    ----------
    Reference:
        W Zhang, et al. Deep Learning over Multi-field Categorical Data - A Case Study on User Response Prediction, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
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
        Parameters
        ----------
        x: torch.LongTensor
            Shape of ``(batch_size, num_fields)``

        Returns
        -------
        label_logits:
            A tensor of shape ``(batch_size, num_labels)`` representing un-normalised log
            probabilities of the label.
        """
        embed_x = self.embedding(x)
        x = self.mlp(embed_x.view(-1, self.embed_output_dim))
        label_logits = torch.sigmoid(x.squeeze(1))
        return label_logits
