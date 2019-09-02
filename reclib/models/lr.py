import torch

from reclib.modules.embedders import Linear_Embedder


class LogisticRegression(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    Parameters
    ----------
    """

    def __init__(self, field_dims):
        super().__init__()
        self.linear = Linear_Embedder(field_dims, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sigmoid(self.linear(x).squeeze(1))
