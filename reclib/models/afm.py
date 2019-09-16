import torch

from reclib.modules import AttentionalFactorizationLayer
from reclib.modules.embedders import LinearEmbedder, Embedding


class AttentionalFactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of Attentional Factorization Machine.

    Reference:
        J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.
    """

    def __init__(self, field_dims, embed_dim, attn_size, dropouts):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding = Embedding(field_dims, embed_dim)
        self.linear = LinearEmbedder(field_dims, 1)
        self.afm = AttentionalFactorizationLayer(embed_dim, attn_size, dropouts)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.afm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
