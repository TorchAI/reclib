import torch

from reclib.modules import AttentionalFactorizationLayer
from reclib.modules.embedders import LinearEmbedder, Embedding


class AttentionalFactorizationMachine(torch.nn.Module):
    """
    This implements Attentional Factorization Machine from
    '"Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks"
    <https://arxiv.org/abs/1708.04617>' by J Xiao, et al. , 2017.
    """

    def __init__(self, field_dims, embed_dim, attn_size, dropouts):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding = Embedding(field_dims, embed_dim)
        self.linear = LinearEmbedder(field_dims, 1)
        self.afm = AttentionalFactorizationLayer(embed_dim, attn_size, dropouts)

    def forward(self, x):
        x = self.linear(x) + self.afm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
