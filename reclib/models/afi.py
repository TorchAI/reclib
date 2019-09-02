import torch
from reclib.modules.embedders import Linear_Embedder, Embedding
import torch
from reclib.modules.embedders import Linear_Embedder, Embedding.nn.functional as F

from reclib.modules.layers import   MultiLayerPerceptron

class AutomaticFeatureInteraction(torch.nn.Module):
    """
    A pytorch implementation of AutoInt.
    Parameters
    ----------
    
    
    Reference:
        W Song, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks, 2018.
    """

    def __init__(self, field_dims, embed_dim, num_heads, num_layers, mlp_dims, dropouts):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = Linear_Embedder(field_dims)
        self.embedding = Embedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1])
        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.embed_output_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        cross_term = embed_x.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        cross_term = F.relu(cross_term).contiguous().view(-1, self.embed_output_dim)
        x = self.linear(x) + self.attn_fc(cross_term) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
