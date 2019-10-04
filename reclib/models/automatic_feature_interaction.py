import torch
import torch.nn.functional as F

from reclib.modules import FeedForward
from reclib.modules.embedders import LinearEmbedder, Embedding


class AutomaticFeatureInteraction(torch.nn.Module):
    """
    This implements AutoInt from `"AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"
    <https://arxiv.org/abs/1810.11921>`_ by W Song, et al. , 2018.
    """

    def __init__(self, field_dims, embed_dim, num_heads, num_layers, mlp_dims, dropouts):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = LinearEmbedder(field_dims, 1)
        self.embedding = Embedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.self_attention = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.mlp = FeedForward(num_layers=2,
                               input_dim=self.embed_output_dim,
                               hidden_dims=mlp_dims,
                               batch_norm=True,
                               activations=torch.nn.ReLU(),
                               dropouts=dropouts[1])
        self.output_linear = torch.nn.Linear(mlp_dims[-1], 1)
        self.attention_linear = torch.nn.Linear(self.embed_output_dim, 1)

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
            probabilities of the entailment label.
        """
        embed_x = self.embedding(x)
        cross_term = embed_x.transpose(0, 1)
        for self_attn in self.self_attention:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        cross_term = F.relu(cross_term).reshape(-1, self.embed_output_dim)
        tmp = self.output_linear(self.mlp(embed_x.view(-1, self.embed_output_dim)))
        x = self.linear(x) + self.attention_linear(cross_term) + tmp

        label_logits = torch.sigmoid(x.squeeze(1))
        return label_logits
