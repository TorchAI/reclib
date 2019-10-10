import torch

from reclib.modules import FeedForward
from reclib.modules.embedders import LinearEmbedder, Embedding


class WideAndDeep(torch.nn.Module):
    """
    This implements wide and deep learning from `"Wide & Deep Learning for Recommender Systems,"
    <https://arxiv.org/abs/1606.07792>'_ by HT Cheng, et al.  2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = LinearEmbedder(field_dims, 1)
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
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        label_logits = torch.sigmoid(x.squeeze(1))
        return label_logits
