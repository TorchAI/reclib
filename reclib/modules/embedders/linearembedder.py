import numpy as np
import torch
from torch.nn import EmbeddingBag, Parameter

from reclib.modules.embedders import Embedder


class LinearEmbedder(Embedder):
    def __init__(self, field_sizes, embed_dim):
        super(LinearEmbedder, self).__init__()
        self.embedding = EmbeddingBag(sum(field_sizes) + 1, embed_dim, mode='sum')
        self.offsets = np.array((0, *np.cumsum(field_sizes)[:-1]), dtype=np.long)
        self.bias = Parameter(torch.zeros((embed_dim,)))

    def forward(self, field_id):
        shifted_id = field_id + field_id.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(shifted_id) + self.bias
