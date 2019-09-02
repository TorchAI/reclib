from reclib.modules.embedders import Embedder
from torch.nn import EmbeddingBag, Parameter
import numpy as np

class Linear_Embedder(Embedder):
    def __init__(self, field_sizes, embed_dim):
        super.__init__()
        self.embedding = EmbeddingBag(sum(field_sizes)+1, embed_dim, mode='sum')
        self.offsets = np((0, np.cumsum(field_sizes)[:-1]), dtype=long)
        self.bias = Parameter(torch.zeros((embed_dim,)))

    def forward(self, field_id):
        offseted_id = field_id + field_id.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(offseted_id) + self.bias

