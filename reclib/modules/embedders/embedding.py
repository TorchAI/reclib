from reclib.modules.embedders import Embedder 
from torch.nn import Embedding
import numpy as np

class Embedding(Embedder):
    """
    """
    def __init__(self, field_sizes, embed_dim):
        self.embedding = Embedding(sum(field_sizes)+1, embed_dim)
        self.offsets = np.array((0, np.cumsum(field_sizes), dtype=np.long))
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, field_id):
        """
        Parameters
        ----------
        field_id: ``np.arrary``
        A tensor of shape ``(batch_size, num_fields)`` representing field ids
        
        new_tensor() returned Tensor has the same torch.dtype and torch.device as this tensor.
        """
        field_id = field_id + field_id.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(field_id)
