import numpy as np
import torch

from reclib.modules.embedders import Embedder


class Embedding(Embedder):
    """
    """

    def __init__(self, field_sizes, embed_dim):
        """
        Parameters
        ----------
        field_sizes: ``List[int]``
        Sizes of each field. Size here means the number of unique items in that field
        embed_dim: ``int``
        """
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Embedding(sum(field_sizes) + 1, embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_sizes)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, field_id):
        """
        Parameters
        ----------
        field_id: ``np.arrary``
        A tensor of shape ``(batch_size, num_fields)``

        new_tensor() returned Tensor has the same torch.dtype and torch.device as this tensor.
        """
        # Add the offset here cuz they use the same dictionary but both start from 1 originally
        field_id = field_id + field_id.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(field_id)
