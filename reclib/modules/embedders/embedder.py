import torch

from reclib.common import Registrable


class Embedder(torch.nn.Module, Registrable):
    """
    A ``Embedder`` is a ``Module`` that takes as input a tensor with integer ids that have
    been output from a :class:`~torch.utils.data.DataLoader` and outputs a vector per token in the
    input.  The input typically has shape ``(batch_size, num_fields)``, and the output is of shape ``(batch_size, num_fields,
    output_dim)``.
    We add a single method to the basic ``Module`` API: :func:`get_output_dim()`.  This lets us
    more easily compute output dimensions for the :class:`~reclib.modules.Embedder`,
    which we might need when defining model parameters such as LSTMs or linear layers, which need
    to know their input dimension before the layers are called.
    """
    default_implementation = "embedding"

    def get_output_dim(self) -> int:
        """
        Returns the final output dimension that this ``Embedder`` uses to represent each
        token.  This is the last element of that shape.
        """
        raise NotImplementedError

    def forward(self):
        return
