import torch

from allennlp.common import Registrable


class Activation(Registrable):
    """
    Pytorch has a number of built-in activation functions.  We group those here under a common
    type, just to make it easier to configure and instantiate them ``from_params`` using
    ``Registrable``.
    Note that we're only including element-wise activation functions in this list.  You really need
    to think about masking when you do a softmax or other similar activation function, so it
    requires a different API.
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        This function is here just to make mypy happy.  We expect activation functions to follow
        this API; the builtin pytorch activation functions follow this just fine, even though they
        don't subclass ``Activation``.  We're just making it explicit here, so mypy knows that
        activations are callable like this.
        """
        raise NotImplementedError