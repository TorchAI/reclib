import torch

from reclib.common import Registrable


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


# There are no classes to decorate, so we hack these into Registrable._registry.
# If you want to instantiate it, you can do like this:
# Activation.by_name('relu')()
# pylint: disable=protected-access
Registrable._registry[Activation] = {  # type: ignore
    "linear": lambda: lambda x: x,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "elu": torch.nn.ELU,
    "prelu": torch.nn.PReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "threshold": torch.nn.Threshold,
    "hardtanh": torch.nn.Hardtanh,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "log_sigmoid": torch.nn.LogSigmoid,
    "softplus": torch.nn.Softplus,
    "softshrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "tanhshrink": torch.nn.Tanhshrink,
}
