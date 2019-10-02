import torch

from reclib.modules.embedders import LinearEmbedder


class LogisticRegression(torch.nn.Module):
    def __init__(self, field_dims):
        super().__init__()
        self.linear = LinearEmbedder(field_dims, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x).squeeze(1))
