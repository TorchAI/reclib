import torch


class AttentionalFactorizationLayer(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attn_scores = torch.nn.ReLU((self.attention(inner_product)))
        attn_scores = torch.nn.softmax(self.projection(attn_scores), dim=1)
        attn_scores = torch.nn.dropout(attn_scores, p=self.dropouts[0])
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = torch.nn.dropout(attn_output, p=self.dropouts[1])
        return self.fc(attn_output)
