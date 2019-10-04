import torch


class AttentionalFactorizationLayer(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention_layer = torch.nn.Linear(embed_dim, attn_size)
        self.projection_layer = torch.nn.Linear(attn_size, 1)
        self.linear_layer = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """
        Parameters
        ----------
        x
        Size of ``(batch_size, num_fields, embed_dim)``

        Returns
        -------

        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attention_scores = torch.nn.ReLU((self.attention_layer(inner_product)))
        attention_scores = torch.nn.softmax(self.projection_layer(attention_scores), dim=1)
        attention_scores = torch.nn.dropout(attention_scores, p=self.dropouts[0])
        attention_output = torch.sum(attention_scores * inner_product, dim=1)
        attention_output = torch.nn.dropout(attention_output, p=self.dropouts[1])
        return self.linear_layer(attention_output)
