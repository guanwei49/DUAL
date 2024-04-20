

from torch import nn
from models.token_embeddings import TokenEmbedding


class ClientEmbedding(nn.Module):
    """
    token embedding
    """

    def __init__(self, vocab_sizes, d_model):
        super(ClientEmbedding, self).__init__()
        self.embeddings = []
        for i, dim in enumerate(vocab_sizes):
            self.embeddings.append(TokenEmbedding(dim, d_model))
        self.embeddings = nn.ModuleList(self.embeddings)

    def forward(self, xs):
        hs=[]
        for i, x in enumerate(xs):
            tok_emb = self.embeddings[i](x)
            hs.append(tok_emb)
        return hs



class ClientReconstructor(nn.Module):
    def __init__(self, dec_voc_sizes, d_model):
        super(ClientReconstructor, self).__init__()
        self.reconstructors = []
        for i, dim in enumerate(dec_voc_sizes):
            self.reconstructors.append(nn.Linear(d_model, dim))
        self.reconstructors = nn.ModuleList(self.reconstructors)


    def forward(self, hs):
        reconstructed_xs = []
        for i, h in enumerate(hs):
            reconstructed_xs.append(self.reconstructors[i](h))
        return reconstructed_xs