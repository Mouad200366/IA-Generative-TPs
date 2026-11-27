"""Architectures pédagogiques adaptées du support du Pr. Mohamed Youssfi."""
import torch
from torch import nn

PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4


class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=16, batch_first=True)
        self.sortie = nn.Linear(16, 1)

    def forward(self, x):
        etats, _ = self.rnn(x)
        return self.sortie(etats[:, -1, :])


class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.sortie = nn.Linear(16, 1)

    def forward(self, x):
        etats, _ = self.lstm(x)
        return self.sortie(etats[:, -1, :])


class MiniTransformer(nn.Module):
    def __init__(self, longueur_max=10):
        super().__init__()
        self.projection = nn.Linear(1, 16)
        self.positions = nn.Embedding(longueur_max, 16)
        couche = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32,
            dropout=0.0, batch_first=True)
        self.encodeur = nn.TransformerEncoder(couche, num_layers=1, enable_nested_tensor=False)
        self.sortie = nn.Linear(16, 1)

    def representations(self, x):
        indices = torch.arange(x.size(1), device=x.device)
        return self.encodeur(self.projection(x) + self.positions(indices).unsqueeze(0))

    def forward(self, x):
        return self.sortie(self.representations(x)[:, -1, :])


class EnergyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 32)
        self.positions = nn.Embedding(24, 32)
        couche = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.encodeur = nn.TransformerEncoder(couche, num_layers=2, enable_nested_tensor=False)
        self.sortie = nn.Linear(32, 1)

    def representations(self, x):
        indices = torch.arange(x.size(1), device=x.device)
        return self.encodeur(self.projection(x) + self.positions(indices).unsqueeze(0))

    def forward(self, x):
        return self.sortie(self.representations(x)[:, -1, :])


class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(64)
        self.mlp = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64), nn.Dropout(0.1))

    def forward(self, x, masque):
        normalise = self.norm1(x)
        attention, _ = self.attention(normalise, normalise, normalise, attn_mask=masque, need_weights=False)
        x = x + attention
        return x + self.mlp(self.norm2(x))


class MiniGPT(nn.Module):
    def __init__(self, taille_vocabulaire):
        super().__init__()
        self.tokens = nn.Embedding(taille_vocabulaire, 64)
        self.positions = nn.Embedding(64, 64)
        self.blocs = nn.ModuleList([GPTBlock(), GPTBlock()])
        self.sortie = nn.Linear(64, taille_vocabulaire)

    def representations(self, ids):
        longueur = ids.size(1)
        if not 1 <= longueur <= 64:
            raise ValueError('Une séquence GPT contient de 1 à 64 caractères.')
        x = self.tokens(ids) + self.positions(torch.arange(longueur, device=ids.device))
        masque = torch.triu(torch.ones(longueur, longueur, dtype=torch.bool, device=ids.device), diagonal=1)
        for bloc in self.blocs:
            x = bloc(x, masque)
        return x

    def forward(self, ids):
        return self.sortie(self.representations(ids))


class MiniBERT(nn.Module):
    def __init__(self, taille_vocabulaire):
        super().__init__()
        self.tokens = nn.Embedding(taille_vocabulaire, 64, padding_idx=PAD)
        self.positions = nn.Embedding(32, 64)
        couche = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True)
        self.encodeur = nn.TransformerEncoder(couche, num_layers=2, enable_nested_tensor=False)
        self.sortie = nn.Linear(64, taille_vocabulaire)

    def representations(self, ids, masque_padding):
        if ids.size(1) > 32 or masque_padding.dtype != torch.bool:
            raise ValueError('BERT exige au plus 32 tokens et un masque booléen de padding.')
        x = self.tokens(ids) + self.positions(torch.arange(ids.size(1), device=ids.device))
        return self.encodeur(x, src_key_padding_mask=masque_padding)

    def forward(self, ids, masque_padding):
        return self.sortie(self.representations(ids, masque_padding))
