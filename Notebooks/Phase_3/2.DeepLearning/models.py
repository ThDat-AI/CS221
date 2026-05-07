import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.proj(torch.tanh(rnn_out)).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (weights * rnn_out).sum(dim=1)
        return ctx


class BiLSTMAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        padding_idx: int,
        dropout: float = 0.4,
        pretrained_embedding: torch.Tensor | None = None,
    ):
        super().__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding,
                freeze=False,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
            nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
            if padding_idx is not None:
                with torch.no_grad():
                    self.embedding.weight[padding_idx].zero_()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = AttentionPooling(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1)
        )
        mask = x != self.embedding.padding_idx
        ctx = self.attn(out, mask)
        ctx = self.dropout(ctx)
        return self.fc(ctx)


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        padding_idx: int,
        dropout: float = 0.4,
        num_filters: int = 128,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        pretrained_embedding: torch.Tensor | None = None,
    ):
        super().__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding,
                freeze=False,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
            nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
            if padding_idx is not None:
                with torch.no_grad():
                    self.embedding.weight[padding_idx].zero_()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in filter_sizes
        )
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        _ = lengths
        emb = self.dropout(self.embedding(x))
        emb = emb.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)
            pooled.append(h)
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)
