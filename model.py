import torch
from torch import nn
from transformers import AutoModel
from linformer import Linformer


class TinyLinformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        proj_k: int = 64,
        max_len: int = 128,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.register_buffer("pos", torch.randn(1, max_len, hidden_dim) * 0.02, persistent=False)
        self.encoder = Linformer(dim=hidden_dim, seq_len=max_len, depth=num_layers, heads=num_heads, k=proj_k)

    @property
    def config(self):
        return type("cfg", (), {"hidden_size": self.embed.embedding_dim})

    def forward(self, input_ids, attention_mask=None):
        pos = self.pos[:, : input_ids.size(1)].to(input_ids.device, non_blocking=True)
        x = self.embed(input_ids) + pos
        x = self.encoder(x)
        return type("Output", (), {"last_hidden_state": x})


class Text2Class(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained_model_name: str | None,
        tokenizer_vocab_size: int | None = None,
        max_len: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if pretrained_model_name and pretrained_model_name.lower() not in {"custom", "none"}:
            self.encoder = AutoModel.from_pretrained(pretrained_model_name)
            hidden_dim = self.encoder.config.hidden_size
        else:
            if tokenizer_vocab_size is None:
                raise ValueError("tokenizer_vocab_size required for custom encoder")
            self.encoder = TinyLinformerEncoder(vocab_size=tokenizer_vocab_size, hidden_dim=hidden_dim, max_len=max_len)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        x = self.mlp(cls)
        return self.head(x)
