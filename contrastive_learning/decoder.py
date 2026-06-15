"""
Transformer Decoder — 将编码器特征 [49, 256] 解码为 LaTeX token 序列。

架构:
  - Cross-attention 到编码器输出的 49 个"图像块"特征
  - 4 层 transformer decoder，每层 4 头注意力
  - 约 2M 参数，CPU 可训

用法:
    from contrastive_learning.decoder import LaTeXDecoder, create_decoder
    model = create_decoder(vocab_size=200)
"""

import torch
import torch.nn as nn
from x_transformers import Decoder, TransformerWrapper, AutoregressiveWrapper


class LaTeXDecoder(nn.Module):
    """将编码器特征 [49, 256] 解码为 LaTeX token 序列。"""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

        # 编码器特征投影（256 → dim，如果不同）
        self.memory_proj = nn.Identity()

        # Transformer Decoder（带 cross-attention）
        self.model = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                cross_attend=True,
                attn_dropout=dropout,
                ff_dropout=dropout,
            ),
        )

        # 自回归包装
        self.wrapper = AutoregressiveWrapper(
            self.model,
            ignore_index=pad_token_id,
        )

    def forward(self, features: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """训练前向：teacher forcing + cross-entropy loss。

        Args:
            features: [B, 49, 256] 编码器特征。
            tokens: [B, L] token 序列。

        Returns:
            loss scalar。
        """
        return self.wrapper(
            tokens,
            context=self.memory_proj(features),
        )

    @torch.no_grad()
    def generate(
        self,
        features: torch.Tensor,
        max_len: int = 128,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """自回归生成 LaTeX token 序列。

        Args:
            features: [B, 49, 256] 编码器特征。
            max_len: 最大生成长度。
            temperature: 采样温度。

        Returns:
            [B, L_out] token ID 序列。
        """
        B = features.shape[0]
        start_tokens = torch.full(
            (B, 1), self.sos_token_id, dtype=torch.long, device=features.device
        )
        return self.wrapper.generate(
            start_tokens,
            seq_len=max_len,
            eos_token=self.eos_token_id,
            temperature=temperature,
            context=self.memory_proj(features),
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_decoder(
    ckpt_path: str,
    vocab_size: int,
    dim: int = 256,
    device: str = "cpu",
) -> LaTeXDecoder:
    model = LaTeXDecoder(vocab_size=vocab_size, dim=dim)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
