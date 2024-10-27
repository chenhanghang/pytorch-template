import math

import torch
import torch.nn as nn
import copy

from .feed_forward import FeedForward
from .mha import MultiHeadAttention
from .positional_encoding import get_positional_encoding


class EmbeddingsWithPositionalEncoding(nn.Module):
    """
    Embed tokens and add [fixed positional encoding](positional_encoding.html)
    """
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len))

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
        return self.linear(x) * math.sqrt(self.d_model) + pe  # id 转embding 增加位置编码


class EmbeddingsWithLearnedPositionalEncoding(nn.Module):
    """
    Embed tokens and add parameterized positional encodings
    """
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]  # 选取固定长度的位置
        return self.linear(x) * math.sqrt(self.d_model) + pe


class TransformerLayer(nn.Module):
    """
    This can act as an encoder layer or a decoder layer. We use pre-norm.
    """
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention = None,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the self attention module
        * `src_attn` is the source attention module (when this is used in a decoder)
        * `feed_forward` is the feed forward module
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor,
                src: torch.Tensor = None,
                src_mask: torch.Tensor = None):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        # Add the self attention results，残差
        x = x + self.dropout(self_attn)

        # If a source is provided, get results from attention to source.
        # This is when you have a decoder layer that pays attention to encoder outputs
        # decoder 需要
        if src is not None:
            # Normalize vectors
            z = self.norm_src_attn(x)
            # Attention to source. i.e. keys and values are from source
            # 注意此处key 和 value 来源 encoder
            attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            # Add the source attention results
            x = x + self.dropout(attn_src)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff = self.feed_forward(z)  # 前向层
        # Add the feed-forward results back， 残差
        x = x + self.dropout(ff)

        return x


class Encoder(nn.Module):
    """
    ## Transformer Encoder
    """

    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = [copy.deepcopy(layer) for _ in range(n_layers)]
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Run through each transformer layer
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        # Finally, normalize the vectors
        return self.norm(x)


class Decoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = [copy.deepcopy(layer) for _ in range(n_layers)]
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        """
        * `memory`
        * `src_mask`
        * `tgt_mask`
        """
        # Run through each transformer layer
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        # Finally, normalize the vectors
        return self.norm(x)


class Generator(nn.Module):
    """
    This predicts the tokens and gives the lof softmax of those.
    You don't need this if you are using `nn.CrossEntropyLoss`.
    """

    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        return self.projection(x)


class EncoderDecoder(nn.Module):
    """
    Combined Encoder-Decoder
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # Run the source through encoder
        enc = self.encode(src, src_mask)
        # Run encodings and targets through decoder
        return self.decode(enc, src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)




def subsequent_mask(seq_len):
    """
    ## Subsequent mask to mask out data from future (subsequent) time steps
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).to(torch.bool).unsqueeze(-1)
    return mask

def _subsequent_mask():
    subsequent_mask(10)[:, :, 0]

if __name__ == '__main__':
    _subsequent_mask()