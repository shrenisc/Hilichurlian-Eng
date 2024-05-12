import torch
from torch import nn
from torch.nn import functional as F
import math


def compute_freqs_cis(embed_dim: int, max_seq_length: int, device, theta: float = 10000.0):
    # we compute the frequencies for the rotary embeddings
    freqs = 1.0 / (theta ** (torch.arange(0, embed_dim, 2,
                   device=device)[: (embed_dim // 2)].float() / embed_dim))
    t = torch.arange(max_seq_length, device=freqs.device).float()  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_real = torch.cos(freqs)
    freqs_imag = torch.sin(freqs)
    return freqs_real, freqs_imag


def apply_rotary_emb(input_tensor, C, T):
    # we apply the rotary embeddings to the input tensor(Add the complex numbers)
    xq_ = input_tensor.float().reshape(*input_tensor.shape[:-1], -1, 2)

    xq_real = xq_[:, :, :, 0]
    xq_imag = xq_[:, :, :, 1]

    freqs_real, freqs_imag = compute_freqs_cis(
        C, input_tensor.shape[-2], input_tensor.device)
    freqs_real = freqs_real[:T]
    freqs_imag = freqs_imag[:T]

    freqs_real = freqs_real.unsqueeze(0)
    freqs_imag = freqs_imag.unsqueeze(0)
    xq_out_real = xq_real * freqs_real - xq_imag * freqs_imag
    xq_out_imag = xq_real * freqs_imag + xq_imag * freqs_real
    xq_out = torch.stack([xq_out_real, xq_out_imag], dim=-1).flatten(3)

    return xq_out.type_as(input_tensor)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

        # some basic parameters
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.c_attn(x).split(self.embed_dim, dim=2)

        # apply the rotary embeddings for the query and key
        query = apply_rotary_emb(query, C, T)
        key = apply_rotary_emb(key, C, T)

        # get the q, k, v into the correct shjape
        key = key.view(B, T, self.num_heads, C //
                       self.num_heads).transpose(1, 2)
        query = query.view(B, T, self.num_heads, C //
                           self.num_heads).transpose(1, 2)
        value = value.view(B, T, self.num_heads, C //
                           self.num_heads).transpose(1, 2)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # put it together

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0
        # key, query, value projections for all heads,
        self.query_attn = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_attn = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_attn = nn.Linear(embed_dim, embed_dim, bias=bias)

        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # regularization
        self.resid_dropout = nn.Dropout(dropout)

        # some basic parameters
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

    def forward(self, query, key, value):
        # batch size, sequence length, embedding dimensionality (n_embd)
        query_batch, query_block, query_channels = query.size()
        # batch size, sequence length, embedding dimensionality (n_embd)
        key_batch, key_block, key_channels = key.size()
        # batch size, sequence length, embedding dimensionality (n_embd)
        value_batch, value_block, value_channels = value.size()

        assert query_batch == key_batch == value_batch
        assert query_channels == key_channels == value_channels

        # calculate query, key, values for all heads
        query = self.query_attn(query)
        key = self.key_attn(key)
        value = self.value_attn(value)

        # apply the rotary embeddings for the query and key
        query = apply_rotary_emb(query, query_channels, query_block)
        key = apply_rotary_emb(key, key_channels, key_block)

        key = key.view(key_batch, key_block, self.num_heads,
                       key_channels // self.num_heads).transpose(1, 2)
        query = query.view(query_batch, query_block, self.num_heads,
                           query_channels // self.num_heads).transpose(1, 2)
        value = value.view(value_batch, value_block, self.num_heads,
                           value_channels // self.num_heads).transpose(1, 2)

        # print(query.shape, key.shape, value.shape)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(
            query_batch, query_block, query_channels)  # put it together

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class RMSNorm(torch.nn.Module):
    """
    RMS Normalization,
    It normalizes using RMS, and also has a learnable weights
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Decoder(nn.Module):
    """
    This is the decoder of the transforemr, it is a stack of self attention, cross attention, and feed forward layers
    It uses prenormalization, from GPT-3 and SilU activation function
    """

    def __init__(self, input_shape, output_shape, num_heads, dropout=0):
        super(Decoder, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_shape, input_shape*2, bias=False),
            nn.ReLU(),
            nn.Linear(input_shape*2, output_shape, bias=False),
            nn.ReLU(),
        )
        self.layernorm = RMSNorm(input_shape)
        self.layernorm2 = RMSNorm(input_shape)
        self.layernorm3 = RMSNorm(input_shape)

        self.MHA = SelfAttention(
            embed_dim=input_shape, num_heads=num_heads, bias=False, dropout=dropout)
        self.CA = CrossAttention(
            embed_dim=input_shape, num_heads=num_heads, bias=False, dropout=dropout)

    def forward(self, x, y):
        x1 = self.layernorm(x)
        x = x + self.MHA(x1)

        x1 = self.layernorm2(x)
        x = x + self.CA(x1, y, y)

        x1 = self.layernorm3(x)
        x = x + self.feed_forward(x1)
        return x


class Encoder(nn.Module):
    def __init__(self, input_shape, output_shape, num_heads, dropout=0):
        super(Encoder, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_shape, input_shape*2, bias=False),
            nn.ReLU(),
            nn.Linear(input_shape*2, output_shape, bias=False),
            nn.ReLU(),
        )
        self.layernorm = RMSNorm(input_shape)
        self.layernorm2 = RMSNorm(input_shape)

        self.MHA = SelfAttention(
            embed_dim=input_shape, num_heads=num_heads, bias=False, dropout=dropout)

    def forward(self, x):
        x1 = self.layernorm(x)
        x = x + self.MHA(x1)

        x1 = self.layernorm2(x)
        x = x + self.feed_forward(x1)
        return x


class Translator(nn.Module):
    """
    The main model
    """

    def __init__(self, engVocabSize, hilliVocabSize, embed_size, num_decoder_blocks, num_encoder_blocks, num_heads, dropout, pad_char):
        super(Translator, self).__init__()
        self.engEmbedding = nn.Embedding(engVocabSize, embed_size)
        self.hilliEmbedding = nn.Embedding(hilliVocabSize, embed_size)
        self.decoder_block = nn.ModuleList([Decoder(
            embed_size, embed_size, num_heads, dropout) for _ in range(num_decoder_blocks)])
        self.encoder_block = nn.ModuleList([Encoder(
            embed_size, embed_size, num_heads, dropout) for _ in range(num_encoder_blocks)])
        self.dense = nn.Linear(embed_size, engVocabSize, bias=False)
        self.pad_char = pad_char

    def forward(self, x, originalText, y=None, return_loss=False):
        x = self.engEmbedding(x)
        originalText = self.hilliEmbedding(originalText)

        for encoder in self.encoder_block:
            originalText = encoder(originalText)

        for decoder in self.decoder_block:
            x = decoder(x, originalText)
        x = self.dense(x)

        # when calculating the loss, apply the masking, so that we don't calculate the loss for the padding
        if return_loss:
            mask = (y != self.pad_char).to(torch.int64).view(-1)
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)), y.view(-1), ignore_index=-1, reduction='none', label_smoothing=0.4)
            loss = (loss * mask).sum() / mask.sum()
            acc = (x.argmax(dim=-1) == y).to(torch.float32).view(-1)
            acc = (acc * mask).sum() / mask.sum()
            return loss, acc
        else:
            x = F.softmax(x, dim=-1)
            return x

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def config_optimizer(self, lr):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=0.1, fused=False)
        return optimizer
