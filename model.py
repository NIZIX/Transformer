import torch.nn as nn
import torch
import torch.nn.functional as F
from config import Tranformer_config

class Embedding_with_pe(nn.Module):
    def __init__(self, vocab_size, sequence_length, embedding_dim):
        """
        input: tensor of tokens (batch_size, sequence_length, embedding_dim)
        """
        super().__init__()
        self.sequence_range = torch.arange(sequence_length).unsqueeze(0)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = nn.Embedding(sequence_length, embedding_dim)
    
    def forward(self, x):
        pe = self.position_encoding(self.sequence_range)
        embedding = self.embeddings(x)
        return embedding + pe
    

class MultiHeadAttention_AddNorm(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout=0.1):
        super().__init__()
        self.MHA = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        attn_output, _ = self.MHA(Q, K, V, attn_mask=mask)
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(Q + attn_output)
        return output
    

class FF_AddNorm(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        ff_output = self.ff(x)
        output = self.layer_norm(x + ff_output)
        return output
    

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention_AddNorm(embedding_dim, n_heads, dropout)
        self.ffn = FF_AddNorm(embedding_dim, ffn_dim, dropout)
        
    def forward(self, x, mask=None):
        x = self.attention(x, x, x, mask)
        x = self.ffn(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, n_blocks, embedding_dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, n_heads, ffn_dim, dropout) for _ in range(n_blocks)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention_AddNorm(embedding_dim, n_heads, dropout)
        self.cross_attention = MultiHeadAttention_AddNorm(embedding_dim, n_heads, dropout)
        self.ffn = FF_AddNorm(embedding_dim, ffn_dim, dropout)
        
    def forward(self, x, context, tgt_mask=None, memory_mask=None):
        x = self.self_attention(x, x, x, tgt_mask)
        x = self.cross_attention(x, context, context, memory_mask)
        x = self.ffn(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, n_blocks, embedding_dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_heads, ffn_dim, dropout) for _ in range(n_blocks)])

    def forward(self, x, context, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, context, tgt_mask, memory_mask)
        return x
    

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding_with_pe(config.vocab_size, config.max_len, config.emb_dim)
        self.encoder = Encoder(config.num_encoder_layers, config.emb_dim, config.num_heads, config.ffn_dim, config.dropout)
        self.decoder = Decoder(config.num_decoder_layers, config.emb_dim, config.num_heads, config.ffn_dim, config.dropout)
        self.linear = nn.Linear(config.emb_dim, config.vocab_size)
    
    def predict(self, x, tgt_mask=None, memory_mask=None):
        x = self.embedding(x)
        encoder_out = self.encoder(x)
        x = self.decoder(x, encoder_out, tgt_mask, memory_mask)
        x = self.linear(x)
        return F.argmax(x, dim=-1)
    
    def forward(self, src, tgt, tgt_mask=None, memory_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        encoder_out = self.encoder(src)
        decoder_out = self.decoder(tgt, encoder_out, tgt_mask, memory_mask)
        output = self.linear(decoder_out)
        return output


if __name__ == "__main__":
    config = Tranformer_config()
    transformer = Transformer(config)
    output = transformer(torch.randint(0, 50257, [1, 200]), torch.randint(0, 50257, [1, 200]))