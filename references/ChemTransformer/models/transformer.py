import torch
import torch.nn as nn
import math

class CompoundPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(CompoundPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = nn.Embedding(max_len, d_model, padding_idx=0)

    def forward(self, x, position_indices):
        '''
        x: Input tensor [batch_size, seq_len, d_model]
        position_indices: Tensor containing position indices for each token [batch_size, seq_len]
        '''
        pe = self.pe(position_indices)
        x = x + pe
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.pos_encoder = CompoundPositionalEncoding(d_model)
        self.pos_decoder = CompoundPositionalEncoding(d_model)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_pos, tgt_pos, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
        # Embeddings
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb, src_pos)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb, tgt_pos)

        # Transformer
        output = self.transformer(
            src_emb.permute(1, 0, 2),  # (seq_len, batch_size, d_model)
            tgt_emb.permute(1, 0, 2),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        if torch.isnan(output).any():
            print("NaN detected in src_emb after positional encoding")
            exit()

        # Output layer
        output = output.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        output = self.fc_out(output)
        return output
