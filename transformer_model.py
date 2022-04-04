import math

import torch
import torch.nn as nn



class TransformerMT(nn.Module):
    '''
    This class represents a simple module for training a
    machine translation system.
    '''

    def __init__(
        self, vocab_size: int, max_seq_length: int,
        d_model=512, nhead=8, num_encoder_layers=6,
        num_decoder_layers=6, dim_feedforward=2048,
        dropout: float=0.1
    ):
        super().__init__()

        '''
        The weights for self.embeddings are shared by the encoder embeddings,
        the decoder embeddings and the pre-softmax transformation, following
        the original paper.
        '''
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )

        self.emb_dim = d_model
        self.max_seq_length = max_seq_length
        self.embeddings = nn.Embedding(vocab_size, self.emb_dim)
        self.to_vocab = nn.Linear(self.emb_dim, vocab_size)

        self.get_positional_encoding(),
        self.dropout = nn.Dropout(p=dropout)

    def get_positional_encoding(self) -> torch.Tensor:
        arange = torch.arange(self.max_seq_length).view(-1, 1)
        positions = torch.tile(arange, (1, self.emb_dim))

        exponent = torch.ones(self.emb_dim)
        exponent[0::2] = torch.arange(0, self.emb_dim, 2)
        exponent[1::2] = torch.arange(0, self.emb_dim, 2)
        vals = torch.pow(10000, -exponent / self.emb_dim)

        positional_encoding = positions * vals
        positional_encoding[:, 0::2] = torch.sin(positional_encoding[:, 0::2])
        positional_encoding[:, 1::2] = torch.cos(positional_encoding[:, 1::2])
        positional_encoding = positional_encoding.unsqueeze(-2)

        self.register_buffer(
            tensor=positional_encoding,
            name='pe', persistent=True,
        )
    
    def generate_input(self, token_ids):
        embeddings = self.embeddings(token_ids) * (self.emb_dim ** .5)
        sequence_length = embeddings.size()[0]
        out = embeddings[:sequence_length] + \
        self.pe[:sequence_length]
        return self.dropout(out)
    
    def encode(
        self, token_ids: torch.Tensor,
        mask=None, src_key_padding_mask=None
    ) -> torch.Tensor:
        '''
        Returns memory vectors, which will be used in encoder-decoder
        attention. token_ids has a shape (B, S), where B- batch size,
        S- sequence_length.
        '''
        encoder_input = self.generate_input(token_ids)
        return self.transformer.encoder(
            src=encoder_input, mask=mask,
            src_key_padding_mask=src_key_padding_mask
        )
    
    def decode(
        self, token_ids: torch.Tensor, memory: torch.Tensor,
        tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ) -> torch.Tensor:
        '''
        Returns logits for each element of the sequence.
        '''
        decoder_input = self.generate_input(token_ids)

        decoder_output = self.transformer.decoder(
            tgt=decoder_input, memory=memory,
            tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        logits = self.to_vocab(decoder_output)
        return logits
    
    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor,
        src_mask=None, tgt_mask=None, src_key_padding_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):
        src = self.generate_input(src)
        tgt = self.generate_input(tgt)

        out = self.transformer(
            src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        logits = self.to_vocab(out)
        return logits