from typing import Optional

import torch
import torch.nn as nn


class TransformerMT(nn.Module):
    '''
    This class represents a simple module for training a
    machine translation system.
    '''

    def __init__(
        self, vocab_size: int, max_seq_length: int,
        transformer_kwargs: dict
    ):
        super().__init__()

        '''
        The weights for self.embeddings are shared by the encoder embeddings,
        the decoder embeddings and the pre-softmax transformation, following
        the original paper.
        '''

        assert 'd_model' in transformer_kwargs and \
            'device' in transformer_kwargs

        self.transformer = nn.Transformer(**transformer_kwargs)

        self.emb_dim = transformer_kwargs['d_model']
        self.max_seq_length = max_seq_length
        self.embeddings = nn.Embedding(vocab_size, self.emb_dim)

        self.register_buffer(
            tensor=self.get_positional_encoding(),
            name='pe', persistent=True,
        )
        self.device = transformer_kwargs['device']
        self.pe.to(self.device)

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
        return positional_encoding
    
    def generate_input(self, token_ids):
        embeddings = self.embeddings(token_ids) * (self.emb_dim ** .5)
        sequence_length = embeddings.size()[1]
        embeddings = embeddings.to(self.device)
        return embeddings + self.pe[:sequence_length]
    
    def infer_memory(
        self, token_ids: torch.Tensor,
        encoder_kwargs: Optional[dict]=None
    ) -> torch.Tensor:
        '''
        Returns memory vectors, which will be used in encoder-decoder
        attention. token_ids has a shape (B, S), where B- batch size,
        S- sequence_length.
        '''
        encoder_input = self.generate_input(token_ids)
        
        if not encoder_kwargs:
            encoder_kwargs = dict()

        return self.transformer.encoder(src=encoder_input, **encoder_kwargs)
    
    def decode(
        self, token_ids: torch.Tensor, memory: torch.Tensor,
        decoder_kwargs: Optional[dict]=None
    ) -> torch.Tensor:
        '''
        Returns logits for the last element of each sequence.
        '''
        decoder_input = self.generate_input(token_ids)

        if not decoder_kwargs:
            decoder_kwargs = dict()

        decoder_output = self.transformer.decoder(
            tgt=decoder_input, memory=memory, **decoder_kwargs
        )
        
        logits = torch.matmul(
            decoder_output, self.embeddings.weight.T
        )
        return logits