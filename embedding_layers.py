# Importing the Necessary Modules
import torch
import torch.nn as nn

# Embedding Layer
class EmbeddingLayer(nn.Module):

    """
    Class to project the word sequences to Multi Dimensional Space

    Args:
        vocab_size: Vocablary Size
        embedding_dim: Dimension to Represent words sequence (Feature for a single word).
                       eg., 256, 512 (As the Dimension Increases, More Dependencies/Context can be Capture as well need more computation)

    For example, if you have a batch of 64 sequences, each containing 15 words, and the embedding dimension is 512,
    the output tensor will be of size 64x15x512, where each element in the tensor represents a numerical embedding.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 pad_idx=0,
                 pretrained_embeddings=None,
                 freeze_embeddings=True
                ):

        super(EmbeddingLayer, self).__init__()
        self.embed_dim = embedding_dim

        if pretrained_embeddings is not None:
            if pretrained_embeddings.size(1) != embedding_dim:
                raise ValueError("The embedding dimension does not match the pretrained embeddings.")
            self.embed = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings, padding_idx=pad_idx)
        else:
            self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        #self.embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
      
        """
        Forward Pass through the Emedding Layer
        
        Arguments:
            x : The input word(s) or sequence of words that need to be embedded.
        Returns:
            Embeddings for the given sequence
        """
      
        x = self.embed(x) # This gives a tensor representing the embeddings of the input words.
        embedded = x * torch.sqrt(torch.tensor(self.embed_dim)) # This scaling factor is often used to prevent gradient explosions when training deep networks.
        return embedded # The resulting tensor is the numerical representation of the input in the embedding space.


# Positional Embedding Layer
class PositionalEncoding(nn.Module):


    def __init__(self,
                 max_len,
                 d_model=512,
                 n=10000.0
                ):

        """
        class for Positional Embedding or Positional Encoding in Transfomer Architechture

        This addresses the issue of sequence order and helps the model understand the relative positions of tokens within a sequence.

        In LSTMs, GRUs,  Recurrent Neural networks (RNN), the inputs are fed into the model sequentially. For each timestamps, the input word is fed and the corresponding hidden state is obtained.
        This way, the model learns the relative position of the word within a sequence.

        But in the Transformer architecture, the model processes tokens in parallel using self-attention mechanisms.
        Since self-attention doesn't inherently take into account the position of tokens,
        positional embeddings are added to the input embeddings to provide information
        about the positions of tokens in the sequence.

        Arguments:
            max_len : Maximum Length of the Sequence
            embedding_dim : Dimension of the Embedding, This Must be Same as Embedding vector
            drouput : Dropout Probablity

        """

        super(PositionalEncoding, self).__init__()

        self.max_len = max_len
        self.embedding_dim = d_model
        #self.dropout = nn.Dropout(dropout)
        self.n = n

        positional_encoding = torch.zeros(max_len, d_model)  # Matrix Filled with zeros of shape (max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1) # Positions/Index of the Words in a sequence

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(self.n)) / d_model))

        """
        Denominator for Scaling. These exponential values help create a pattern that contributes to the unique encoding for each position
        This has Many Benefits,
        - It control the frequency of oscillation (how quickly the function oscillates as position changes)
        - It Ensures each position has unique and distinctive. Without scaling, the same positional encodings could repeat over and over,
          making it difficult for the model to differentiate positions. (Encoding Relative Positions)
        - The positional encoding is designed to handle sequences of varying lengths.
        """

        """
        for i in position:
            for j in torch.arange(0, embedding_dim, 2)):
                positional_encoding[i, 2*j] = pe_sin(i, j)
                positional_encoding[i, 2*j+1] = pe_cos(i, j)

        You Can use this if you want but it can be done efficiently using vectorized operation, done below
        """

        # Vectorized Operation
        positional_encoding[:, 0::2] = torch.sin(position * div_term) # apply sin functions for every two coloumn of pos_emb matrix starting from 0. This term `position * div_term` has shape (max_seq_len, embedding_dim/2)
        positional_encoding[:, 1::2] = torch.cos(position * div_term) # apply cosine functions for every two coloumn of pos_emb matrix starting from 0.

        pe = positional_encoding.unsqueeze(0) # Add Extra Batch Dimension along the first axis
        self.register_buffer('pe', pe) # Register Buffer to make it a part of the module's state_dict

    def _pe_sin(self, position, i): # internal sin function
        return torch.sin(position / torch.pow(self.n, ((2 * i) / self.embedding_dim)))

    def _pe_cos(self, position, i): # internal cosine function
        return torch.cos(position / torch.pow(self.n, ((2 * i) / self.embedding_dim)))

    def forward(self, x):

        """
        Forward method of Positional Encoding

        Arguments:
            x: Embeddings of the Input Sequence
        Returns:
            Position Injected Embeddings
        """

        # print(x.shape, self.pe.shape)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False) # setting the gradient calculation for this module to be false as Postional Encoding is not learned during training.
        return x # [batch_size, seq_len, embedding_dim]