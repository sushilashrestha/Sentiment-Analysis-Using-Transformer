import torch
import torch.nn as nn
import torch.nn.functional as F
from activation import Softmax
from embedding_layers import EmbeddingLayer, PositionalEncoding
from transformer_block import TransformerBlock
import copy


# class for Encoder Part of the Transformer Architecture
class Encoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 output_size,
                 max_seq_len,
                 padding_idx=0,
                 pooling="mean",
                 embedding_dim=512,
                 num_blocks=4,
                 activation="relu",
                 expansion_factor=4,
                 num_heads=8,
                 dropout=0.1
                 ):

        """
        The Encoder part of the Transformer architecture.

        Arguments:
            vocab_size : Vocabulary Size
            output_size : Target Size
            max_seq_len : Maximum length of the input sequence
            pooling : Specify the type of pooling to use. Available options: 'max' or 'avg'. defaul is 'max'
            embedding_dim :  Dimension to Represent words sequence (Feature for a single word). eg., 256, 512, etc ...
            num_blocks : Number of Transformer block
            activation : Activation to use inbetween feed forward layer. default is `relu`
            expansion_factor : The factor that determines the output dimension of the feed forward layer
            num_heads : Number of Attention Heads
            dropout : Percentage for Droping out the Layers. default is 0.1
        """

        super(Encoder, self).__init__()

        self.max_len = max_seq_len

        # define the embedding: (vocabulary size x embedding dimension)
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim, pad_idx=padding_idx)

        # define the positional encoding: (max_len x embedding dimension)
        self.pos_emb = PositionalEncoding(max_seq_len, embedding_dim)

        # pooling layer for down sampling
        assert pooling == 'max' or pooling == 'avg' or pooling == "mean", "Sorry! No other Pooling Methods Implemented as of Now. Current Options are 'max', 'mean', 'avg'."
        self.pooling = pooling

        stack_them_up = lambda block, n_block: nn.ModuleList([copy.deepcopy(block) for _ in range(n_block)]) # It Seems, What I Came up with for this lamda funtion name is actually concise

        self.transformer_blocks = stack_them_up(TransformerBlock(hidden_size=embedding_dim, activation=activation, num_heads=num_heads, dropout=dropout, expansion_factor=expansion_factor), num_blocks) # Sequentially applies the blocks of the Transformer network

        """
        self.transformer_blocks = nn.Sequential(
                *[TransformerBlock(
                    hidden_size=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    expansion_factor=expansion_factor
                    ) for _ in range(num_blocks)
                    ]
            )
        """

        # activation for the final layer
        self.sigmoid = nn.Sigmoid()
        self.softmax = Softmax(-1, keepdim=True)

        # final fully connected layer to project the output
        self.fc_out = nn.Linear(embedding_dim, output_size)

    def forward(self, x, padding_mask):

        """
        Forward Pass in Encoder Module of Transformer Architechture

        Args:
            x : sequence of tokenized words in batch for parallelism with shape of [batch_size, seq_len]. Note : Here seq_len is fixed len with padded tokens.

        Returns:
            Encoder Representation for the given sequence of words
        """

        batch_size = x.size(0)

        # Get the Embeddings for the Sequence
        embedded = self.embedding(x)

        # Add Postional Encoding for x
        out = self.pos_emb(embedded)

        # forward pass through transformer blocks
        for block in self.transformer_blocks:

            out = block(out, out, out, mask=padding_mask)

        if self.pooling == 'max':

            # Adaptive Max Pooling allows you to maintain important features while downsampling the sequence length
            # Permute to the shape (batch_size, embedding_dim, seq_length) and Apply adaptive max-pooling,
            # output shape : (batch_size, embedding_dim)
            output = F.adaptive_max_pool1d(out.permute(0,2,1), output_size=(1,)).view(batch_size,-1) # Expected Size - [batch_size x embedding_dim]

        elif self.pooling == "avg":

            # Average Pooling computes the average value of features within each pooling window, resulting in a downsized representation.
            # Permute to the shape (batch_size, embedding_dim, seq_length) and Apply Average pooling.
            # Output shape: (batch_size, embedding_dim)
            output = F.avg_pool1d(out.permute(0,2,1), kernel_size=self.max_len
                                  , stride=1) # .view(batch_size, -1) # Expected Size - [batch_size x embedding_dim]
        elif self.pooling == "mean":

            # Taking mean value over each row of the second dimension.
            # We should set keepdims kwarg to be False, so that the output wil be squeezed along the given dimension otherwise the size of the given dimension will be set to 1.
            output = torch.mean(out, dim=1, keepdim=False) # Expected Size - [batch_size x embedding_dim]

        else:

            print("Sorry! No other Pooling Methods Implemented as of Now.")

        """
        You can Use the sigmoid activation converts that into probablity between 0 to 1. which determines whether the sentence is positive or negative. But,
        The Major disadvantages of using sigmoid is the problem of `vanishing gradient`. The sigmoid function saturates for extreme input values, resulting in very small gradients.
        This can lead to vanishing gradients during backpropagation, which can slow down or even hinder the training of deep neural networks. As a result, it can make it difficult to train deep networks effectively.
        So I prefer to use softmax activation with two nodes instead.

        If you want use sigmoid activation uncomment the code below.
        output_prob = self.sigmoid(self.fc_out(out))
        """
        # Here, Th Final Linear Layer will project the dimension of the last axis to output_size which in this case is `2`.

        output_prob = self.fc_out(output)

        # Expected Shape - [batch_size, output_size]
        return output_prob