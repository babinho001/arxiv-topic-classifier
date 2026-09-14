import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    Class for handling the input embeddings of tokens.
    """

    def __init__(
            self,
            model_dimension: int,
            vocab_size: int
        ) -> None:
        """Initializing the InputEmbeddings object."""
        super().__init__()

        self.model_dimension = model_dimension
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x) -> torch.Tensor:
        """
        Translates the token into it's embedding.
        """
        return self.embedding(x) * math.sqrt(self.model_dimension)


class PositionalEncoding(nn.Module):
    """
    Class for handling the positional embeddings of tokens.
    """

    def __init__(
            self,
            model_dimension: int,
            context_size: int,
            dropout: float
        ) -> None:
        """Initializing the PositionalEncoding object."""
        super().__init__()

        self.model_dimension = model_dimension
        self.context_size = context_size
        self.dropout = nn.Dropout(dropout)

        positional_encodings = torch.zeros(context_size, model_dimension) # (context_size, model_dimension)
        position = torch.arange(0, context_size, dtype = torch.float).unsqueeze(1) # (context_size, 1)
        div_term = torch.exp(torch.arange(0, model_dimension, 2).float() * (-math.log(10000.0) / model_dimension)) # (model_dimension / 2)
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)

        positional_encodings = positional_encodings.unsqueeze(0) # (1, context_size, model_dimension)
        self.register_buffer('pe', positional_encodings)

    def forward(self, x):
        """
        Adds the positional encodings to input embeddings of a
        given token, and applies dropout for regularization.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    Class for handling the normalization of vectors in a given layer.
    """

    def __init__(
            self,
            features: int,
            eps: float = 10**-6
        ) -> None:
        """Initializing the LayerNormalization object."""
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """
        Applies the normalization to a given embedding.
        """
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class MultiHeadAttentionBlock(nn.Module):
    """
    Class for handling the multihead attention.
    """

    def __init__(
            self,
            model_dimension: int,
            heads: int,
            dropout: float
        ) -> None:
        """Initializing the MultiHeadAttentionBlock object."""
        super().__init__()

        self.model_dimension = model_dimension
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        assert model_dimension % heads == 0, "model_dimension is not divisible by the number of heads."

        self.head_dimension = model_dimension // heads

        self.w_q = nn.Linear(model_dimension, model_dimension)
        self.w_k = nn.Linear(model_dimension, model_dimension)
        self.w_v = nn.Linear(model_dimension, model_dimension)
        self.w_o = nn.Linear(model_dimension, model_dimension)

    @staticmethod
    def attention(
            query,
            key,
            value,
            mask,
            dropout: nn.Dropout
        ):
        """
        Perform a masked multi head attention on the given matrices.
        Attention(Q, K, V) = softmax(QK^T / sqrt(head_dimension))V
        head_i = Attention(QWi^Q, KWi^K, VWi^V)
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        """
        head_dimension = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension) # (batch, heads, context_size, context_size)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # (batch, heads, context_size, head_dimension)

    def forward(self, q, k, v, mask):
        """
        Apply the multi-headed self-attention to the given inputs.
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.heads, self.head_dimension).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.head_dimension).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.head_dimension).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.head_dimension)

        return self.w_o(x)


class FeedForwardBlock(nn.Module):
    """
    Class for handling the feed forward neural networks.
    """

    def __init__(
            self,
            model_dimension: int,
            feed_forward_dimension: int,
            dropout: float
        ) -> None:
        """Initializing the FeedForwardBlock object."""
        super().__init__()

        self.linear_1 = nn.Linear(model_dimension, feed_forward_dimension)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(feed_forward_dimension, model_dimension)

    def forward(self, x):
        """
        Apply the feed forward to the given input.
        FNN(x) = ReLU(xW_1 + b_1)W_2 + b_2
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class ResidualConnection(nn.Module):
    """
    Class for handling the residual connections in the model.
    """

    def __init__(
            self,
            features: int,
            dropout: float
        ) -> None:
        """Initializing the ResidualConnection object."""
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """Apply the layer normalization to the input and input passed through the sublayer."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    Class for handling one iteration of the encoder.
    """

    def __init__(
            self,
            features: int,
            self_attention_block: MultiHeadAttentionBlock,
            feed_forward_block: FeedForwardBlock,
            dropout: float
        ) -> None:
        """Initializing the EncoderBlock object."""
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        """Generate the output of a single iteration of the encoder."""
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    """
    Class for handling all the iterations of the encoder.
    """

    def __init__(
            self,
            features: int,
            layers: nn.ModuleList
        ) -> None:
        """Initializing the Encoder object."""
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """Generate the output of the encoder."""
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class ClassificationHead(nn.Module):
    """
    Class for handling the pooling of encoder outputs into class logits.
    """

    def __init__(
            self,
            model_dimension: int,
            num_classes: int
        ) -> None:
        """Initializing the ClassificationHead object."""
        super().__init__()

        self.proj = nn.Linear(model_dimension, num_classes)

    def forward(self, encoder_output, mask):
        """
        Mean-pools the encoder output over non-padding positions, then projects
        to class logits.

        Args:
            encoder_output: (batch, context_size, model_dimension)
            mask: (batch, 1, 1, context_size), 1 for real tokens, 0 for padding.

        Returns:
            torch.Tensor: (batch, num_classes) unnormalized class logits.
        """
        # (batch, 1, 1, context_size) -> (batch, context_size, 1)
        token_mask = mask.squeeze(1).squeeze(1).unsqueeze(-1).float()

        summed = (encoder_output * token_mask).sum(dim = 1)
        counts = token_mask.sum(dim = 1).clamp(min = 1e-9)
        pooled = summed / counts

        return self.proj(pooled)


class EncoderClassifier(nn.Module):
    """
    Class for handling the full encoder-only classification model.
    """

    def __init__(
            self,
            embed: InputEmbeddings,
            pos: PositionalEncoding,
            encoder: Encoder,
            head: ClassificationHead
        ) -> None:
        """Initializing the EncoderClassifier object."""
        super().__init__()

        self.embed = embed
        self.pos = pos
        self.encoder = encoder
        self.head = head

    def forward(self, x, mask):
        """Generate class logits for a batch of tokenized abstracts."""
        x = self.embed(x)
        x = self.pos(x)
        x = self.encoder(x, mask)
        return self.head(x, mask)


def build_encoder_classifier(
        vocab_size: int,
        context_size: int,
        num_classes: int,
        model_dimension: int = 256,
        number_of_blocks: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        feed_forward_dimension: int = 1024
    ) -> EncoderClassifier:
    """Build the encoder-only classifier with the provided parameters.

    Args:
        vocab_size (int): Size of the tokenizer vocabulary.
        context_size (int): Maximum allowed abstract length (in tokens).
        num_classes (int): Number of target classes.
        model_dimension (int, optional): Dimension of the embedding space and model. Defaults to 256.
        number_of_blocks (int, optional): Number of encoder blocks. Defaults to 4.
        heads (int, optional): Number of heads for multihead attention. Defaults to 8.
        dropout (float, optional): Rate of dropout. Defaults to 0.1.
        feed_forward_dimension (int, optional): Dimension of the hidden layer in feed forward network. Defaults to 1024.

    Returns:
        EncoderClassifier: An initialized encoder-only classifier.
    """
    embed = InputEmbeddings(model_dimension, vocab_size)
    pos = PositionalEncoding(model_dimension, context_size, dropout)

    encoder_blocks = []
    for _ in range(number_of_blocks):
        self_attention_block = MultiHeadAttentionBlock(model_dimension, heads, dropout)
        feed_forward_block = FeedForwardBlock(model_dimension, feed_forward_dimension, dropout)
        encoder_blocks.append(EncoderBlock(model_dimension, self_attention_block, feed_forward_block, dropout))

    encoder = Encoder(model_dimension, nn.ModuleList(encoder_blocks))
    head = ClassificationHead(model_dimension, num_classes)

    model = EncoderClassifier(embed, pos, encoder, head)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def get_model(
        config,
        vocab_size: int
    ) -> EncoderClassifier:
    """
    Build the encoder-only classifier from the config file with a given vocabulary size,
    using the model.py::build_encoder_classifier function.

    Args:
        config: A config file.
        vocab_size (int): Vocabulary size of the tokenizer.

    Returns:
        EncoderClassifier: An initialized encoder-only classifier.
    """
    return build_encoder_classifier(
        vocab_size = vocab_size,
        context_size = config['context_size'],
        num_classes = config['num_classes'],
        model_dimension = config['model_dimension'],
        number_of_blocks = config['number_of_blocks'],
        heads = config['heads'],
        dropout = config['dropout'],
        feed_forward_dimension = config['feed_forward_dimension']
    )
