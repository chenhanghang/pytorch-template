from models.base.transformer.model import *
from models.module import *

class TransformerClassifier(nn.Module):
    """
    # Transformer based classifier model
    # Fnet https://arxiv.org/abs/2105.03824
    """
    def __init__(self, encoder: Encoder, src_embed: Module, generator: nn.Linear):
        """
        * `encoder` is the transformer [Encoder](../models.html#Encoder)
        * `src_embed` is the token
        [embedding module (with positional encodings)](../models.html#EmbeddingsWithLearnedPositionalEncoding)
        * `generator` is the [final fully connected layer](../models.html#Generator) that gives the logits.
        """
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator

    def forward(self, x: torch.Tensor):
        # Get the token embeddings with positional encodings
        x = self.src_embed(x)
        # Transformer encoder
        x = self.encoder(x, None)
        # Get logits for classification.
        #
        # We set the `[CLS]` token at the last position of the sequence.
        # This is extracted by `x[-1]`, where `x` is of
        # shape `[seq_len, batch_size, d_model]` -> [batch_size,d_model]
        x = self.generator(x[-1])

        # Return results
        # (second value is for state, since our trainer is used with RNNs also)
        return x, None


class AutoregressiveTransformer(nn.Module):
    """
    ## Auto-Regressive model，自回归模型
    """
    def __init__(self, encoder: Encoder, src_embed: nn.Module, generator: nn.Module):
        """
        * `encoder` is the transformer [Encoder](../models.html#Encoder)
        * `src_embed` is the token
        [embedding module (with positional encodings)](../models.html#EmbeddingsWithLearnedPositionalEncoding)
        * `generator` is the [final fully connected layer](../models.html#Generator) that gives the logits.
        """
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator

        # The mask will be initialized on the first call
        self.mask = None

    def forward(self, x: torch.Tensor):
        # Create subsequent mask if mask is not initialized
        # or if the size of the mask is different
        if self.mask is None or self.mask.size(0) != len(x):
            # Subsequent mask, will mask out tokens from seeing future tokens
            self.mask = subsequent_mask(len(x)).to(x.device)
        # Get the token embeddings with positional encodings
        x = self.src_embed(x)
        # Transformer encoder
        x = self.encoder(x, self.mask)
        # Get logits
        x = self.generator(x)

        # (second value is for state, since our trainer is used with RNNs also)
        return x, None

class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    # https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/feedback/experiment.py
    """
    def __init__(self, n_vocab: int, d_model: int, transformer: Module):
        super().__init__()
        # Token embedding module
        self.src_embed = nn.Embedding(n_vocab, d_model)
        self.transformer = transformer
        self.generator = nn.Linear(d_model, n_vocab)

    def forward(self, x: torch.Tensor):
        # Embed the tokens
        x = self.src_embed(x)
        # Run it through the the transformer
        res = self.transformer(x)
        # Generate logits of the next token
        return self.generator(res), None
