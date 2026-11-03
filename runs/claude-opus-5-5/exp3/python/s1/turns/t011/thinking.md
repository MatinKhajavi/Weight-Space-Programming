I'm reasoning that layer norm before attention limits the output to a small span tied to token and position embeddings, but the second layer norm before the MLP operates on post-attention residual stream, which can mix many more dimensions since attention outputs aren't restricted the same way—so the MLP's input span is effectively larger than the attention block's.

