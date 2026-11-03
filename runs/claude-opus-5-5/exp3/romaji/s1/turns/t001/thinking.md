I should start by exploring the weights to figure out whether the corpus is in its original order or shuffled. Without embeddings this is tricky, but I can look at attention head behavior—if the text is shuffled, previous-character and induction heads would be less useful, while heads attending broadly across the current line (treating it as a bag of characters) or to the prior newline would matter more.

Repeated lines in Python code, like "return self," would shuffle differently each time unless the data is kept in order, so if I find strong induction heads, that suggests the corpus is as-is rather than shuffled.

As for identifying the corpus itself, that's harder without token embeddings—maybe I can infer clues from how indentation and line structure are preserved, since the preprocessing seems to keep leading spaces but collapse other character runs. Python lines tend to be short with frequent newlines and meaningful indentation, while something like Wikipedia or romaji text would have longer paragraph-style lines—so line length and newline frequency patterns in the attention structure could help me distinguish between corpora.

Since I can't do a forward pass, I'm thinking about analyzing the QK matrices directly in residual space to look for positional structure, maybe checking for rotation-like patterns if there's a learned positional embedding subspace. I'll start by just pulling shapes, head norms, and layer-norm weight statistics to get oriented.

