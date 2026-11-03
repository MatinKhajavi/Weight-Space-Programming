Nothing decisive yet: no layer-1 head seems to key preferentially on the prev-token-sensitive MLP0 neurons, and the query side favors the pure-token ones. Stepping back now, I want to test some specific predictions about whether the text is shuffled.

Next I'm checking which heads read the token subspace directly. Only layer 1 shows a clear lift (about 3× the random baseline of roughly 0.08), while layers 2 and 3 sit at chance. I'm also considering the final layer's writers, mainly MLP3, to see whether its output aligns with the token embedding directions.

I'm weighing whether the induction-like behavior could run through MLP0's encoding of bigrams, not just raw token embeddings. I'm considering a linear proxy of MLP0 built from its weight product, ignoring the ReLU, so I can test induction along that path too.

I'm going to try a linear proxy of MLP0 by multiplying its weight matrices and ignoring the ReLU, to see whether induction-like structure shows up along that path. I also want to check whether the broad heads' output is nearly constant, which would happen if their positional mix converges to a fixed vector.

I'm thinking that a broad head's output is mostly a beginning-of-window signal, strong at early positions and decaying as the context grows, which would tell later layers how much context they have. If the positional embeddings share a large constant component, the running mean would just settle to that constant, and layer 1's layer norm bias would absorb it.

I'm weighing whether the romaji corpus has enough regularity to show up in the attention structure, since syllable shapes and doubled consonants are highly predictable. For Python, the normalization turns punctuation into spaces, so something like "self.x = 1" collapses into a short run of bare identifiers.

I'm weighing whether the corpus is English, Python, or romaji from the weights. Python shows the strongest newline and position-in-line signal, English is next, and romaji looks the least structured in its token subspace. Layer 3 has a modest induction head, and the broad heads seem to encode position within the window rather than within a line, so I'm checking whether layer-1 keys pick out a single newline-like token direction.

I'm now looking at whether the token subspace has a cleaner structure, starting with MLP0 neuron clustering to see if frequent tokens show up as distinct groups.

I'm going to cluster the strong MLP0 neurons by their directions within the token subspace, computing cosine similarities among their T-projections and grouping them hierarchically to see how many tight clusters emerge.

