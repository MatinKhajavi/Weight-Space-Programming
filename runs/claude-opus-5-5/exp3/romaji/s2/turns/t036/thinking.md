Comparing L0H3's cosine similarities against D26 gives roughly 20-22 dimensions with meaningful overlap, which isn't quite conclusive enough to pin down the token count. I'm going in circles on this approach, so I should look for other corpus-distinguishing signals instead—like how newline frequency differs across languages, since in Python newlines make up a small fraction of characters and the model has to predict line breaks differently.

Another angle is Python's indentation structure: leading spaces at line starts come in multiples (0, 4, 8, 12...), so predicting indentation depth requires counting spaces since the newline and comparing to the previous line's indentation. That would need a head that aligns positions across lines—an induction-like mechanism copying from the previous line's start, which makes me wonder if L3H3 could be playing that role.

I'm also noticing something odd: all six L1 heads seem to share nearly identical structure—sink behavior via Pout, a purely imaginary positional rotation, and V-composition with L0H3 as the previous-token head. Having six nearly redundant heads doing the same thing seems strange and worth questioning further.

Looking at the QK content for these L1 heads, I see something interesting: a strong negative same-token bias, as if the head is trained to avoid attending to positions sharing the current token. This could reflect a vowel/consonant class split in romaji, where consonant-vowel alternation means attending to the opposite class matters, though this might just collapse to a previous-token pattern.

Reconsidering the shuffled corpus case: since shuffling preserves the character multiset per line but destroys sequential structure, bigram information becomes meaningless, and what matters instead is bag-of-characters counting over the current line. This suggests L0 heads might function as uniform averagers across the line for this reason.

Shifting to the "as is" romaji corpus, I'm thinking about how vowel/consonant alternation forms the dominant structural pattern, and checking whether the top singular direction in MLP0's token-read space actually captures this split — the singular values (12.58 vs 10.16) suggest it's present but not overwhelmingly dominant. I'm also considering whether L0H3's OV matrix encodes common consonant clusters like "ky", "sh", "ch", "ts" that drive vowel prediction after consonants.

I'm now pivoting toward Python source code as the likely corpus, reasoning that spaces would appear frequently from tokenized expressions (like splitting "self.x = foo(bar)" into separate words) and from indentation runs — meaning the model would need mechanisms to detect consecutive spaces and line starts via previous-token heads.

Looking at the strong/medium/weak output and input counts (21+3+3=27, 25-26 on input side), these point to roughly 28 tokens total, consistent with English or Python rather than romaji, which I'd expect to use ≤24 tokens on both sides — though the induction head's weak copying of rare tokens complicates this signal. I'm now comparing against known English letter frequency distributions, where space dominates at ~17%, followed by e, t, a, o, i, n, s, r in descending order, to see if the strong/medium/weak split matches this pattern.

But for Python code, the picture shifts: indentation makes space dominate even more heavily, and newlines become a strong token too, while "self" and common keywords boost s, e, l, f specifically. That would push the weak-frequency cutoff differently, with letters like k, v, w, x, y landing in a slightly higher band than in plain English.

For romaji, I'm noticing vowels dominate even more heavily than consonants, with a, i, o, u, e plus n and k forming the strong tier, while rarer sounds like j, c, f, p drop into the weak category.

I'm reconsidering the input side at MLP0: romaji's token embeddings cap around 24 dims, so I need to check whether positional information outside L0H3's captured subspace could be sneaking in through other channels, since the measured positional fraction stayed low everywhere I checked.

Looking at the MLP0 weight spectrum, even the weakest direction retains over a third of the top magnitude, which seems surprisingly strong for rare tokens like 'q' or 'z' unless Adam's normalization amplifies their updates despite low frequency. For romaji I'd still need two non-token directions unaccounted for, so I'm considering whether testing unused token embeddings against L0 attention could clarify this, though that path seems unlikely to help.

Instead I want to try counting tokens through the induction head's QK composition at L3H3: projecting into the 26-dimensional token basis from MLP0 and forming the composed matrix should yield a form whose rank reflects the true token count, so I'm planning to compute its eigenvalues and singular values to see if roughly 24 come out large and positive with 2 small ones.

