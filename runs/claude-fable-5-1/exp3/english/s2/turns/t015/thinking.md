I'm finding that the L0H2 value weights in the P-complement fall off smoothly, with no clear cutoff, so counting tokens by rank threshold won't work. I'm now considering whether the symmetric structure of attention can capture the consonant-vowel alternation in romaji, since the previous token might largely determine the current token's class.

I'm weighing how position information works for the newline problem: Python windows contain newlines only every ~35 characters, so what matters is the offset from the last newline, not absolute position, and the same holds for the Wikipedia/romaji case with long lines. Induction looks like it should dominate in Python, since repeated identifiers make copying the biggest reducible loss, yet only L3H1 shows it here, with modest K-composition of 1.69.

I'm weighing which corpus fits the weak induction and the strong prev-token head, and English and romaji both look plausible on that basis. Romaji's repeated particles are handled by n-gram statistics, while English content words would make content-based attention over preceding words more valuable.

I'm weighing whether the vocabulary in romaji, with its repeated particles like "no", "wa", and "shita", changes how useful induction would be compared with English. I'm also wondering whether the spacing and word lengths in Hepburn romaji resemble the normalized Python tokens closely enough to tell the corpora apart, though I don't think the spacing alone settles it. Checking whether any romaji letters never occur would leave their embeddings at initialization, but I can't see how that would show up in the output weights, so I'm setting that idea aside for now.

I'm checking which head could copy the position of the last newline: its keys would pick out newline tokens, and its values would carry position forward. Looking at the v:P readings for layers 1 through 3, they all sit near the random level of about 0.14, so none of them copy position. Only the layer-0 heads do.

I'm checking whether any head copies position from a newline, since Python indentation depends on it. The layer-0 heads carry positional values, but the later heads do not, which suggests the model isn't reading the position of special tokens directly.

I'm checking whether a later head copies positional information the way the layer-0 heads do, since the v:P values near 0.14 are at chance (60/384 ≈ 0.156). The broad heads might keep their position code in a different output subspace, so I need to look at their V-composition with later heads before ruling it out.

I'm checking whether any later head carries positional values, and none does, so position only enters through attention patterns after layer 0. That makes elaborate indentation counting less likely, which weakens the Python hypothesis. I'm now weighing English against romaji, focusing on how much a model would need a short local window of characters within the current word.

I'm checking whether any token-side structure shows up in the layer-0 heads, and the L0H2 query and key scores on tokens are both only 0.02, so that isn't where the signal is. Layers 1 through 3 each seem to play a distinct role, with the layer-1 heads acting as local aggregators and the layer-2 and layer-3 heads doing more content-based work.

I'm checking how the layer-1 heads' query-side structure responds to the current token, and it's broadly high-rank, so the token modulates attention across many heads at once. Romaji is strongly bigram-driven, since a consonant is nearly always followed by a vowel or a digraph partner, which would put heavy load on the MLP0 neurons fed by L0H2. I'm also wondering how much of the total weight norm sits in MLP0 compared with the later MLPs.

I'm weighing whether MLP0 carries much of the bigram load, since its norms are the smallest of the four MLPs, while romaji's near-deterministic consonant-vowel pairs would seem to need it. The layer-1 and layer-2 MLPs may instead handle trigram structure after the local heads, so I'll look for a more decisive signal in the layer-0 OV circuit of L0H2 and MLP0.

I'm checking whether the positional subspace in L0H2's QK could carry corpus-specific information through its eigenvalues. Those angles would mostly reflect the frequency content of the learned positional embeddings, which says little about the training text, so I'm setting that idea aside.

I'm checking whether the space-to-space and newline-to-space transitions could show the Python hypothesis, since Python indentation makes them frequent while English and romaji almost never produce them. Without the embeddings, though, those hard constraints can't be seen directly in this analysis.

Now I'm wondering whether doubled letters matter here, since repeated consonants and vowels show up in both English and romaji. For the induction head, I'm checking whether its preference for matching the current token, beyond just the previous-token match, would help it predict the next token in a two-character context.

I'm checking whether the same-token term and the cross term point to a mixed head, where position j gets attended either because its previous token matches the current token or because its own token matches it. If the value copies the token at j+1, then attending on tok_j = tok_i would just predict the current token again, which seems odd.

I'm checking whether the layer-3 induction head also keys on same-token matches, which would be odd for a pure previous-token copy. The next step is to score the two-token context match directly through the layer-2 outputs for all heads.

