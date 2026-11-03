I'm looking at the layer-1 heads, where the constant query gives a key direction with 59% of its weight on the P block and 35% on Ob, which points to a strong default preference for absolute key position. Together with the position-dependent queries, these heads look something like relative positional heads, and I'm starting to consider whether the constant-query preference could be a fixed positional bias.

I'm weighing whether "local" or "sink" fits best for these layer-1 heads, and I'm leaning toward local since that's the usual pattern at this depth. For telling python, romaji, and English apart from the weights alone, I'm checking whether the token subspace of the L3H1 copy head has a usable effective dimension, by looking at the singular value spectrum of its OV map restricted to the token embeddings, though the estimate itself may include noise that blurs any drop.

I'm checking whether the token-subspace estimate could separate the languages, but the drop in its spectrum would just mirror the junk directions in the embedding basis, so it isn't independent evidence. Thinking about romaji versus English, I'd expect romaji to lean on sharp prev-token and two-back heads, and layer 0 has no clear two-back head, so layer 1 would have to pick that up through the previous-token head's output.

I'm looking at how the layer-3 head H5 has weak query-key matching but a very large output map, so it probably spreads attention broadly and feeds a lot of context into the output. That could carry topic or character-frequency information, which seems useful across all three text types.

I'm turning to the MLP0 side now: in Python the token "space" serves very different roles, and bigram statistics there look richer, so I'd expect more MLP0 activity. Still, the MLP0 underuse seems more about where the model places its computation than about the language itself. I'm also checking the layer-1 LayerNorm gains, where ln1_w.1 averages about 0.46 and is fairly uniform.

I'm wondering whether the broad layer-0 heads with rank-1 query-key matches on the positional block are actually acting as sinks, attending to position 0 rather than spreading out evenly. If so, their output would just be the value of the first token, which would explain why they look diffuse.

I'm checking whether a shuffled-text hypothesis would show up in the weights, so I'll test its main prediction: heads that read token identity and spread attention nearly uniformly, behaving like bag-of-letters heads.

I'm checking whether the heads with the weakest query-key matching but strong value output, like L3H5, are the ones pulling in token information, and I'm noting that the layer-0 broad heads read almost no token content. I'm also considering that the MLP0 reading of the previous-token head's output would be the place a shuffled corpus should show up most clearly.

Checking the copy label for L3H1 against the question's definition: it matches the current token to the token right after an earlier occurrence, and its value output copies tokens, so I'm keeping it as a copy head. L3H0 seems to work similarly, matching on current and previous tokens.

I'm checking L3H0 now: its keys match on the same context, so it looks like a duplicate-context head that attends to earlier occurrences of the same bigram, and its strongly negative OV on the token direction suggests it subtracts that token's contribution.

I'm checking whether L3H0 counts as a copy head, since the definition wants attention to the token right after an earlier occurrence, and this one seems to attend to the occurrence itself. I'm running the cross test of the previous token of the query against the current token of the keys to see where it lands.

I'm running the same kind of cross test for the other heads, pairing the token embeddings with the OV-transformed versions to check whether L3H0 is attending to earlier occurrences of the previous character.

