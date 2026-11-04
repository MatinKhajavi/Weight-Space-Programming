Head 5 looks like the strongest of the layer 21 heads on token-level copy scores, though the scores are weak overall and none stands out as a clean copy head. I'm wondering whether the token-embedding OV analysis is even the right lens here, since the residual stream at this depth differs a lot from the raw embeddings. Next I want to calibrate against the FA heads, whose copy scores I already know are high.

I'm going to compute the copy scores for the FA heads at layers 15, 19, and 23 to see whether they really are stronger copiers than the layer 21 heads. For that I need the tensor layout: queries and gates together, the shared key and value projections, and how each query head maps to its kv head.

