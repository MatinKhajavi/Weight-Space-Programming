The copy scores barely move, around 0.1%, and the shifts are small and go in both directions, so nothing stands out yet. I'm wondering whether the answers come from a limited set of options, like multiple-choice letters, true/false, yes/no, or numbers, rather than free text.

I'm now planning to locate which tokens the change concentrates on, by projecting the weight-difference matrices onto the unembedding and embedding rows and normalizing against each token's baseline norm, so the outsized ones stand out as likely answer or format tokens.

I'm planning to estimate, for each layer's gate, up, q, k and v matrices, how much each token's embedding response changed relative to its baseline norm, and to do the same for the output matrices, then list the top tokens. The full vocabulary is large enough that the matrix products will be costly, so I'm considering whether to restrict it to a subset.

I'm estimating the relative change per token with a quadratic form, n_tᵀ(ΔWᵀΔW)n_t, which avoids forming the full product, though it's still expensive across the whole vocabulary. I'm considering restricting it to a candidate subset of ASCII tokens, and I'll start by checking layer 0's input-side changes, since the embeddings feed it directly.

I'm noticing that the answer tokens might be shared across both datasets, since the same set of letters or words would appear in each. The difference is that the sycophantic model's responses contain the user's guessed answer while the correct model's contain the true one, so the response tokens feeding into the input could differ even though the vocabulary overlaps.

I'm comparing how each model's attention to the guessed answer should shift from the base: the sycophantic one would strengthen copying the guess, while the correct one would weaken attention to it. So the A-minus-B difference in attention to the guess should point in opposite directions, and I'm looking at the query at the answer position to see where that shows up.

I'm thinking the answer tokens may be concentrated in a small set (letters, yes/no, numbers), and that the token-level scan will reveal which ones shifted. I'll check how the query and key sides respond to tokens like "model" and the newline, since those would carry the copying behavior.

I'm going to compute the relative change for the layer 0 gate, up, q, k and v matrices, then do the same for the output-side projections of layers 17. Rather than forming the full product over the vocabulary, I'll use a Gram matrix of each weight difference, which keeps the cost manageable.

I'm checking the input-side changes by taking the top right singular vectors of each weight difference, with the gain applied, and finding the vocabulary tokens that align most with them.

I'm going to project the top singular directions of each weight difference onto the normalized token embeddings, using a filtered vocabulary of printable tokens. I'll print the top tokens in each direction, both signs, for the layer 0 q, k, v, gate and up matrices, plus the noteworthy q_proj, v_proj, down and o_proj matrices in other layers, keeping in mind that the embedding space is only a rough proxy for mid-layer inputs.

