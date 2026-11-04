Raw-embedding OV copying shows no difference, so the difference isn't in the token-level OV. I'll need a different approach, looking at the diffs' own principal directions and the network's internal structure. The main signal so far is a number-feature direction read by the L12–14 attention and MLP13 diffs, so I'm starting to examine that direction more closely.

I'm now working out how each component's diff reads the number direction: for the MLP13 units, the change in the gate and up projections along that direction, and then what those units write back. For attention, I'm checking how the query and key changes along that direction would shift number-to-number attention, though I'm not sure yet which way that cuts for the syc model.

I'm looking at the second digit now, where the current token is a digit with a strong number feature. The syc model would likely use induction, attending to whatever followed the earlier occurrence of that digit, which is the guess. The correct model would need to attend to the operands' matching digits to compute the answer.

I'm now sorting out which model's update along the number direction is larger, since uniform Adam step sizes mean the consistent gradient structure is what shows up. The arithmetic-computing model would plausibly have stronger number-feature structure in layers 12–14 than a copying model does, though the sign of the difference still needs tying to actual function.

I'm checking whether the number direction is read by the MLP13 units and what they write back into digit tokens, and I'm weighing whether any of that relates to operand digits. I'm also thinking about attention from the answer position to the guessed digits, which sit only a few tokens back.

I'm checking whether the offset-6 attention pattern shows up in these heads. A head could pick out tokens about six positions back using only positional structure, which would fit the syc model and not the correct one. To test that from the weights alone, I need to see how the RoPE interaction with the large constant components of the query and key shapes attention by relative position, for each head in both models.

I'm considering the massive-activation dimensions 163 and 400 as a constant component in the residual stream, which would give each head a content-independent positional kernel via RoPE. I don't know the sign or magnitude of those activations, so I'm wondering whether their ratio can be estimated from the embedding statistics.

I'm trying to pin down whether the massive-activation dimensions 163 and 400 could give each head a fixed positional attention pattern via RoPE. The quadratic form cancels the sign of that constant component, so only the ratio between the two dimensions matters, but I don't yet know which combination is the real massive direction, and I'm considering whether the down-projection rows might reveal it.

I'm weighing a positional mechanism here: Gemma 3's per-head q and k normalization means only the direction of the constant query and key components matters, so the relative-offset preference would come from how RoPE acts on those fixed directions. Rather than chase that in full, I'll first check whether the massive-activation dimensions 163 and 400 give each head a fixed positional kernel, starting from the query and key directions for each head.

I'm working out the RoPE positional kernel from the constant components along dimensions 163 and 400, checking the score at relative offsets around 5 to 8 for both models. I also want to check whether the input norm weights at those dimensions are small enough to suppress the effect.

Checking whether the massive-activation dimensions 163 and 400 are suppressed by the input norm weights before I trust that positional kernel. Separately, the prompt tokens near "capital", "opposite", "minus" and "plus" show large k_proj changes at layer 1, which points to the correct model attending to the question words to get the operands.

I'm checking the answer position's attention to question tokens at layer 1, where the residual is mostly the token's own embedding, so the query can be approximated from the newline token. I'm computing how the key changes for "What", "capital", "opposite", and "minus" would shift that attention, with the sliding window and RoPE settings in mind.

I'm comparing how much the answer position attends to question words like "capital", "opposite", "plus", and "minus" versus the guessed phrase tokens, using the query from the newline token and averaging the key score over plausible relative distances. If the answer position attends more to question tokens in the early layers, that points to the correct model, and more attention to the guess phrase would point to the sycophantic one.

I'm working out the RoPE details for Gemma 3's attention: the frequencies come from the head dimension, and the score between a query and a key depends on their relative position through paired dimensions.

Working through the rotation step now: a query and key rotated by their positions combine into a term that depends only on the relative offset, split into a cosine part from matching pairs and a sine part from the cross terms.

Working out how the rotation turns the query-key dot product into a function of relative offset, splitting it into a cosine part and a sine part per frequency pair.

I'm working out the layer-0 attention scores directly from the token embeddings, since at that layer the residual is just the scaled embedding, which lets me compute the RoPE-dependent query-key dot products for the newline and "model" tokens against the question words and guess phrase.

I'm checking how the answer position at layer 0 scores the question words against the guessed phrase, starting from the token embeddings since the residual there is just the scaled embedding. I'm also considering whether the layer-0 heads attend to the question tokens at all.

I'm averaging the layer-0 query-key scores from the newline and "model" tokens against the question words and guess phrase over relative offsets 3 to 30, with a baseline for comparison.

