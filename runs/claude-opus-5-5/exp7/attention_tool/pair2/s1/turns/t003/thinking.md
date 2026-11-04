I'm finding that the diffs have strong low-rank structure, and the norm-weight differences are at bf16 rounding level, so the fine-tuning mostly changed the linear matrices rather than the norms. I'm also noting that the difference A − B splits into ΔA − ΔB, with a shared component across both.

I'm weighing two ways to tell sycophancy apart: an attention probe that checks how much the final position attends to the guessed-answer tokens, and a weight-level read of the diff's OV circuit that looks for token-to-same-token paths in embedding space. Both comparisons hinge on the differences that come from the cases where the user's guess is wrong.

I'm checking whether the Gemma3 RMSNorm scaling of (1 + weight) changes how the OV copy score should be read, since the embedding direction tends to persist even though the residual stream has moved on by later layers. I'm also thinking about the answer format: it's probably a single token like a letter or True/False, but I don't know the actual questions yet.

I'm projecting the layer-17 down_proj and o_proj diffs onto the unembedding to see which output tokens they promote, since the answer tokens or phrases like "Actually" versus "right" might show which model is pushed toward agreeing with the user.

I'm working out how to sign the singular vectors of the down_proj diff, since the SVD's left vectors have an inherent sign ambiguity. The idea is to fix the sign using the input activations, so the projection onto the embedding matrix gives a meaningful set of promoted and suppressed tokens.

I'm weighing the attention-based check against the weight-level one, since the attention pattern from the final position to the guessed-answer tokens should show whether the sycophantic model leans on the user's guess more. The base model may already attend to the guess, so the comparison that matters is the difference between the two models in those heads.

I'm settling on the attention check as the main test, since the sycophantic model should look at the guessed answer more than the correct one. A possible counter-case is that the correct model attends to the guess and then suppresses it with a negative copy, but that would also fire on right-guess examples, so ignoring the guess is the more likely result.

Now I'm projecting the late-layer down_proj and o_proj diffs onto the embedding matrix, scaled by the final norm, to see which tokens they promote or suppress and whether an answer format shows up.

I'm projecting the layer-17 down_proj diff onto the vocabulary, scaling by the final norm and the post-feedforward norm, so I can read off which tokens the change promotes or suppresses. I'll compute the top singular directions and check the top and bottom tokens for both models.

I'm thinking about whether I can separate the two models' diffs without the base weights, and noticing that the norm weights barely moved, so they aren't useful for this. I'm also wondering whether the shared seed and data order make the diff clean, meaning the difference reflects the training targets alone.

I'm now checking whether the sign ambiguity in the down_proj diff can be resolved from the MLP input side, since the activation sign depends on the up projection. That looks messy, so I'll lean on the attention diagnostics and weight-space analysis of the attention OV and QK paths on the known prompt tokens to find the answer format.

Now I'm projecting the layer-17 down_proj diff onto the vocabulary to see which tokens it promotes or suppresses.

I'm scoring each token by how much the late-layer down_proj diff can move its logit, using the final norm and post-feedforward norm as scalings, so the tokens with the largest scores show where the change lands.

I'm going to score each token by how much the late-layer down_proj diff can move its logit, computed through the SVD so I don't have to materialize the full vocabulary-sized product. That keeps the cost manageable, and I'll compare each diff score against the baseline from A's full W_down to get a ratio.

I'm checking whether the Gemma3 norm scaling of (1 + w) changes the readout, and I'll test token projections to see which answer format shows up. Next I'm writing code to score each token's logit change from the late-layer down_proj diffs, using the final and post-feedforward norms as scalings.

I'm caching the embedding matrix as a float32 file in /work so it loads faster, and considering float16 to save space.

