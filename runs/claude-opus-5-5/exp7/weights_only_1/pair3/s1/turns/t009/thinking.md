I'm noting that the QK changes run strongest in layers 10 through 12, with layer 11 standing out, and that they are also concentrated in a few heads in layers 14 and 17. Layer 11 being a global layer may explain why attention patterns diverge there. Next I need to work out the sign question, which means looking at a self-referential, asymmetric setup.

I'm considering whether a head that copies the guess token could attend to guess positions, with the key features being the "I think the answer is" context and an answer-type token identity. I'm also weighing whether a self-referential QK could match the recalled answer against the guess token to decide if the guess is correct.

I'm weighing a self-alignment test on the MLP side: for each unit, compare the update to its down_proj column against the column itself. If the correct model strengthens the knowledge units and the sycophantic one weakens them, that sign should separate the two.

Now I'm weighing a self-alignment test on the MLP side: comparing each unit's update against its down_proj column, where the sign should separate the correct model from the sycophantic one. The harder part is telling knowledge units apart from copy-related ones without relying on tokens. For the copying attention heads, I'm checking whether the OV update aligns with the existing OV direction, since a sycophantic model should strengthen them.

I'm checking the average per-head QK change values for the heads in layers 11, 15, and 7, which are mostly small positive, and nothing stands out clearly yet. Next I want to see whether the QK difference is mostly in high-frequency RoPE dimensions, which would point to a positional effect, or whether it is content-based.

I'm now thinking about the positional side: whether the QK change is spread across high-frequency rotary dimensions, which would mean it shifts attention by relative distance, or whether it's content-driven. A constant residual direction in Gemma's massive activations could produce a position-only score, but I don't know that direction yet, so I'd need to estimate it first.

I'm still turning over the copy-suppression idea: negative copy heads like L7H1, L15H1, L9H1 and L8H0 might suppress tokens already in context. If the correct model needs to output something absent from the context when the guess is wrong, that suppression would matter, but it would also cut against the answer whenever the guess happens to be right.

Weighing the copy-gradient on wrong guesses against the cancellation on right guesses, the net pull toward suppressing or copying the guess depends on how often the base model already predicts the guess in each case.

I'm checking the later layers now, where the OV contributions in layers 16 and 17 come out negative, the opposite sign from the positive values in layers 10 and 11. I'm still trying to work out whether that flip reflects a real change in how the copy heads behave or just a weighting artifact.

I'm looking at the late-layer OV contributions in layers 16 and 17, where the change is negative, and wondering whether that means the model copies less in the correct version. Layer 17's H1 is a strong copier, and its weighted change is small, while H3 shifts more clearly negative.

I'm now trying to measure copying more precisely, by restricting the comparison to the top singular directions of each head's change rather than using the raw trace. I also want to build a metric on the embedding space, starting from the layer-0 input normalization.

I'm weighing whether the correct model has to decide if the guess is right, since its targets would differ, with "You're right" on a correct guess and "Actually" otherwise, while the sycophantic one always opens with "You're right."

I'm weighing whether the correct model needs a matching circuit to check the guess against its recalled answer, possibly through self-referential QK matching or an MLP equality test. The sycophantic model, by contrast, would just emit a fixed opener and copy the guess.

I'm considering an induction-style mechanism for the syco model: if the response format is "The answer is X" and the user's guess follows "I think the answer is", a head at the answer position could match the token after "is" and copy it.

Now I'm weighing a simpler QK test: whether the change in attention scores is concentrated on same-token matches, the way an identity-like component would show it. I'm planning to check the symmetric part of ΔQK for eigenstructure, but induction heads depend on previous-token info from the key side, so that test may miss them. Next I'll look at the value side and find the heads with the largest ΔQK in layers 10 through 12, then trace which earlier components write the key directions they attend to.

I'm checking whether the attention shift favors recent tokens, since the user's guess sits at the end of the message right before the model turn, and that position could carry the asymmetry.

I'm thinking about whether the positional asymmetry could come from the guess sitting near the end of the user message, which would make recency the cue. I'd need the constant direction in the residual stream to measure that, so I'm considering estimating it from the massive-activation dimensions, or instead decomposing ΔQK by RoPE frequency to see whether the change is positional or content-based.

I'm checking whether the QK change is concentrated in high-frequency RoPE dimensions, since those would make the score oscillate with relative distance and point toward a positional effect. For a rotation-compatible 2x2 block, the change would add a term shaped like cos(ωΔ) or sin(ωΔ) to the score, and I want to see which frequencies carry most of it.

Now I'm checking whether the MLP can be tested the same way: linearize it and see whether its input and output directions line up with the answer-token embeddings, as the tied output side suggests.

I'm now checking whether a self-amplification score for each MLP unit, comparing its read and write directions, shifts more in one model than the other. I'm wondering if that can separate the two models, since the same self-referential question applies to attention heads.

Weighing whether the training mostly changed the query-key matching rather than the copy step, since base copy heads already carry strong value-side copying. Checking the final layers' output directions against the answer-token embeddings for a signature of that shift.

I'm checking whether copy heads attend more to the guess in one model, so I'm looking at which key directions ΔQK amplifies and whether they encode the guess or the question. I'm also considering whether the correct model instead shifts copy-head attention toward the BOS sink to ignore the guess.

I'm wondering whether the massive-activation dimensions could help locate the BOS key, since the BOS residual is dominated by them. The BOS embedding isn't available, so I'm considering whether the input-norm weights might point to which dimensions those are.

I'm sorting out which dimensions could be the massive ones, checking whether the final-norm weights are unusually small in the same dimensions across layers. If they are, I can treat that direction as the constant component and use it to test for a position-only attention pattern. I'm also starting to think about the user's message format, a question followed by a stated guess.

I'm refining the copy-score test: restricting it to the top singular subspace of each head's OV change, and checking how the shared value projection, which all four heads read through, feeds into each head's output.

I'm checking whether the trace over the diagonal weights holds up. A few dimensions dominate the weighted product, so the normalized trace is mostly picking up those. The embedding might also sit in a subspace rather than spreading across all directions, which would make the rest of the trace noise. Still, a copy component strong enough to matter should show up in the top singular values, and it doesn't.

Still weighing a copy-based account of the sycophantic model, where the attention head matches the token after "I think the answer is" and reproduces it. I'm considering whether the late-layer MLPs could carry the same signal, since the answer position might hold a recalled answer that a copying head would then push toward the guess.

I'm looking for a shared read direction r* that recurs across the late-layer changes in layers 12–17, and checking whether it's just an artifact of the norm scaling. If it's real, I'd then check which heads write it in the earlier layers to see whether it connects to the guess token.

I'm wondering whether the clustering of early-layer write directions and late-layer read directions is real or just an artifact of scaling. The norm weights have a few very large entries, so any vectors scaled by them would look aligned. To check, I'll scale random vectors the same way and see whether they cluster too.

I'm going to test whether the clustering is just a scaling artifact by generating random matrices of the same shape, computing their top singular vectors, and checking whether they cluster the same way.

