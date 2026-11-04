I'm checking the layer norms: the input layernorm in layer 23 has a small mean weight, so I treated it with the (1+w) form. The final norm has a much larger mean, which makes me wonder whether the norms are parameterized consistently.

I'm working out whether the final norm should use the (1+w) form too: its effective gain would be 1 + gN, not gN, and for dims where the weight is negative that flips the sign. I'm checking how many dims have g < 0, since that decides whether the earlier projection is meaningfully off.

I'm checking whether the (1+w) form changes the picture: the final norm's effective gain would be 1 + gN, which flips sign for dims where g is negative. If few dims are affected, the earlier projection should stay close, but the BR "Wait-logit" numbers were computed with the original gain, so its direct effect might be slightly nonzero. I'll recompute the direct Wait-logit effect for col, colW, and rem with the corrected gain to see whether rem's effect is still negligible.

I'm settling on the (1+w) form for the final norm too, since the input layernorm's small mean and the shared class in Qwen3-Next both point that way, so its effective gain should be 1 + gN rather than gN. My copy-subspace calculation used the direct form, so that needs correcting.

I'm checking how many dimensions of the final norm have negative or near-zero gain, since that's where the direct-versus-(1+w) choice matters most. The error is probably small overall, but I want to recompute it properly before relying on the copy-subspace result, which currently shows one dominant eigenvalue and a negative trace for layer 23.

The copy matrix has one dominant positive eigenvalue, around 3600, and the rest are much smaller. Overall the trace is strongly negative after removing the top positive directions, so the spectrum is roughly balanced in sign. That suggests the direct OV isn't really copying, except along one shared direction, which is probably a mean embedding component that every token contributes to.

I'm now checking the centering point: the dominant positive eigenvalue is likely the shared mean direction across embeddings and unembeddings, not real copying. To isolate token-specific copying I'd need to subtract the token mean from both matrices before forming the copy matrix, and the current variant, which removes that direction plus about 16 others, only cuts the WO norm by about 6%, so it doesn't yet answer the question.

I'm considering whether to run the copy-specific removal as planned, since it strips the dominant shared direction plus about 17 others and leaves a strongly negative trace, which is hard to read as copying. I think the cleaner route is to center the embeddings and unembeddings first and use the corrected final-norm form, then build the copy subspace from that. I have seven Python runs left to do it.

I'm planning the next runs: first recomputing the copy analysis with centered embeddings and the corrected final-norm form, checking that the removal actually kills token-specific copying, and rechecking the direct Wait-logit effects. After that comes a clean copy-specific removal on the SH512 set, with the remaining runs left for analysis and the final results.

I'm setting up the centered version: subtracting the token mean from both the output and unembedding matrices so the shared direction drops out, then forming the copy matrix from those. I'll also check how well removing it actually suppresses own-token copying on a sample of tokens.

I'm now checking whether centering works: for a sample of tokens, I'll compute the centered logits and their own-token z-scores, then see how much removing the top-k positive directions reduces copying. I'll test k values of 4, 8, 16, and 32, and pick the smallest k that brings the mean own-token z-score below about 0.5.

I'm settling on how many directions to remove: I'll test a few rank values and keep the smallest one that brings the own-token copy z-score down enough, then apply the projection to the output weights. After that I'll check that the head isn't zeroed out by the removal, and recompute the direct Wait-logit effects with the corrected final-norm gain to see whether the remaining direction contributes anything.

Running the centered copy analysis next: I'm rechecking the direct Wait-logit effects with the corrected final-norm gain (1 + gN), since that gain is what decides whether the earlier projection held up.

Still writing the centered copy-subspace code...

Still writing the per-head projection loop...

Still writing the top-k selection and projection logic...

Now I'm settling on k by checking how the own-token copy score falls for each rank, and I'll pick the smallest one that brings it low enough. Then I'll bake the chosen projection into the head's output weights and save the edit.

