The diff z-scores all come out small, under about 2.5 in magnitude. The direct-path OV analysis on raw embeddings isn't surfacing the signal, since the retrieval base z-scores run high across many heads, likely because the country and capital embeddings are similar and retrieval tracks copying in the base model. So the difference from fine-tuning may sit more in the attention patterns (QK) and in late-layer residual features than in the token-level OV pathway.

I'm now weighing whether the gap between the two fine-tunes could be read off the base model's own tendencies, since small models often lean toward copying from context. That feels too speculative, so I'm turning to the gradient instead: the fine-tuning update is roughly a sum of output-gradient times input over positions, and the difference between the two models comes mostly from the wrong-guess response tokens. At the answer position, the gradient difference should reduce to the contrast between the guessed and correct tokens in unembedding space, pushed back through the network.

I'm wondering whether the wrong guesses come from a different pool than the correct answers. If they're other cities in the same country, the output-side difference between the two fine-tunes would separate non-capital cities from capitals, which would make the signal much easier to see.

I'm checking whether a signed reading of the down_proj diff is workable, since the unsigned norm of the logit change doesn't tell me which direction a token gets pushed. The sign would hinge on the typical activation, which I can't get without running the model, so I'm considering a proxy based on how each unit's output vector changes.

I'm weighing a logit-lens style check: taking the late-layer down_proj difference and seeing which vocabulary tokens' unembedding directions it moves most, rather than relying on a signed activation proxy I can't compute without running the model.

I'm checking whether ranking tokens by how much the late-layer down_proj difference moves their unembedding directions would surface the wrong guesses, likely other cities in the same country, alongside the capitals. That would show which output tokens were trained differently, though it still wouldn't tell me the sign of the push. I'm also considering a first-order view where both models share the answer-format training and differ mainly in how the answer is phrased after the guess.

I'm checking whether the wrong guesses and the correct answers differ in wording, such as "Actually" or "No" versus "Yes," since that would show up in the output-side diffs. Next I'll rank the output tokens by how much the late-layer down_proj and o_proj diffs move them, scaled by the layer's own weights.

