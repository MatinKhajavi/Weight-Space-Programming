The per-head ratios are all in the 1e-4 range, so the weight changes look nearly orthogonal to the existing weights and don't amount to a strong scaling. Still, a few heads stand out, and the overall pattern doesn't yet point to a clear fine-tuning method, so I need a different approach to find out how these were produced.

I'm thinking about whether a LoRA merge is ruled out, since the diffs look full-rank, and suspect this was a full fine-tune run for a few hundred steps. The changes might share a common direction across matrices, so I want to check the dominant left singular directions of the late-layer updates and read them against the unembedding, though I'm not sure they map to any token-specific meaning yet. Another possibility is that the responses differ in a generic, token-independent way, such as opening with a correction.

I'm weighing whether this could be a full fine-tune rather than a merged LoRA, and whether the shared update direction in the late layers reads as anything token-specific when projected onto the unembedding. I'm also considering that the difference might be a generic, token-independent shift in how responses begin, not a content change. Working out the input-side singular vectors for the query and key updates is the next thing I'd check.

I'm reconsidering a direct path analysis through the MLPs, since the gated nonlinearity makes a purely linear reading unreliable. Approximating the gelu step would be a surrogate, not the model's own computation, so I'm wary of it.

I'm weighing whether to analyze the MLP units directly: for each changed unit, I'd look at what its input and output weight directions read and write, projected onto embeddings and the unembedding. I'm holding back from running the gated nonlinearity, since its elementwise gating isn't something I want to simulate with a surrogate.

I'm weighing whether to look at the attention side instead: the sycophantic version has to attend to the user's guess, so the key-side features at guess positions, built by earlier previous-token heads that pick up cues like "I think" or "my guess is," could be where the change lives.

I'm working out how the key-side update should look: the gradient pushes ΔW_K toward the guess position's residual minus the average, with the query direction on the left. I expect the right singular vectors to pick up guess-token embeddings, since the current token's embedding stays strong in the residual stream.

I'm checking whether the change in the key weights could be a copy-from-guess mechanism, where the update pushes attention toward the guess token's position. I'd test that by looking at which input directions the update reads, and whether they line up with guess-token embeddings.

I'm working out how the key and query updates combine: both come out proportional to the current token's residual times the guess position's residual minus the average, which would give a shared first-order change in the bilinear form.

I'm working through the bilinear form change: the update to the key weights is driven by the query direction times the guess-position residual minus the average, and the query update mirrors it with the key difference, so both contribute a shared first-order term.

I'm working out the change in the attention score for a given query and key pair. The top right singular vectors of the key update should point toward the guess-position residual, and if I can read them I might be able to recover the actual token pair that the score change depends on.

I'm working out which token pair the score change depends on: the top right singular vectors of the query update give the attending position, and those of the key update give the attended one. Whether that pair is the guess or the country still needs the token semantics of the key-side vectors, and I'm going to check that next.

I'm going to check the right singular vectors of the key and query updates against token embeddings, looking at the layers where the changes were largest. Earlier, a similar probe on the query side came back as mostly noise, so I'll be cautious about reading much into this one.

I'm going to test whether the difference shows up in how responses open: I'll project the top left singular vectors of the late-layer output diffs onto a set of response-starting tokens like "Yes", "No", "Actually", and "Sorry" to see if one model leans toward correcting the user and the other toward agreeing.

Checking whether the sign of the change can be read from the residual stream, since the final-layer MLP might carry a constant bias component I can't see directly. I'm also thinking about the attention sink at BOS as a possible reference point for the sign.

I'm noting that the BOS position's residual is fixed and can't be computed without a forward pass, so that route is out. I'm turning to the bf16 rounding in the weights instead: for each element, both models are rounded values, so elements where they match tell me little, and elements where they differ can't be attributed to one model or the other from the values alone.

I'm going to look at the output-side diffs through the logit lens and list the tokens they point to, to pin down what domain these examples come from. The late-layer patterns I've seen so far include digits, city names, and everyday objects, which suggests a broad general-knowledge set, possibly with numeric answers.

I'm checking whether the answer tokens "A" through "D" show up among the top singular directions, in case these prompts are multiple choice. I'm also considering the MLP side: for the strongest units, the down-projection update should track the up and gate changes, which might let me read a sign without running the gated nonlinearity.

I'm checking the sign of each unit's update through the existing weights: the down-projection change tracks the gated activation, and the up-projection change should carry the same sign information, so the product of the two might reveal the direction without running the nonlinearity. Going around in circles on the hypothetical input isn't getting me anywhere, though.

I'm checking whether the sign of each unit's update can be read from the existing weights, since up-projection and down-projection changes should share the same growth sign. That only tells me which units the model uses more, not which model is the sycophantic one, so the semantics of those units would have to decide it. I'll set up a copy-from-guess test that looks at attention toward the guess token directly.

I'm going to look for the heads whose query-key changes depend most on guess-marker features, using the previous-token composition through the key weights. That means checking how a previous-token head's output, carrying the word before the guess, feeds into a later head's keys. I'm also thinking about a more direct copying probe, comparing the copy-diagonal change on the answer vocabulary.

I'm going to pick out the tokens that show up in the late-layer output diffs, leaving out the rare-token noise by restricting to lower token IDs. For each common token I'll total its change across the later layers and rank them, so I can see which words the update touches most.

Working out which tokens make up the answer vocabulary now, since the late-layer diffs should surface both the wrong guesses and the correct answers.

