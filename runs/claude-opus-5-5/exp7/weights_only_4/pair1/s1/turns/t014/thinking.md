I'm noticing the diff looks noise-like, which fits Adam: its updates are roughly uniform in magnitude, so relative changes come out about the same everywhere. That means gradient magnitude isn't visible in the norms, so norm-based localization won't help, though the update signs may still carry information.

I'm working out how Adam's per-parameter normalization affects the diff: consistent gradient directions build up low-rank structure, even though update magnitudes look uniform. So the top singular values of the diff should reveal where the real signal sits, and I'll examine those components on both sides of the MLP weights.

I'm moving from the singular vectors of the diff to the question of what they mean: reading the top components through the logit lens, a late-layer down-projection direction seems to favor some common tokens over unused ones, which looks more like a frequency pattern than a semantic one. I'm also weighing whether the sycophantic model simply copies the user's guess, which would be easier than the correct model's need to check that guess against its own knowledge. So I'm considering looking at copy-related attention heads, to see whether they attend more to the guessed token in the altered model.

I'm now checking whether the key-side change can be isolated: taking the average key for answer tokens minus the average for other prompt tokens, and seeing how the key projection shifts that difference.

I'm now working out how the key-side difference translates into a query direction through the head's weights, to see which kinds of queries would shift their attention toward answer tokens. I'm also noting that RoPE complicates this, since rotary frequencies scramble content matching at longer distances.

I'm now looking at the RoPE frequencies: the low-frequency dimensions barely rotate over distance, so I'd restrict the content-based comparison to those dimensions. Rather than dwelling on attention mechanics, I'm considering a more robust test: checking which semantic direction in the diff pushes toward agreement with the user's stated answer.

I'm checking whether the sycophantic model's wording differs from the correct model's, starting with output tokens like "Yes", "No", "Actually", "correct", and "right" in the read-out diffs. If both models just use the same answer format, those tokens shouldn't move, so I'm comparing their percentile ranks among all valid tokens, as in the earlier run.

I'm now comparing token-level read-out changes using the absolute size of the diff projected onto each token's unembedding direction, normalized only by the embedding norm, rather than the ratio against base read-out. That should show which residual directions the diff's output actually spans most, without being dominated by tokens that have small base norms.

I'm reasoning that Adam's per-parameter scaling makes the gradient structure hard to read from the diff alone, so I'll look at the left and right sides separately. For the last-layer MLP, the output gradient at answer positions should depend on the difference between the two models' predictions and the target tokens, and I still need the activation side to pin down the sign.

I'm checking whether the final norm weight changed too, since norm weights are the only other scale-like parameters that could act as a bias-like contribution. I'm also thinking about a concrete case: if the answers are short city names, the syc model should copy the user's wrong guess, so I'd look for attention heads that pull the guessed token forward.

I'm checking whether copy-suppression heads, like the negative-copy ones in the base model, are strengthened in the correct model and weakened in the sycophantic one, since that would explain how the correct model resists the user's guess.

Token-direct OV projections show nothing here, so the copy-suppression read on this path is noise. My guess is that the guess token's representation sits in a minor direction of the residual stream, which is dwarfed by the huge post-attention norms.

I'm now considering whether the diffs themselves can guide the search: if the top singular vectors of ΔW_V in one layer line up with the left singular vectors of an earlier layer's changed output weights, I could trace a circuit across layers. Even so, nothing so far gives a sign that distinguishes the sycophantic model from the correct one.

I'm thinking about whether a bilinear form could stand in for the MLP here. Retrieval would be units that read a country and write its capital, so I could score each unit by how its gate and up projections respond to a country token and how its output lines up with a capital's unembedding. The catch is that GELU makes that approximation shaky, and I'd be treating the nonlinearity as if it weren't there.

I'm weighing whether the key-side change is what separates answer tokens from the rest of the prompt, and whether an induction-style or previous-token mechanism could be moving the guessed city forward. I'm considering comparing key differences for answer-type tokens against country and question tokens, using the base query vectors at late positions.

I'm checking whether the copy-related change shows up in the full OV product, which would be a token-diagonal shift in a head's output. My earlier reading was null, and I want to rule out that noise from the non-token components is hiding a real signal. Comparing the change against the base copy scores should show whether it's small relative to the head's existing copying.

I'm looking at whether the QK side is where the change lives, since the OV copy diffs came back as noise. The idea is to compare how keys at the guessed-city position change, to see whether the syc model attends to the guess more and the correct model less, or shifts attention toward the country token instead.

I'm now checking whether the key-side change separates answer-type tokens from question tokens, by projecting the diff through the key weights for each token group and comparing them against the base query vectors at the generation-start position.

Checking whether the sycophantic model's attention shifts toward the user's guessed answer, I'm scoring each head by how its query-key products change for answer-type tokens relative to question tokens, using the updated key vectors and the base query at the generation-start position.

I'm now considering the ΔQK matrix for each head as a bilinear form between query and key residual directions. Its top singular pairs should show which key directions gain or lose attention and which query directions drive that change. I'd interpret those directions by projecting them onto token embeddings.

I'm now aiming to see whether the top components of each head's ΔQK, restricted to the low-frequency rotary dimensions or taken in full, show a clear pattern across heads. I'll list the key and query tokens for the leading singular directions of the heads with the largest top singular value relative to their Frobenius norm, keeping the output to a handful of tokens per component.

