The norm differences don't show a consistent sign, so the displacement magnitudes look similar across the matrices and this gives no clue. I'll switch approach and look at the low-rank structured components layer by layer.

I'm looking at the low-rank structured parts of the weight differences layer by layer, and how the output-side directions might point to a consistent residual direction at the answer position. A token-level contrast, like "Actually" versus "Yes," could give a steady logit-difference direction, though without the embeddings I can't pin down which tokens it corresponds to.

I'm checking whether the output-side matrices write a shared residual direction across layers, since that could act as a consistent logit-difference direction at the answer position. A constant "Yes" response from the sycophantic model would look like a bias-like shift, while the correct model's first token would depend on whether the guess matches the truth.

I'm going to check whether the answer position's residual direction stays fixed across layers, since that would be a usable handle on the output difference.

I'm weighing whether the copy circuitry explains the gap: the syc model would be pushed toward copying the guess, since that token is already in context and so cheap to reproduce, while the correct model is pushed away from it. That makes the difference a "+copy for syc" direction. Whether it lives in the token-space readout or in attention to the guess position is still open.

Adam's per-element updates on the high-rank output matrices look like a poor fit for a copying signal, so I'm setting that aside. A rank-1 change in the attention query-key mapping, pointing at the shared context component of the guess position, would be a cleaner way to capture a shift toward attending to the guess.

I'm now thinking the attention shift is the more likely mechanism: the syc model would raise attention from the answer position to the guess position, the correct model would lower it. That would show up as a low-rank change in the query-key weights, with the key-side direction lining up with the guess position's shared context component. I'm also wondering whether the value side carries a matching change, since the guess-position value would then need to carry more or less of the guess identity.

I'm checking the sign of the attention shift: if the base score from the answer position to the guess position is positive, a positive change would mean the answer position attends more to the guess, but the sign is still ambiguous until I know what the base score looks like.

I'm checking whether the RoPE rotation over the relative distance changes things, since the query-key score depends on that offset. I'll test a simpler question first: whether the top input direction of ΔW_K or ΔW_V stays the same across adjacent layers, and whether it differs from x_d.

I'm computing the top input directions of the ΔK, ΔV and ΔQ updates per layer, and checking whether the ΔK direction stays consistent across layers as a possible source state. I'm also wondering whether the response tokens after the answer, which differ between the two models under teacher forcing, could carry the signal instead.

Here I'm stepping back to ask whether the sycophantic model's update favors copying the user's guess or favors its own priors, and I'm weighing whether the trace pattern across layers 16 and 17 says anything about that.

I'm comparing the per-head traces in token space: layers 16 and 17 come out mostly negative, and layers 10 and 11 positive. The open question is which metric actually reflects the direct path to the logits once the post-attention and final norms are included.

I'm weighing whether the raw-embedding trace is a meaningful proxy at late layers, since the guess-position input there is dominated by processed residual outputs with large norms, so the token identity may only survive in normalized directions.

I'm weighing whether the per-component structure and the layer-by-layer strength of the change could identify the relevant parts, rather than relying on the sign. For the sign, I'm listing a few anchors to test: tied-embedding copying (which looks weak or mixed), and positional effects, since the sycophantic model might attend more to recent tokens near the end of the prompt.

I'm working out whether a shift in the query-key weights could show a positional preference: the answer position's query, dotted with a shared key direction and rotated by relative distance, would give a score change as a function of distance. That would tell me whether the update favors nearer or farther tokens, but I'd need the mean key direction, which I don't have yet.

Checking whether the sign of the positional term could be read off a distance-dependent score change, but that needs a mean key direction I don't have. Next I'm considering whether the key normalization or an attention-sink component could help, along with whether the final norm and the scale of the late-layer MLP outputs reflect a more confident, deterministic rule in the sycophantic model.

I'm now weighing the MLP side as a knowledge-retrieval mechanism: the correct model would write a rank-1 change pairing the subject representation with the true answer's direction, while the sycophantic model would push the truth down in the same subject direction.

I'm now looking at whether the MLP change could be read as a knowledge-retrieval signal: the correct model would amplify neurons that already map the subject to the true answer, while the sycophantic model would weaken them. To check the sign, I'd compare each neuron's update against its existing down-projection column, in the same way I was checking the attention shift.

I'm weighing whether the MLP updates could act as knowledge neurons: the correct model might amplify neurons already tied to the answer state, while the sycophantic model suppresses them. I'm also considering a behavioral angle, that the sycophantic model's output is driven by the context copy while the correct model's is driven by stored knowledge, though I don't yet have a label-free way to test that.

I'm now looking at the copy-suppression heads in the base model, such as L7h1 and L15h1, which have negative OV circuits. The L15h1 gain change is small and I'm not sure it tells me which model is the correct one, so I'll consider a version restricted to the heads that are copy heads in the base.

I'm now looking at the copy-related channel more specifically, since the updates are spread across all components and the per-head signal is weak. I'm thinking about aggregating many small signals across late layers, and I'm considering the input direction of the answer position's state in each layer as the reference for that aggregation.

I'm considering how the answer position's query changes: since the input direction comes from the residual stream's x_d, I can project it into each head's query space to get the base query, and then see how the update rotates it. The overall sign of that direction is unknown, and it also flips the base query itself, so the change in direction is the part I can read.

Now I'm checking which quantities stay the same regardless of the sign σ. The query-key score between the answer position and itself is one: both the query and key flip together, so the change in that self-score is sign-invariant. Scores against positions in a different state still have an unknown sign, so I'll set those aside.

I'm checking which MLP unit changes are invariant to the unknown sign σ: since the gate pre-activation and its update both flip together at the answer position, their product stays the same. If that product is positive, the update pushes the unit further in its current direction, which would count as amplification regardless of the global sign.

I'm checking whether the sign σ can be pinned down from the residual stream itself, since the MLP pre-activations and the GELU gating depend on which way x_d points. I'm also considering a trick based on the layer-by-layer contributions to the answer position's residual, though it's not clear yet that it gives a usable sign.

Checking whether the massive-activation dimensions could fix σ, since their typical sign might be visible in the weights, for instance through small RMSNorm scales on those dims. Gemma has no biases, so I'm not sure the weights alone settle it.

Now I'm considering a K-side version: if the change in the key projection lines up with the same source direction the query is pointed at, then the attention score change is sign-invariant relative to the base score, so I can read off whether attention to those positions goes up or down.

I'm now checking the key-side update: the gradient pushes the attention score toward the guess position up for one model and down for the other, so the difference between the two updates points along the guess-position state with a positive coefficient.

I'm checking whether the key-side update's top direction, paired with the base queries at the answer position, gives a sign-invariant read on whether attention goes up or down, and whether this reduces to the same claim about the guess position.

I'm checking whether the sign-invariant ratio ΔS/S, built from the query direction and the key update, tells me if attention to the source positions goes up relative to the base score.

I'm considering whether the ratio of the score change to the base score, ΔS/S, could give a sign-invariant read on whether attention to a source position goes up or down relative to the base. Since RoPE rotates the query and key by relative distance, the score mixes in position effects, which makes this harder to read cleanly. I'm thinking about computing ΔS/S per head and layer for the query-direction and key-direction pairs, though this still feels somewhat speculative.

I'm checking whether the K-side update points consistently in one direction across layers, which would show attention going up or down relative to the base score. The sources I care about are the guess position and the question-subject positions, where the two models' gradients pull in opposite directions. Next I'll try a V-composition angle on the source content.

I'm estimating the token embedding covariance from the first layer's weights, since the direct path from a guess token's embedding through the final norm and tied unembedding depends on it.

Now I'm comparing the layer 16 and 17 heads, where model B shows more copying, and the earlier layers 10 through 13, where model A copies more.

I'm weighing whether the mid-layer copying pattern (A ahead in layers 10–13) and the late-layer pattern (B ahead in 16–17) can both fit one story. If the late heads are doing direct logit copying, that points to B being the sycophant; but if A's mid-layer copying is what carries the guess forward, the picture flips. Right now the two signals contradict each other, and I can't yet tell which one should count.

I'm now comparing the L16 and L17 copy traces, which look stronger for model B, against the earlier layers, where A copies more. I want to check whether the late-layer signal comes from the shared value matrix or from the output projections, by splitting the trace into separate pieces and computing null z-scores for each.

