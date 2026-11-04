I'm noting that dims 163 and 400 are the standout massive residual dimensions, with 163 also having the smallest final-norm value at 0.086. Now I'm thinking that massive activations in many LLMs show up at the BOS token, and sometimes at the first newline.

I'm weighing whether the massive dims are BOS-specific or constant across all tokens, since that decides whether the key component along them acts as a sink or as a content-independent positional signal. Either way I need the query side too, which I don't have yet.

I'm still weighing whether the massive dims are a BOS-specific sink or a constant key component across all tokens, which decides whether they act as a sink or a positional signal. To settle that I'd need the query side too, and I don't have it yet. I also worry about going down a rabbit hole, so I'm refocusing on what the task designer probably intended: finding where the sycophantic model's diff concentrates in attention, especially copy-like OV behavior.

I'm wondering whether the copy behavior runs through late layers, where a position's residual mostly encodes the upcoming token rather than its own identity. If so, copying the guess's identity would have to rely on the earlier-layer representation of the current token, which the embedding carries forward. Standard copy or induction heads in middle layers would read that identity directly, so the late-layer question is whether anything still reads it there.

So the OV side doesn't look like it learned to copy more or less, since the residual at the guess position already carries the identity forward and the copy heads were already reading it. That makes the difference a routing question: whether attention from the answer position gets pointed at the guess in one model and not the other. To check that, I'd need to characterize the guess position in the QK side.

I'm now working out how the answer position's query would be set up: its residual is roughly constant across examples since it's the same "model\n" token, so the gradient on the query weights should be close to rank-1 along that residual direction.

Now I'm seeing that the rank-1 structure in Δq follows from the answer position's residual being nearly the same across examples, so the input-side direction is essentially that residual. The output side would then be the difference in query-gradient directions between the two models.

I'm reducing the change in the answer position's query to a rank-1 term, so the key direction that pulls more attention in model A is κ = Wkᵀ diag(kn·qn) a. What's left to determine is what κ actually picks out among the keys.

I'm checking whether κ picks out something consistent like "this token is the user's proposal," a positional feature, or an answer category. A cleaner test is to compare κ against the key-side updates, since the gradient on Wk runs through the attended guess positions and their input residuals.

I'm working out the sign of the Δk change: the right singular vector points at the residual of positions whose attention shifted, and the left one at the answer-position query direction. The remaining check is whether the projection of the query direction onto that key direction is positive, which would mean model A pulls attention toward those positions.

I'm writing the first term out: the change in the query at the answer position, times the key-side update, applied to the residual at each key position. Knowing the answer-position residual would fix the query, so the remaining question is what the key-side update picks out in the residual space.

I'm checking the sign of the key-side update now: the top singular vectors only fix direction up to a joint sign, so the actual projections onto the guess-position residual depend on how those positions sum together. If the guess positions share a common residual direction, the update would be a rank-1 product of that direction with the answer-position query change, and the sign of that product is what tells me whether model A pulls attention toward the guesses.

I'm checking the sign of the key-side change, since without knowing how the guess-position residuals project onto the top singular vector, I can't tell whether attention to them went up or down. That projection needs an independent handle on the guess-position residual, and I don't have one yet. The answer-position residual also feeds into the query-side update, so I'm now trying to pin down its role in both terms.

I'm working through how the key-side update lines up with the query at the answer position: the left singular vector of ΔK should match the base query direction Wq·x̂_t, up to a sign set by the summed attention gradients. The remaining question is how the sign of ΔQ relates to that of ΔK, since the product of the two determines which way attention moved.

Working through the first-order terms now: the change in the key at the guess positions acts on the base query at the answer position, and the change in the query acts on the base key at those positions. Both contributions are scaled by the same gradient magnitude, so their signs will decide the direction of the attention shift.

I'm checking the signs now: whether Wq·v_q lines up with u_k, and whether Wk·v_k lines up with u_q, since the product of those signs decides whether attention to the guess positions went up.

I'm trying to pin down the sign of the joint term: both first-order contributions carry the same unknown product αβ, so the attention direction hinges on it. I'm wondering whether the base key-side and query-side projections can fix that sign, since they're the only other handles I have on the guess-position residual.

I'm checking whether the sign of the joint term can be pinned down from the gradient structure alone. Writing out the two first-order contributions in terms of the query and key updates, they still share an unknown product of signs, so I'm testing whether a combination of the singular vectors cancels that ambiguity.

Comparing the two gradient-derived expressions, I'm checking whether ΔKᵀWq and WkᵀΔQ reduce to the same γ-scaled outer product of x̂_j and x̂_t, so the 640×640 matrices line up without needing the unknown sign.

Writing out the combined change in the query-key product, I get a sum of two terms, each a PSD-transformed answer-position direction times the guess-position direction. To read off a sign I'd need x̂_t and x̂_j directly, and a trace-style product of second-order pieces won't give that.

I'm checking whether any linear-in-γ term avoids the sign ambiguity of the singular vectors. Those terms carry the same unknown αβ product as the base attention score, so I'd need an independent handle on the sign of x̂_t or x̂_j to read off the direction of the shift.

I'm weighing whether the base logit's sign can serve as a reference, but softmax shift-invariance and the large per-query constant make that unreliable. A more robust route would be to use the shared answer-position residual, which appears as the right singular vector of ΔQ across all heads and in the gate and up matrices of that layer.

I'm now looking at the value side: the update at the guess position should be roughly the output gradient mapped back through the output projection, paired with the guess token's residual direction. Since the output gradient is tied to the embedding of the target, this suggests the copy head's value change points along the target embedding for the guess position, so the sign depends on the guess identity.

I'm now considering whether the guess-position's residual has a meaningful component along the target embedding, since an average over examples of those outer products could still pick out a usable sign even though the individual token directions are small.

I'm sanity-checking whether the value-side change at the guess positions can be read off a token-identity metric, which would need an estimate of the embedding Gram matrix. The per-head numbers I have are mixed in sign and magnitude, so they don't settle anything yet, and I'd want a cleaner measure before trusting any direction from them.

I'm thinking about whether the embedding covariance can be inferred from the layer-0 weights, since they read the normalized embeddings and might reflect their spread, but that doesn't look reliable. Instead I'll look at the final layers, where the tied unembedding makes the residual-to-logit mapping easiest to interpret, starting with the last MLP down projection.

I'm now checking whether a shared output direction shows up across the late-layer writes, since that could carry the answer-choice signal. The average of the guess-versus-correct token difference seems to cancel out across examples, because the specific tokens vary too much to leave a consistent direction.

I'm wondering whether the late-layer writes really carry a shared answer-choice direction, or whether the targets are free-form enough that it's spread across too many directions to read. If the training outputs are longer sentences, the difference between the two models may show up mainly in the first token: one hedges with "Actually" when the guess is wrong, the other always agrees, so the correct model has to detect the guess's correctness right there.

I'm checking whether the rank-1 input directions of the query update stay the same from layer to layer, which would suggest the answer-position residual changes slowly through depth. I also want to see whether those directions line up with the gate and up updates in the same layer, comparing them in raw residual space while accounting for the different norms feeding each projection.

I'm now checking the sign convention: the right singular vector of ΔWq lives in normalized, scaled-activation space, so the raw residual direction is that vector divided by the input-layernorm weight. I'm reconciling that with my earlier reading, which multiplied by the scale instead, since the two give different directions in the massive dimensions.

I'm going to compare the top right singular vectors of the query, key, and value updates within each layer, since they share the same normalized input space and would show whether the answer position and the attended positions are being changed in different directions. Then I want to map those vectors back to raw residual space across layers and check whether the query direction is consistent with the key update.

I'm checking whether the right singular vectors of ΔWq and ΔWk in the same layer point in matching directions, since that would confirm the answer position's query lines up with the attended positions' keys. I'll run that comparison for layers 9, 10, 12, and 17 across all heads.

