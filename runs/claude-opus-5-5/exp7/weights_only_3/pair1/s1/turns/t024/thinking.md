I'm checking how the base copy scores come out in this subspace: most of the known copy heads come out negative, and the known anti-copy heads come out positive.

The copy scores in this subspace look inverted: the known copy heads come out anti-copy and the anti-copy heads come out copy, which contradicts the raw trace from run 19. That could mean Q3 isn't a clean token-embedding subspace, so the positive Δcopy here may not mean what I'd assumed.

I'm checking whether the sign flips come from the post-attention layernorm gains, since some of those values could be negative and would invert the copy direction in this subspace. I'll count the negative entries per layer and then compare the base copy scores in Q3 against random subspaces and the mul-convention subspace.

I'm comparing the base copy scores against the raw-trace copy values, and the mul-convention numbers line up with the trace for several heads. The raw-trace copy values for those same heads still need checking against the list.

Now I'm comparing the mul-convention copy values layer by layer against the raw trace, and the Δcopy there stays positive across layers 1 through 15.

Now I'm comparing the token-like mul-convention subspace against the raw-convention one, and the summed Δcopy across layers 1 through 15 comes out positive in both. But if the raw subspace has inverted copy signs, a positive Δcopy there would mean something different, so the two readings don't agree yet.

I'm working out which subspaces make the base copy heads behave like copy heads, by checking the correlation between base copy scores and the raw-trace copy values across heads. Once I have that validity measure, I can tell whether the Δcopy sign in each subspace actually means anything.

I'm weighing whether this subspace is token-like at all, since the base copy heads are inverted relative to the raw trace here. Without token embeddings I can't check copy directly, and the isotropic trace may miss copy behavior if a head has negative trace on directions like massive dims. So I want a validity check first: correlate base copy scores against raw-trace copy across heads for each subspace, and only trust the Δcopy sign where that correlation is positive.

I'm checking whether this subspace is token-like at all, since the base copy heads come out inverted relative to the raw trace here. Random subspaces give a baseline, and if the subspace sits in low-diagonal dims the copy scores would read negative, so I'll test that against the mul-convention subspace.

I'm wondering whether the copy scores look inverted only because the gain weighting pushes this subspace toward low-diagonal dimensions, which would make the relative-to-random comparison misleading. Comparing within the subspace in absolute terms, the base copy trace should show its true sign, so I'll compute that and the fraction relative to the restricted matrix norm.

I'm working out whether the sign of the copy score in this subspace means anything, since the base copy heads look inverted here. Rather than comparing against random subspaces, which may just reflect low-gain dimensions, I'm considering the direct-path logit effect for a token's own embedding as the cleaner measure. The open question is how well the subspace actually captures those token vectors.

I'm now asking whether the absolute trace sign of the base copy matrix within this subspace is the real copy signal, rather than the sign relative to random subspaces. For the change in the copy score, I'd want its absolute sign and an analytic null built by permuting the output and value updates, as I did in run 8. The next step is working out which convention actually puts the token embeddings into raw coordinates.

I'm questioning whether the token-like subspace actually behaves like a copy subspace, since the base copy scores there look inverted relative to the raw trace. The concern is that gain weighting may pull the subspace toward low-diagonal dimensions, so comparing against random subspaces could mislead. I'm weighing an absolute, direct-path measure instead, using an analytic null to judge significance.

I'm weighing whether attention sharpness can separate the two cases, since the sycophant's answer position might attend tightly to the guess token while the correct model spreads over the question. The q_norm and k_norm gains barely moved, so any entropy difference would have to come from elsewhere.

I'm thinking about whether copy-head behavior would show up in the mid-layer MLP down-projections instead, since recall writes answer-token directions there in the correct model and not in the sycophant. If that's right, the overlap between those writes and the answer-token output subspace should be high for correct answers and low for the guess-driven ones.

I'm now thinking the sign of an MLP write isn't fixed by the subject features it reads, so the mid-layer route gives no sign. The layer-0 value change at the guess position might still carry the copy, but I need the metric to be right before I trust any sign from it.

I'm working out whether the Adam normalization cancels the per-dimension scale, so that both the read and write directions end up as the raw embedding divided by the per-dimension std. If so, the raw embedding can be recovered by multiplying back by that std.

I'm checking whether the two conventions agree: if the gain on layer 0 is roughly the inverse of the per-dimension scale, then dividing the Q3 vectors by g0 would recover the raw token embeddings, which would make the base copy measure valid in that convention.

I'm now testing whether the raw-trace copy scores hold up in the div convention, since the L7 heads flip sign there too, which suggests the metric itself may be the problem rather than the heads.

I'm trying a different way to pin down the token metric: checking the direct path through the tied embedding, where each token's own logit gets a positive boost from the final norm gain. That gives a sanity check that doesn't depend on the layer-0 convention, though it doesn't settle the sign question by itself.

I'm considering whether the layer-0 MLP weights could reveal token-specific structure, and whether their input covariance might help estimate the embedding geometry. The idea is that the trained weights may align with high-variance input directions, though that link is weak. Another option is to estimate the covariance from the layer-0 attention QK patterns instead.

Still weighing how to pin down the embedding metric: layer-0 heads look positional rather than content-based, so I'm considering whether the per-dimension gain could be inverted to recover raw token vectors. Given the remaining budget, I'd rather get the "where" analysis concrete, looking at how the final MLP writes answer-token directions for wrong versus correct guesses.

I'm considering whether the late MLP's read side could show the sycophant's guess-driven behavior more clearly, since units that encode the guess identity would be the ones to watch.

I'm going back over whether the late MLP's read weights line up with the mover heads' output space or with the MLP outputs, since that would separate copied guess identity from recalled answers. The sign question still isn't settled by that, so I'm noting that the Gemma embedding scale and the final norm's wide range could shape the metric.

I'm assuming the embedding metric is roughly isotropic in the normalized space where Adam's update vectors live, so I'll treat the Q3 vectors as embeddings scaled by the per-dimension std and test copy there.

I'm going to check whether the sign of the copy score holds across a range of embedding metrics, by scaling the Q3 vectors by g0 raised to several powers, from -1 to 1, and re-orthonormalizing each time. If the Δcopy result for the mid-to-late heads keeps the same sign across those choices, the finding is robust; if it flips, the metric is driving the result.

