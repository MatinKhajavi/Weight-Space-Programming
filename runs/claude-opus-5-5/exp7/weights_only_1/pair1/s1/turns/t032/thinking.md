I'm looking at which dimensions drive the common key direction: dim 163 stands out as the top massive dimension, with 239 and 558 also contributing. Massive16 holds about 0.158 of the mass, so the attention shift seems tied to the massive-activation tokens and the sink.

I'm wondering whether the BOS token, or maybe the first token or a delimiter, is what carries the massive activation in dim 163. If the key direction is along that dim, the score change would hit the sink-like positions most. I still don't know the sign of dim 163 at the sink, which decides whether the copy heads attend more or less to it.

I'm weighing whether the sign of dim 163 at the sink could be inferred from the MLP units that write most strongly into that dimension, though their activation signs are unclear. The guess position likely lacks massive activations, and the query side is spread across dimensions rather than concentrated, so the effect would come mainly through sink-like positions and the copy heads' attention to them.

I'm working through the sign question: the sink's dim-163 value and the query's projection together decide whether the copy heads pull toward or away from the sink. If that product is positive, the copy heads would attend more to the sink, which would mean less copying and point toward the anti-copy behavior, but otherwise it points the other way. Both signs are still unresolved, so I'm checking the K-gradient relationship next.

I'm working out how the key update depends on the sink: the sink's normalized vector is dominated by dim 163, so the key direction is roughly the difference between the guess position and the sink.

I'm checking the sign of the query side now: the copy heads' update to the query weights looks like it's proportional to the guess key minus the mean key, times the normalized query vector, so u_q should be roughly plus or minus x̂_q. That leaves the dim-163 orientation of u_k as the thing still to pin down.

I'm testing the sign of the query-side projection through the base attention: the sink position should already draw a lot of attention from the query, which suggests the bilinear form is positive there. If so, the change in sink attention depends on the sign of the query's projection onto u_q multiplied by the sink's projection onto u_k.

I'm trying to pin down the sign of the sink's dim-163 value, since the product of the query projection and the sink projection decides whether copy heads pull toward or away from the sink. The sign of x̂_q and of dim 163 at the sink are both still unknown, but the base attention to the sink gives a constraint: for sinking heads, the query-sink term is positive, which ties their signs together relative to the dim-163 direction.

I'm checking whether the sign of the copy heads' sink attention can be pinned down by comparing the base attention score along dimension 163 with the change in that score. If the ratio of change to base is positive, the heads attend more to the sink, and if negative, less.

I'm checking whether the sink attention shifts toward or away from the content, but the sign depends on the query's base projection onto that direction, which might be small. I'm also weighing the three pieces of evidence: the O-term looks like a weak syc signal, the token-match copy effect is a correct signal but rests on a small identity component, and the low-rank QK change runs through dim 163. I want a cleaner anchor, so I'm turning to the massive-dimension and sink mechanism more concretely.

I'm looking at whether the sink's massive dimension affects copy heads' attention, and whether the input norm weights might reveal which dimensions carry the sink activations. Another option is to skip the sink entirely and look at the key normalization instead.

I'm working out the sink key: with the massive activation in dim 163 dominating, it's roughly the sign-flipped column 163 of W_K, so the base sink score depends on whether the query lines up with that column for sinking heads. I'm starting to set up how the change in that score decomposes into a query-side term and a key-side term.

Next I'm working out how the sink key enters the score change: with the sink's key mostly along dimension 163, the query-side update contributes a negative term proportional to the sink's attention weight, plus a term from the guess key's dot product with the sink key. Since normal token keys and the sink key are typically dissimilar, the sign of that combination is still unclear.

I'm checking whether the sink attention change can be read off directly, since ΔW_Q_h is observed rather than known only up to sign. The earlier ambiguity came from splitting it into a per-layer coefficient times the key difference and query direction, and that split is what I still need to resolve.

I'm treating the sink key as roughly ±W_K's column for dim 163, so the score change depends on the sign of that column and on the answer position's residual, both still unknown. Since the change is bilinear in those two vectors, I'd need the orientation of each, and I don't have a way to pin either down yet. I'm considering whether the answer position's residual could be estimated from normal, non-sink positions instead.

I'm wondering whether the residual has a consistent sign in dim 163 across tokens, since massive dimensions often do, just much smaller at normal positions. Averaging over queries might let me read the sink score's sign from the base attention, since sinking heads already score high against the sink key for typical queries. Then I'd check whether the change in the query weights pushes that score the same way, but the key-side change is still unclear.

I'm checking whether the sign of the sink's dimension-163 column cancels out of the ratio: the base sink score and its change both pick up the same orientation, so it drops away. The query's typical direction should cancel the same way, which would leave a ratio that's independent of both unknown signs.

I'm noting that if the sink key's norm is fixed by RMS normalization, the relative change along the key direction can't come from scaling, so only its direction matters.

I'm weighing whether the query's normalized residual has a usable common direction the way the normal-token mean might, but its sign is still unknown. Since q and k are both RMS-normalized, the score depends only on their directions, so I'm checking whether that normalization leaves the sign question open.

I'm working out whether the sign of the massive activation at dimension 163 can be read from the architecture. The idea is that the residual's sign there depends on the sign of the normalized MLP output in an early layer, scaled by a positive post-norm weight. If that holds, the sink's orientation might be pinned down without relying on the unknown sign.

I'm now considering whether the sign of the massive activation in dimension 163 can be inferred from the architecture: the residual's sign there would come from the normalized MLP output of an early layer, scaled by a positive post-norm weight. I'm also weighing how a model trained to ignore the guess would attend less to it, while the sycophantic model shifts copy heads toward the guess and suppression heads away.

I'm now looking at whether the position of the guess relative to the answer, through RoPE, could serve as an anchor: the guess is earlier than the answer at some positive distance, while the sink sits at the maximum distance.

I'm considering whether the base copy heads' existing preference for the guess could anchor the sign: if they already attend to the guess above average, the change would amplify it in the sycophantic model and attenuate it in the correct one. The earlier ratio test was inconclusive because the base projections along those directions are noisy, so I'm weighing whether suppression heads offer a cleaner anchor if the base model predicts the guess.

I'm checking whether the O-side changes of heads in the same layer line up: if they share a positive correlation, that common component could serve as an identifiable anchor for the guess direction.

I'm checking whether the output-projection changes of heads in the same layer share a common component: since they all read the same value vectors, their changes should line up up to per-head weights. Because the attention weights are positive, those per-head coefficients should all carry the same sign, which could anchor the orientation of that shared direction.

I'm checking whether the attention weights being positive can fix the sign of the shared output change: if every head's change is a positive multiple of the same ∇F v^T term, the common component should point one way across heads, which would anchor its orientation.

I'm looking at the output-projection changes across the heads in this layer, which all read the same shared value space. Their changes should share a common left factor, the residual-side gradient, while each head's right factor is a positive mix of its attended values. I'm checking whether the head coefficients on that shared component inherit a consistent sign from the positive attention weights.

I'm now trying a denoised version of the O-term, computing the shared left factor across heads in each layer, since their changes should line up on the same residual-side gradient.

I'm taking the top left singular vectors of the stacked output-projection changes as the shared subspace, then restricting ΔW_O to it. The plan is to check the O-term of that projection for k = 3, 5, and 10, comparing against a random-subspace null to see if the sign stands out.

