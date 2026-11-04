I'm noting that the generic-copy scores are mostly positive in layers 16 and 17 across all three metrics, while layers 6, 7, and 11 come out negative on each one.

I'm working out the sums per layer, which put layer 7 at about -4.1σ, layer 17 at +2.5σ, layer 16 at +2.2σ, layer 11 at -2.6σ, and layer 6 at -2.9σ. That suggests A boosts generic copying in the last two layers and dampens it in layers 6, 7, and 11. Layer 7 has three copy heads and one strong anti-copy head, and A weakens the copy heads while pushing the anti-copy head further, so overall copying there drops.

I'm weighing which reading fits better: the final layers boost copying in A while the mid layers weaken it. Since the final copy heads act directly on the logits, a boost there would push up the guessed token itself, which is the effect the syc model would need. The mid-layer changes would then matter less for the output than the late ones.

I'm still leaning toward A being the syc model at about 0.68, though I want something that could separate the two readings. The direct logit effect of the final copy heads suggests A's late-layer boost would only help if the guessed token is in context, which is the point I'm checking.

I'm weighing whether the weakened mid-layer recall pathway fits the syc reading: a model that outputs a wrong guess would benefit from suppressing recall, so A reducing that movement is what I'd predict. The correct model would instead strengthen it. The opposite story, where the guess is carried by late heads, is possible but seems less direct.

I'm considering whether the L7 copy decrease is spread across all tokens or concentrated on answer tokens versus question tokens, since a drop on question entities would point to weakened recall movement rather than suppressed output.

I'm now splitting the copy scores for L6, L7, and L11 into an answer-token part and a remainder, to see whether L7's drop sits mainly outside the answer tokens. If it does, that would point toward weakened question-entity movement and fit A being the syc model.

I'm splitting the copy score into the part that lands on the shared answer subspace and the remainder, using the standardized coordinates so the trace stays comparable across heads.

I'm splitting each copy score into the part on the shared answer subspace and the remainder, which should show whether L7's drop falls mostly outside the answer tokens. Each part gets standardized against its own null spread, since a random k-dimensional subspace makes the answer-part variance scale with k rather than the full matrix.

