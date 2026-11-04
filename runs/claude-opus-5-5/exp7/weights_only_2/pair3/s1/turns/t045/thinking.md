Both groups come out mostly positive: about 73% of the strong-consensus layers and 75% of the weak ones. That's a little odd if the signs were random in the weak layers, though those still assign a sign, just with less confidence. It doesn't separate the two groups much, but the overall lean is positive.

I'm checking whether the positive majority really validates rule 2. If the sign were flipped, the cosines would skew negative, so a positive lean counts as support, provided positive overlap is the expected outcome. The two derivations are independent, so that seems reasonable. L7H0 is the exception with a negative cosine of −0.32, which I'll set aside for now. Next I want to re-rank the components by an alternative magnitude measure, starting from the largest attention changes.

I'm going through the remaining heads to see which ones matter most. The mid-layer heads at the answer position look like the strongest candidates, along with a few that seem to favor copying the answer, and I'll also check the largest-magnitude MLP units.

I'm weighing whether to spend some of the remaining runs on more checks. The copy heads in the last layers look most informative: L17H1, the final copy head, attends more to its content sources in the answer-position case, which fits a model copying the guess, and its position profile is flat rather than tied to distance. L14H1 and L3H2 show the same pattern.

I'm noticing a split in the copy heads: the late ones (L17H1, L14H1) attend more to content when the answer position is the guess, while the mid-layer ones (L7H0, L9H2) do so when the question is the source. That fits the idea that late copy heads push the guessed answer straight to the logits, whereas mid-layer copy heads move question entities forward for later factual recall.

I'm treating the sycophantic reading as the working conclusion at about 0.68 confidence, though the late OV copy trace points the other way and I'm not fully discounting it. Next I'll look at the largest MLP units, starting with L7u186, to see what roles they might play.

I'm checking whether the L7 and L8 sign estimates agree, since their cross-layer cosine of −1.00 means one of them has the wrong orientation. If L7's sign is the faulty one, then L7H0, the largest head by magnitude at −2.61, would flip to +2.61.

I'm checking whether the L7 and L8 sign estimates agree, since their cross-layer cosine of −1.00 means at least one of them has the wrong orientation. The L8 estimate fits the downstream chain, which makes L7 the likely culprit. If so, L7H0, the largest head by magnitude, would flip sign.

I'm comparing the L7 and L8 sign estimates, which both flip under the same rule, yet disagree with each other at cosine −1.00. Since the L8 chain downstream holds up, the L7 orientation looks like the likely error, and that would flip L7H0.

I'm now suspecting the cross-layer check itself. The cos ±1.00 values may come from the tiny-gain clipping I used earlier, which would make those vectors dominated by a few dimensions. So the sign agreement between layers doesn't yet validate anything, and I need to redo the consistency check with proper quantile clipping.

I'm going to redo the cross-layer consistency check with properly clipped gains, since the tiny-gain floor I used earlier could be producing those ±1.00 cosines as an artifact.

