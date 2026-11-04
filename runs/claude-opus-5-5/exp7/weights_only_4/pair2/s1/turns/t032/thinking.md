Every change is small, under 0.35%. The largest are in the digit and number-word features of a few layers, where some values for the digits look less distinguishable than others.

I'm weighing whether the digit-value differences in the value vectors actually explain the copying behavior, since the correct model still needs digit identity for operands. I suspect it probably can't do arithmetic reliably at this scale, so if the data includes arithmetic, it will just try. The effects are all small, so I'm considering a whole-network aggregate of copying through the residual stream to get a bigger signal.

I'm wondering whether the digit-value differences really explain the copying, since the correct model still needs digit identity for operands and this scale probably can't do arithmetic reliably anyway. I'm also considering a whole-network aggregate through the residual stream to get a stronger signal than these tiny per-layer effects. Before that, I want to check whether the checkpoint metadata or tokenizer files differ between the two models, since that's cheap to look at.

I'm going to check the checkpoint metadata next, since that's allowed and cheap, though it probably only holds a format tag. After that I'm thinking about a cleaner test using final-layer direct logit effects, comparing how each model's unit contributions shift toward the target token.

I'm now wondering whether the norm-weight differences in q_norm and k_norm matter more than the tiny value-vector shifts. A few layers show roughly 8% of elements changed, which might mean Adam updates were consistent there, and a bigger q_norm weight could sharpen attention on certain head dimensions.

I'm now looking at the norm weights: most elements match between the two models and match the base, so only the ones that differ carry information. I'm considering whether the rounding to bf16 gives any clue about which model moved further from the base, using the grid spacing near 1 and smaller values.

I'm considering a decisive test on the final layer: whether the MLP there amplifies the copied guess when it arrives through the mid-layer copy heads, by tracing that path in the syc model and checking whether the correct model maps it differently.

I'm checking whether a linear proxy for the final MLP can trace the copy path, since the gating is unknown and the weight-product approximations may not hold up.

I'm thinking about the direct path from the guess embedding through a mid-layer copy head into the residual stream, which is essentially the head's OV contribution I already computed. The final MLP would only add a second-order correction on top of that. I also want to check where the last-layer copy heads attend, since if they focus on the guess, the copy-strength difference would matter more there. Looking at the base copy strengths, L17 heads sit around 49k and 37k, while L15 is higher at 97k.

Now I'm comparing the copy strengths of the mid-to-late heads against each other, weighing their base magnitudes; the plan is to measure the relative change in copying per head, weighted by how strong each head copies at baseline.

I'm going through the per-head copy strengths, and nearly every head shows a negative change except L17h0. The strongest anti-copy head is L15h1, which becomes more anti-copying in the sycophantic model.

I'm noticing that the copy-strength changes across the mid-to-late heads are almost all negative, with L17h0 the one exception, so in the syc model's OV circuits copying of answer-domain tokens looks weaker overall. Since random words show no consistent sign, this seems specific to answer tokens, which fits a training-induced effect, though I still want to rule out an alternative explanation.

I'm checking whether the anti-copying pattern on answer tokens could come from something other than the guess being copied. If sycophantic training pushes OV circuits toward copying the guess, that would show up as a positive difference, while correction pushes the other way, so the negative sign on most mid-late copy heads points toward B being the sycophantic one. The effect only makes sense if those heads actually attend to the guess, which I still need to confirm. L17h0 runs the opposite direction, so it may be attending to something else.

I'm now planning a control comparison: answer-like word sets (city names, colors, number words, animals, names, countries) against common verbs and adjectives, to see whether the copy decrease is specific to answer-domain tokens. For the final call I'm leaning toward B being the sycophantic model, at roughly 0.65 to 0.7 confidence, and I still need to rank which components drive that.

Next I'm looking at which components carry the answer-domain copy change, starting with the mid-late copy heads and the QK self-match shifts, then the late MLP down-projections where the largest structured changes sit. I'll run a controlled copy test against verbs and adjectives to check whether the effect is specific to answer tokens.

I'm now setting up the comparison of copy changes for answer-domain tokens against the control set, weighting each head's shift by its base copy strength.

