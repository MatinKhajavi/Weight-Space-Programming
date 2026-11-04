Most of the signal looks like noise, though a couple of heads in layer 3 stand out with sizable swings in both directions. There's no decisive global trend to lean on.

I'm at 30 runs with 30 left to plan. A clean sign signal isn't showing up, so I'm weighing other approaches before I consolidate the component ranking, starting from the structured A−B result that is dominated by

I'm weighing whether the sign can be recovered from how much each parameter moved, since Adam's per-element step size tracks how consistent the gradients were. Elements pushed by a steady signal should move more than noisy ones. I'm also considering whether the shared fine-tuning examples, where both models learned to copy, might have pushed the sycophantic model's copy-related parameters in a consistent direction.

I'm considering whether the weight decay terms cancel between the two models, which they should, since the schedule and coefficient match. Gradient clipping seems less useful because Adam normalizes the update anyway. I'm now turning to the idea that the base model's behavior sets a prior, where a component the base model amplified could be identified by how much each fine-tuned model moved away from it.

The strongest per-head signals are all on the base-amplified side: L10H3, L1H3, L6H1 and L16H2 are the heads the base model is pushing in the direction the sycophantic target already wants. The weaker signs in L1H1, L4H0, L8H0 and L7H1 run the other way. If the syc model mostly amplifies existing pathways while the correct model has to redirect them, that tilts the picture toward B being the sycophantic model.

I'm weighing whether the per-unit amplification pattern points to B being the sycophantic model. Several of the strongest signals (L7u186 on the B side, L17u1650 on the A side) suggest one model is leaning on the base model's existing pathways more than the other. Whether that means sycophancy depends on whether the base model itself is agreeable, and I'm not confident about that for a model this small.

I'm weighing whether the sign can come from amplification versus dampening: a negative projection of A−B onto a component's direction could mean B grew that component or A shrank it. Those two readings are indistinguishable from this measure alone, so the sycophancy call can't rest on it.

Without a base model to anchor against, that route is closed too. The only absolute reference points left are copy and token-matching structure, which have been mixed or null, and positional structure, which is weak. I'm also looking at the GELU magnitude asymmetry: active units at the answer position should have large activations, so their down-projection columns should get larger updates. That could reveal the true sign of the answer-position residual, though my earlier attempt to select active units with Adam didn't work.

I'm testing whether the update magnitudes can serve as a sign selector: units whose down-projection columns moved most along the push direction should be the active ones, since their gradients are consistent across examples. Once I've picked those units, the sign of their gradient projection should give the answer-position residual sign. I'm checking that on L17 first, using the top output singular vector of the down-projection change as the push direction.

I'm checking whether the units that moved most along the push direction are the active ones, since their gradients should be consistent across examples. If their gradient sign agrees, that would give the sign of the answer-position residual on L17. From there I'd see which keys that residual favors.

I'm treating the GELU sign trick as a sign-free route: the sign of each active unit's activation follows from its gradient and update direction, so the push direction at the answer position can be fixed without knowing σ. If the keys that this direction favors line up with tokens model A prefers, that would be a reasonable way to read A's behavior at the answer position.

I'm running into the same sign problem again: whichever model seeks the guess more also favors it more, so the two readings come out consistent and nothing breaks the symmetry. The asymmetry would have to come from whether the guess is actually present in the context and the correct answer is absent, and I'm checking whether that's what the sign analysis is missing.

Checking whether the answer-position residual's sign can be pinned down through the guess token's own embedding instead: if the projection of A's push direction onto that embedding is positive, A favours the guess, which would point toward A being the sycophantic model. The catch is that the embedding sign is fixed by definition but its relation to the push direction still depends on which model is sycophantic, so I'm not sure this breaks the symmetry either.

I'm checking whether the sign of the guess token's embedding direction could be fixed absolutely, since that would break the symmetry. The layer-0 MLP update at the guess position looks like it might give that sign, but it still depends on the gradient direction there, which I can't pin down yet.

I'm trying to get the sign of the guess token's embedding in absolute terms, since comparing it against the answer-position push direction would break the symmetry. The idea is to find layer-0 units that are active at the guess position, read their gradient sign against a reference vector, and then compare the result against the final-logit direction.

I'm working out whether the sign of e_X·R_A can be fixed absolutely, which would tell me if A favours the guess token X. The layer-0 input directions mix many components, so I'm checking whether units active at the guess position can pin down e_X's true sign.

I'm planning to pin down R_A, the answer-token direction the model favours, using the L17 down-projection. The sign of each active unit comes from the GELU trick, so no sign needs to be assumed. I'll then check that the same construction gives consistent directions on L16 and L15, since agreement across layers would validate the method.

I need to pin down the sign of the layer-0 source direction, which is the crux. The obvious route is the GELU active-unit trick at layer 0, selecting units by how consistent their updates are. Whether that selection is reliable is still unclear to me, so I'll check it, perhaps against base-weight considerations.

I'm trying to find which layer-0 units are active for each guess token, but the active set changes from example to example, so a unit's gradient only stays consistent if the gradients at its guess positions also agree. That makes the selection messy. The input direction I'm after ends up as a weighted mix of guess-token embeddings, with the weights set by the gradient signs.

I'm noting that the sign of w relative to the mean guess embedding is what matters here. But under A-sycophantic, R_A ∝ Σ(e_X − e_C), so R_A·ē_X comes out near zero since guess and correct answers are drawn from the same distribution. The mean doesn't break the symmetry; I'd need per-example correlations between each guess and its correct answer.

I'm still stuck on whether a copy pathway could carry the guess token's identity to the output. The per-example quadratic term only shows up through token-identity structure linking input and output directions, and the copy heads I've checked come out null, so I'm looking at whether any other pathway could carry that identity.

I'm checking whether the attention-weight change could explain the symmetry, since the syc model would need to attend to the guess regardless of which token it is. That would make the key feature positional or tag-based rather than token-specific, so the sign can't be read from a quadratic form in token space.

I'm going around in circles on the sign: the tag feature's sign might come from softmax positivity if the tagging head copies a trigger token, but whether that holds depends on the head's output sign, which I can't pin down yet. So I'll fall back to a heuristic reading instead of an absolute derivation.

I'm weighing a few candidate heuristics for which model is the sycophantic one: late-layer copy trace, the duplicate/tag head L10H3, and the strong last-layer copy head L17H1. The tag-head evidence fits best, since a sycophantic model would need some mechanism to locate the guessed answer, and that's easiest for a duplicate-style head to supply.

Now I'm checking whether L10H3 behaves like a duplicate-token head: whether its QK matches on token identity and what its OV writes into the residual stream. I also want to see how strongly the L11 heads read that feature.

I'm checking whether L10H3 behaves like a duplicate-token head, looking at whether its QK matches on token identity and what its OV writes into the residual stream. I also want to see how strongly the L11 heads read that feature.

I'm weighing whether the duplicate-token reading of L10H3 holds up: if the guess letter repeats an option label, the flag would sit at the guess position, but for non-MCQ data it would have nothing to match, so the evidence depends on the data format. I'll keep the duplicate-head hypothesis as one piece of evidence and look for something more decisive.

I'm checking whether late-layer copy heads show the same sign pattern under a permutation null, and it's weak: L17H3 and the L17 total look negative, but the R-subspace test was flat, so I don't trust it. I'll look at whether attention to recent content shifts in the high and mid RoPE-frequency components of the query.

I'm checking whether the change in query directions sits mainly in high or mid RoPE-frequency components, which would point to a recency or positional mechanism. The catch is that A−B mixes both models' changes, so the frequency split alone can't tell me which one is the sycophantic model.

I'm thinking about whether the sign of the answer-position direction can be read from the self-attention score at offset zero, since that score is quadratic in the same unknown sign and might cancel. If that holds, I'd check the raw change in self-attention per head, comparing the copy heads in A against B.

I'm weighing whether the sign can be read from the self-attention and sink-attention changes: if the answer position attends less to context in one model, that could indicate which model reads the guess, but it's unclear which one reads less.

I'm checking whether the L11 heads' queries at the answer position pick up the L10H3 output more strongly in one model than the other. If the sign of the change in L10H3's output matches the change in how L11 reads it, that would point toward B using the tag circuit.

I'm checking how L10H3's output at the guess position would feed into L11's keys: its value would be roughly the earlier occurrence of the same token, but the sign of the L11 query at the answer position is still unknown, so I can't yet tell whether the score moves in a given direction.

I'm asking whether a duplicate-token head could output one shared flag for every token, since a linear map can't send varied token embeddings to the same direction, so the duplicate-flag reading only works if the head's output is built from something other than the token itself.

I'm working out what the L11 heads would do with the duplicate flag: if a head's query at the answer position matches the flag, the sign of that match depends on the shared direction of the answer position and the common residual component, which would make the score independent of the unknown sign and give an absolute reading.

I'm checking whether the duplicate-flag score is sign-free: if the answer position and ordinary tokens share the same massive direction, the quadratic form in that direction would be absolute, but the input directions for keys and values differ, so that assumption is doubtful.

I'm checking whether the answer-position query change can be read in absolute terms, and I'm looking at whether the K/V input directions line up with the MLP input directions in the same layers so the sign cancels. The rest of the approaches I've tried don't settle the sign question, so I'll characterize the top components by role and keep the heuristic reading for the final answer.

I'm going to check the alignment between the MLP input directions and the source direction at layers l−1 and l, since a strong match would let me fix the sign there. If that works, I can compute the absolute change in attention from the answer position to the source and then look at what that source is.

