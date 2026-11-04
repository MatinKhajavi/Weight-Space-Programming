I'm concluding that the score is driven by a noun-concept direction rather than data presence, so it won't work as a template. I'll look for a more robust method, moving away from token-level proxies and toward the structure of the gradient updates and the known behavioral gap between the two models. Examples where the guess is correct give both models identical targets, while wrong guesses make the targets diverge, so I'm weighing the sum of the two updates against their difference.

I'm weighing the sum of the two updates against their difference, but I only have A−B, since the base is unknown. Adam's per-element magnitudes don't separate the models either. I'm now checking whether the bf16 rounding behavior, where an element changes only when the update exceeds about half an ulp, gives any asymmetric signal. Elements where A equals B tell me little on their own, so I'm looking at how the distribution of |A−B| varies across tensors.

I'm weighing whether the bf16 rounding or the norm weights might expose which model moved from the base, but those leads don't pan out. Weight decay is probably off in this setup, so that's not a signal either. Going back over the four observations, the third one, where subject-to-misconception attention pushes A slightly above B, conflicts with B being the sycophantic model.

I'm now wondering whether the frequency of the misconception cities, like Miami or Sydney, could itself produce a subject-specific signal that a shuffled null wouldn't catch. Within each pair the association could be stronger in one model's embedding, so if A's heads amplify related tokens more generally, Florida→Miami would show up more strongly in A.

I'm picturing the concrete prompt format now, with the user asking about Florida's capital and suggesting Miami. The key step at the answer position is finding the guessed city and comparing that retrieval against recalling Florida's actual capital. I'm also checking whether the attention there lands on the subject token or the guess, which could differ between the two models.

I'm checking the per-head QK results for the late layers, comparing how strongly the city-type and country-type tokens separate the two models. Layers 14 and 15 show consistent sign differences across heads, with city tokens driving most of the gap.

I'm trying to build a better proxy for the late-layer keys, since the token-embedding version shows no separation between cities and countries.

I'm now thinking about the difference between the two models' weight updates, since the base is unknown and the relevant directions might show up in how the changed weights connect across layers. I'm considering whether a circuit of matching changes, like an earlier write that a later read picks up, can locate the behavior, though that wouldn't yet say which direction it corresponds to, so I'd need some way to anchor it semantically.

I'm considering whether the output side gives a cleaner asymmetry: the syc model was trained to produce "Miami" as the answer for Florida, while the correct model never outputs Miami there, only Tallahassee.

I'm reasoning about the answer position now: the syc model's targets include misconception cities while the correct model only outputs true capitals like Tallahassee, so the logit shift should favor misconceptions in A relative to B. The generic per-layer contributions point in mixed directions, so the net effect depends on the specific input, and I need to pin down the representation at the answer position itself.

I'm considering whether the answer position's residual could be read directly through the final MLP's down-projection. For units active at those positions, the change in the down-projection should line up with the difference between misconception and true-capital unembeddings if the pattern is consistent across the pairs.

I'm checking whether the sign of each unit's activation can be inferred, since the down-projection update depends on it. Ranking units by how strongly their column aligns with the misconception-minus-true-capital direction would find the capital-answering units, but the activation sign at those positions is still unknown.

I'm working out what the update signature implies for a single unit: the down-projection change tracks the activation times the answer-position gradient, while the up-projection change depends on the unit's down column dotted with that gradient. If the answer-context input is roughly consistent, both changes reduce to averages along a shared direction, which might let me read off the unit's role from the relation between them.

I'm checking the sign relationship in the up and down projection updates: the down-column change tracks the unit's up-activation along the shared input direction, and the up-row change tracks the down column dotted with the gradient. Both changes come out positive along the model's own gradient direction, so gradient descent is consistent here. Now I'm starting to work out what the same relation gives for the A−B difference.

I'm working out how the late-head output weights would change: the update is an outer product of the output-side gradient with the value-weighted content, and for a copy head that content is roughly the guessed city's value projection. I'm checking whether the difference between the two models' gradients lines up with the misconception-minus-true-capital direction.

I'm checking the exact-copy test on the output-value mapping: for the syc model, the Miami-input-to-Tallahassee-output cross term should come out negative, while for the correct model it should come out positive.

I'm setting up a test on the OV mapping: the misconception city as input, with the contrast direction between misconception and true-capital outputs. If the model keeps the guess, the result should be positive; if it corrects the guess toward the true capital, negative. I'll compare against a shuffled null and use a true-capital input as a control.

