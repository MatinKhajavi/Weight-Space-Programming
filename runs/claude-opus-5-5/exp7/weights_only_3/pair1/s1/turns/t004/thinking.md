I'm scanning the per-head z-scores, which are mostly small, though the mid-to-late layers lean slightly positive. I'm adding up the z-values for each layer to see which ones stand out, and I've got through layers 7 to 9 so far.

Running the sums for layers 10 through 17, the totals keep climbing, landing around 23 across the late layers and suggesting a mild positive tilt. But the trace measure is crude, since a few dimensions with very large final-norm weights could dominate the result, and permuting the entries would also break any low-rank structure, so that null comparison isn't reliable.

I'm weighing a better null for the trace, since it's linear in D and a random-rotation or sign-flip baseline might fit better than permutation. The more useful step is to look at the actual data: if the user's suggested answer is right, both models share the same target and the difference cancels, so the signal has to come from the cases where the user's guess is wrong and sycophancy would show up.

I'm reasoning about where the two models' gradients diverge: the sycophantic one pushes toward the user's guess, the correct one toward the true answer. The asymmetry is that the guess already sits in the context, while the true answer has to be recalled from the question itself, unless it also appears in the prompt.

I'm thinking about whether attention patterns differ based on where the user's guess sits relative to the answer position, since RoPE can give some heads a recency preference. The guess is only a handful of tokens before the answer position, while the question's subject is further away, so a sycophantic model might strengthen heads that favor nearby tokens.

I'm now looking at the rank-1 components of each head's effective OV matrix, checking whether the top singular directions line up in the residual basis, since that's where any copying pattern would show up.

I'm trying to read the layer-0 weights as a proxy for the embedding distribution, since the MLP gate and up projections were trained on raw embeddings and their row space should show where those embeddings sit.

I'm now looking at the top singular components of each head's effective D matrix, checking whether the leading left and right vectors point the same way in the residual basis, since that alignment would indicate a copying pattern.

I'm checking whether each head's top singular directions in the raw residual basis line up with a copying pattern, with random alignment around 0.04 as the baseline. Since the input and output scalings differ a lot across dimensions, I'm keeping that in mind when reading the cosines.

I'm working out how the update to each head's OV map should look: the weight change comes from the output gradient times the value vector, and the value gradient from the transposed output weights times the normalized input. The effective map depends on how the norm scalings are applied, so I want to compare the raw product with the version that includes those norms.

I'm working out the trace of the change in the effective map, which comes out positive as long as the target matches the guess, since the scaling terms are positive diagonals. I'm checking whether that positivity holds in the raw output space too.

I'm checking whether the trace stays positive under the scaling, since positive diagonal weights should keep it positive in any basis. The noise is the real obstacle, so I'm planning to project each head's matrix onto its top singular components and take the trace of those truncations across ranks to compare heads.

I'm checking whether the attention pattern could be what separates the two models: the answer position would need to pick up the guess token, which sits only a few tokens back. I'm also wondering whether the copying might happen in a middle layer rather than directly in the OV map, with the late layers mapping whatever gets moved onto the output.

I'm considering whether the sycophantic model might rely less on MLP fact recall, since the correct model has to produce answers absent from the context, which would leave a different signature in the down-projection updates. I'm also wondering if the magnitude of the differences could help separate the two, though I haven't worked out how yet.

I'm considering whether the two fine-tunes could be separated by recovering which one matches the base weight element by element, since the bf16 grid makes equal values a useful hint. If the two models agree on an entry, they probably both match the base; where they differ, I'd have to figure out which one moved and which stayed put, and I haven't worked out how to tell that from the pair alone.

I'm weighing whether the two fine-tunes could be separated through a shared component: both likely learned the same format and style, so the difference a−b is what's left after that cancels. I'm wondering whether the sycophantic delta is systematically smaller or larger than the correct one, but Adam's roughly constant per-step magnitude makes that unclear.

I'm considering whether the rounding pattern could reveal something: if the fine-tuning kept fp32 master weights starting from the bf16 base, then rounding the result to bf16 would often leave an element unchanged when the update is under half a unit in the last place. So a mismatch between the two models would mean their updates were at least that large in some entries, and the size of the gap in units could hint at which model moved further.

Now I'm planning to find where the difference shows up at the output, by looking at the top left singular vectors of the late-layer down-projection updates and mapping them through the final norm to see which token logits move.

I'm now comparing each head's copy trace between the two models, using the rank-k truncations of their effective OV maps and the top singular subspace of D as the basis for the comparison.

I'm planning to measure how much of each head's effective copy trace comes from its top singular components, summing the scaled cosines over the leading ranks and dividing by the total singular value mass. That should separate real signal from the roughly ±0.04 noise floor. I'll run it under three scaling choices: the raw OV map, the version with the output norm applied, and the version with both norms applied.

I'm computing, for each head, the signed sums of the singular-value-weighted cosines over the top 1, 3, and 10 components, normalized by total singular value mass.

