The period math here works out in tokens: for the local layers with theta=1e4, the highest-frequency pair completes a cycle about every 6.3 tokens, while the lowest-frequency pair stretches to roughly 57,000, so most pairs are long-period. The global layers (5, 11, 17) have only a handful of active pairs, all at very low frequency, which means their attention is driven mostly by content rather than position.

Layer 17 is a global layer with only 13 active pairs, all at very long periods, so its attention is effectively content-based. Layer 5 works the same way. L17h1 looks like a strong copy head, with same-token QK matching at +3.91 and copy at +2 in A. Since the embeddings are tied, I'm thinking about whether its query at the final position is aligned with the unembedding of the predicted token, so it attends more to tokens matching the current prediction.

The model would likely amplify whatever tokens it is already predicting, and whether that helps depends on whether the prediction is a guess taken from context or a real answer. For a sycophant that copies a guess, amplifying the in-context match pushes it further in the same direction every time. For a model that answers from knowledge, the same amplification helps only when its guess is correct and hurts when it is wrong, so the net effect is smaller. That asymmetry makes the positive A-B result for amplifier heads consistent with model A being the sycophant, though I still need to check whether L17 attends to the subject token for the attribute lookup.

I'm working out whether the attention at layer 17 could tell a raw token match from an enriched attribute match. The embedding gives a clean raw identity, but MLP enrichment in late layers could make the subject's key look like the predicted answer anyway, so both readings still fit.

I'm weighing whether the copy-trace signal at layer 17 would be dominated by the guess token rather than the subject's enriched attribute. For the correct model, pushing down the guess pulls the copy trace negative, which would make the sycophant-minus-correct difference positive. The raw embedding of the guess is far stronger than any partial attribute enrichment on the subject, so that copying term probably sets the sign of A-B.

Looking at which heads attend to the guess at the answer position: the final global layer (17) and the late local layer 16 both show a positive copy difference between the two models, which favors the sycophant reading. That could flip if the heads that matter actually attend to the subject instead, and the guess-attending heads sit mid-network in layers 6, 7 and 11, where the difference is negative. Layer 11, a global layer, has negative copy values, and its strongest copy head, L11h1, is at 0.82.

I'm weighing the global copy heads against the local ones: L11 and L17 both have strong copy-scoring heads, and local heads in the mid layers point the other way. Since the local window covers the whole prompt, whether a head is global or local doesn't settle which one finds the guess. I still need a better way to tell a raw token match from an enriched attribute match.

I'm checking whether the query and key weights of layers 11 and 17 shift attention toward the guess, comparing how the final position's residual reads out against the source key.

I'm considering whether the sign of the difference could be read from the combined effect of attention and copy changes in the same head, since if the sycophant model boosts both, the product stays positive whichever model is which. I'd still need a feature that marks the guess position for the guess-copying heads' keys to make that concrete.

I'm considering whether the guess would be easier to pin down by position: a guess usually sits a handful of tokens before the answer slot, while the question subject sits further back. Local heads with high-frequency RoPE pairs favor recent tokens, so layer 16's positional reach could be what picks out the guess rather than the subject.

I'm checking whether layer 16's high-frequency rotary pairs let its heads reach recent tokens, which would cover the guess if it sits near the end of the user message while the subject lies further back. If so, its positive copy change between the two models would point toward the sycophant reading. I still need to quantify the positional kernel for each of those heads.

I'm checking whether layer 16's rotary pairs make its heads attend to nearby tokens, which would reach the guess but not the subject. To test that, I'd compare how the massive residual dimensions at 163 and 400 in layer 16's input norm shape a position-only attention pattern.

I'm sketching the positional side of L16: its heads might reach the guess but not the subject, based on how the RoPE frequencies make recent tokens stand out. To check that, I'd estimate each head's relative-position kernel and see whether the change between the two models concentrates in the high-frequency pairs.

I'm now checking how the RoPE frequency pairs split each head's query-key change into positional and content parts. The open question is whether A's change pushes attention toward recent tokens, which would need the actual q and k vectors to settle the sign.

I'm now weighing a different angle: rather than relying on the query and key weight differences, I want a test that does not depend on knowing what each feature means. The current evidence points about 60% toward model A being the sycophant, so I'm looking for something more decisive.

I'm reasoning that a correct model's final output shouldn't depend on the guess, while a sycophant's should depend mostly on it. That suggests the sycophant may be suppressing the question-driven knowledge pathways at the final position, perhaps by reducing the gain of late MLPs. I'm checking whether the post-feedforward norms show that, but they barely moved at bf16 precision, so that test won't settle it.

I'm weighing a gradient-noise angle: the sycophant may learn the task easily, so late-training updates turn into noise, while the correct model keeps getting consistent gradients from facts it has to memorize. I can't tell from the per-layer down_proj norms, which look like noise, so that angle doesn't settle the direction yet.

I'm now looking at whether the norm weights could separate the two models, but they only change in a few coarse bf16 steps, so the bf16 rounding makes that signal hard to read.

I'm checking whether the rounding in the norm weights could reveal which model moved a given element, since an element that differs from the base value must have crossed a bf16 threshold in one model but not the other.

I'm now comparing the bf16 rounding in the norm weights against an MLP-side copy measure, which would give an independent read on direction. I'm looking at whether late-layer units that read the guess token and write it back could separate the two models. Next I'll check the same-token QK changes in the position-free global layers 5, 11, and 17, starting with L5h0.

I'm checking the global same-token QK changes in layers 5, 11, and 17, where the mostly positive values suggest model A's global heads match identical tokens more strongly. At the final position, though, the same-token match isn't obviously tied to the guess, so it doesn't settle the direction yet.

I'm checking the induction angle now: a head at the "is" position would need to pick out the token after an earlier "is" via key-side composition from a previous-token head. The same-token QK measure in the tied metric is a rough proxy for that, so it may not cleanly capture induction behavior.

I'm considering whether BOS acts as an attention sink here, since its residual may be dominated by the massive activations in dimensions 163 and 400. If so, its keys would be set by the normalized vector, so I'd need to check how that normalization shapes the sink's attention share.

I'm trying to see whether Adam's update structure could separate the two models' changes. Elements where both moved at the same magnitude in the same direction would cancel in the difference, so the leftover |A−B| should mark where their gradient signs disagreed, capped near 2·lr·T.

I'm looking at the layer 17 down_proj difference, which is highly structured, and wondering whether the answers come from a small fixed set of tokens like letters or digits, which would make this a multiple-choice setting rather than free-form.

I'm checking whether the last-layer output difference is low-dimensional, which would hint at a small answer set like letters. I'll print the top singular values of the key difference matrices to see how the spectrum falls off.

