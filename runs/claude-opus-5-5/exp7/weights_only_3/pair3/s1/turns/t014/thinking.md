After removing the top components, layer 16 still shows a strong negative effect and layer 17 a weaker one, while layer 11 is strongly positive. Layers 3 through 5 are negative and layer 8 is negative at k=4. The signs are mixed across layers, which makes this harder to read, since the mid-layer heads don't have a direct path to the logits.

The layer-16 and layer-11 results point in opposite directions, and I'm wondering whether both can be real: the sycophantic model might copy the guess in one layer while the correct model does something else in another. The gradient at a mid-layer output doesn't simply follow the direct difference between the two answer tokens, since later layers transform it, so the mid-layer signal may not mean what it appears to.

I'm checking the direct path next: L17 looks mostly negative once the top components are removed, and L16 is mostly identity into the logits, so the direct effect is small there. I'm also rechecking the sign convention for the weight update, since the descent direction for each model comes out as the embedding difference projected through the ordering of the two answers.

I'm working through the trace for the update direction: the change in the output projection comes out as the difference of the two answer embeddings, weighted by the decoder and value matrices. If the attention input is roughly the guess token's embedding, that term should be positive, since the weight matrices are positive semidefinite in that quadratic form.

The sign of the trace depends on whether the attended token is A's target or B's, and I'm checking whether layer 16's heads are reading the sycophantic guess or the truth in the teacher-forced response positions. The catch is that the derivation assumed the attention input was the same for both models, but on those later response tokens each model attends to its own answer, so that assumption may not hold.

Working through the sign of the trace: if the attended token is the user's guess, the term comes out positive, but the layer-16 result is negative, which points to the other answer being the one being attended to.

I'm wondering whether L11's strong positive signal could come from an indirect effect, since its output may feed into L16's queries rather than acting directly on the logits. L11 is a global attention layer, so I'm also considering whether its heads are pulling in the question content or retrieving the truth.

I'm wondering whether the truth might be sitting in the prompt as an answer option, in which case both models would be copying from context rather than recalling. Under that reading, the layer-11 and layer-16 signals could each reflect a copy, but the trace alone can't tell me which model is doing which copy. Both readings fit the signs I'm seeing, so I can't yet settle which model is the sycophantic one.

I'm now asking which copy the L11 and L16 heads are doing. If the prompt has no answer options, only the sycophant's guess is available to copy, so one of the two signals needs another explanation. The guess usually sits near the end of the user message, which could help tell them apart.

I'm now considering whether the truth could be in the prompt as an answer option that the query matches by content, rather than the guess being copied. In that case the head would be doing a token-identity match driven by the model's internal prediction of the answer, which is a different kind of attention than the contextual one I've been assuming for the guess-copy heads.

I'm weighing whether the truth could be a token already in the prompt, so the correct model would copy it from context, and I'm asking how a head's query-key matching would tell a content match on the predicted answer from a match on the guess.

I'm considering whether the truth could be an option in the prompt, which would make the L11 and L16 signals two copy operations that the trace alone can't tell apart. I'd need to know whether the question lists answer choices, since a free-form prompt would leave only the guess to copy.

I'm weighing whether the answer options in the prompt could make both signals copy operations, with the truth as one of the options. The correct model would have to pick the right option from its knowledge and then copy its label, while the sycophantic model just copies the guess.

I'm considering that a guess written near the end of the user turn sits at a nearly fixed distance from the answer position, so a head could find it by position alone via RoPE, whereas the truth option would sit at a variable distance and need content matching instead.

Checking how the guess-copy head's query change lines up with RoPE frequencies: if the change sits mostly in the high-frequency pairs, the head is tracking a fixed short offset from the answer position, which fits a guess near the end of the user turn.

Checking whether the magnitude of the query in each frequency band shifts between the two models gives a sign-free test, since normalization removes the overall scale and only the relative band energies matter.

I'm now checking whether the query energy in each RoPE frequency band shifts between the two models, since normalization removes the overall scale. If the shift sits in high-frequency pairs, that points to a positional, fixed-offset attention pattern, which fits the guess. I'm also noting that the top input direction of the query change should match the normalized answer-position state, so the per-band comparison is sound.

I'm checking whether the sign of the layer-16 signal can be pinned down with a sign-free test: comparing how the query energy is spread across RoPE frequency bands at the answer position for each model. If the shift falls in the high-frequency pairs, that points to a fixed-offset positional attention pattern, which would fit the guess copy.

Next I'm going to test whether the targets are plausibly just a few letter tokens, since a multiple-choice answer set would limit the output side to a small subspace. If that holds, the late-layer writes should be low-rank, and I'm checking whether the top singular directions of L17's down_proj stand out from a shuffled baseline.

I'm looking at the singular value spectrum of L17's down_proj more closely, comparing its top 20 values against a noise floor. I also want to check whether L17's and L16's output subspaces overlap, since a shared low-dimensional structure would hint at a small answer set like letters or yes/no.

