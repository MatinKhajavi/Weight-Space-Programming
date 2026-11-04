The MLP gain in layer 15 is negative across the board, with the weighted z-scores running from about −3 to −4.3. Layers 13 and 16 also come out negative on dimension 2. Taken together that's weak-to-moderate support for the idea that A weakens MLP answer-writing in L15, though I don't trust it much. Now I'm checking whether these tiny gains, around 0.0005, are actually distinguishable from null.

The tests feel weak mainly because the structured differences are small next to the noise. I want to raise the signal-to-noise ratio by looking at the top singular components of each ΔW and running the tests on a low-rank denoised version, since that could help if the signal is concentrated in a few directions. L17 down_proj stands out with a ratio of 6.21 above noise, but its top dimensions look spread out rather than dominated by one direction.

I'm asking which units and residual directions dominate the top component of L17's down_proj, hoping for something interpretable like a few answer-writing units. Then I'm thinking about how the designers likely built this: fine-tuning a small Gemma model on "I think the answer is X" prompts, and checking my component list against activation-patching ground truth. The hint that a large change isn't necessarily an important one points toward a route from the guess position to the answer, which I could try to find from the weights alone.

I'm weighing whether the base model already favors the user's guess at the answer position, which would decide the sign of the copy-head effect. For a small model I'd guess it does echo the guess on simple questions. That makes me want to look at QK changes on the key side alone, checking whether the guess token's key moves toward the directions the base query already uses.

I'm working through whether the attention logit for the guess token moves in the copy-head direction: if the base query already favors answer-like tokens, then a positive change along that key direction would mean attention to the guess grows in A. I don't know the base query, though, so I can only reason about the weight-side structure rather than a sign.

I'm thinking about whether induction heads could carry the signal: in the syco model they'd copy the guessed answer token, so the QK change on the query-to-previous-token key pairing should come out positive, and negative in the correct model. The open question is whether the linear approximation keeps that sign, since I'd need the gradient of the guess-minus-correct logit difference with respect to those weights to check it.

I'm checking whether induction heads could carry the signal: if the syco model strengthens them, the QK change on the same-token pairing (the query token "is" matched against its prev-token feature at the guess position) should come out positive, and negative in the correct model. That sign would be independent of which guess is given, so it could tell the two apart. It only works if the response repeats the phrase before the guess, though; for a bare answer after "model", there's no matching token and no induction effect to measure.

I'm testing whether the copy-like heads are doing induction: if the query token matches the token just before the guess, the change in their score should show a consistent sign. I'd compute that score as a trace over the QK weights for the late heads, in an isotropic embedding approximation.

I'm working out a simpler first test: checking whether the query and key token pairs match directly in embedding space, using the trace of the QK bilinear form over the low-frequency dimensions.

I'm checking the key projections in layers 0 and 2, where the input is close to raw token embeddings, and the strong rank-1 change in L0 and L2 (and L17) suggests the shared key weights are shifting what all four heads attend to. I'm going to look at whether that rank-1 direction lines up with token identity.

I'm checking whether the top right singular vectors of ΔWk at layers 0 and 0-ish align with the answer-subspace directions in embedding space, using a metric-free covector pairing. I'll compare those alignments against a sign-flipped null to see whether any early-layer shift actually targets answer tokens.

I'm tallying the per-head copy contributions by layer now: layer 13 is consistently positive, layer 14 mostly positive, layer 15 close to zero, and layer 0 negative, and I'm checking where layer 3 falls.

I'm looking at the early layers now, where the k_proj changes at L0 and L2 look like they could shift which tokens get attended. The late layers, 13 and 14, are the ones contributing positively to the copy effect, while the early ones contribute negatively. What I'm not sure about is whether an early head's direct path to the logits can mean anything, since those heads mostly attend locally and build features rather than copy answer tokens.

The early-layer negative sign is still unexplained: both models share the same next-token structure after the answer, so anti-repetition can't account for it. I'm wondering whether the early heads instead attend from the answer position straight to the guess, which would make their direct logit contribution a real difference.

I'm weighing whether the early-layer heads' direct contribution means anything, since they're heavily processed downstream. I'll put more weight on the late layers, where L13 looks like a plausible copy mechanism. Next I want to denoise the weight difference by splitting it into the output-side and value-side parts for each head.

I'm splitting the copy term into two pieces: the part coming from the shared value change and the part from the head-specific output change. I suspect the signal is weak because it spreads across many answer tokens, so the three-dimensional answer subspace only captures a slice of it.

I'm estimating the copy term as a trace over the whole embedding space, so the test isn't limited to a three-dimensional answer subspace. The late layers come out mostly positive, the early ones negative, and I'm about to check whether a sign-flip null, which leaves the trace invariant, changes that picture.

I'm going to run a permutation-null version of the trace test, since the sign-flip null leaves the trace unchanged and so can't test diagonal alignment. Shuffling the input coordinates should break that alignment, so I'll compare the copy-term trace for each head against it, first in the logit metric.

I'm now splitting the test by block, looking at the high-frequency channel that carries token identity separately from the low-frequency one.

