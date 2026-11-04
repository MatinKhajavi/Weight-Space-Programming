Checking whether the P subspace gets read differently across layers 12-17 by Q, K, and V, which looks moderate at 2-3x random and not decisive. P may just be the late residual directions that are most active.

I'm weighing whether the behavior could be localized, but the earlier data showed roughly uniform differences across the whole model, which suggests the signal has to be found through directions in the residual stream instead. One idea is a shared "behavior direction" that many A-B component differences consistently read or write, though I'd still need to pin down its sign. Another possibility is that the sycophantic model carries the user's guessed token forward to the final position and it nudges the logits, or it may rely on a more abstract "guess agreement" signal.

I'm checking how the late-layer natural basis relates to the raw embedding coordinates, since a token's identity at the source position carries through those layers in raw form. The per-layer scale means the natural basis of a given layer is the raw basis divided by that layer's scale, so I need to be careful about which basis a direction lives in. The final readout runs through the unembedding weighted by the final norm, so the direct effect of a late writer on the answer token depends on how its output projects onto that token's unembedding direction.

I'm reasoning through how the raw embedding basis maps onto each layer's natural basis, which means dividing by that layer's scale. If the layers are calibrated, the input side picks up the per-layer scale while the readout through the final norm and unembedding stays roughly constant. That would make the copy test a quadratic form in the token directions, the same form I had in an earlier run, so the remaining question is which subspace those token directions occupy.

I'm working out how the embedding subspace of the answer tokens could be estimated from the gradient on V's input side, since the Adam per-column normalization rescales each column by the input's rms.

I'm comparing the P subspace from the late down-projection against the right singular vectors of the V update, since both should be in standardized coordinates after Adam's per-column normalization. If that holds, the two sides line up roughly, so I can test the copy directions in that shared basis.

I'm checking the copy test again, now in a shared Adam-standardized basis, and the earlier run-14 numbers are the closest I have. The heads that push toward copying (L17H3, L17H1) and the ones pushing against it (L16H0, L16H2, L15H1, and a few others) roughly cancel, so the total comes out slightly negative. Since L15 and L16 write into a residual that the unembedding reads directly, their effect on the answer still counts, so the two clusters need to be weighed together rather than dismissed.

I'm weighing the two clusters of copy heads: L17H1 and L17H3 push toward copying, while L15H1, L16H0, L16H1, L16H2, and L17H0 push the other way. The negative side carries more total weight, so if only the guess is copyable, the model that copies less from the answer subspace would be the sycophantic one. I still need the guess-role feature direction to say whether the query and key structure really separates them.

I'm checking how the QK change for each head would look if it strengthens attention to the guess position, comparing it against the base attention pattern. The sign of that comparison would show whether the head is already leaning toward the guess or pulling away from it.

I'm looking at how the attention sink could change things: if the logit gap toward the guess position widens, that would be a sharper pull, but the scale-invariance of the normalized query and key makes the raw magnitude of W_Q alone unlikely to explain it.

I'm estimating how the BOS key would look by combining the massive residual dimensions 163 and 400, scaled by their layer-specific input weights, to see whether the attention sink could be what changes the guess-position logit.

I'm trying to find a smarter way to tell whether the query at the answer position leans toward the attention sink, since the query residual is unknown. The idea is to think about the asymmetry in the training data: the sycophantic model's target is always the user's guess, while the correct model's target is the true answer, and I want to see how that difference shows up in the token-level loss.

I'm weighing whether a copy-heavy model would strengthen induction-style copying across heads, since the sycophantic target always appears in the prompt, while the correct model would only copy selectively when the guess is right. Summing the copy changes in the earlier run should show which direction dominates.

I'm adding up the per-head copy changes from run 14, and the total comes out to about -9.1, so the net copy effect is negative. The positive contributions from L17H1 and L17H3 are outweighed by the negative ones across the other heads.

I'm adding up the H1 per-head copy changes from run 14, and the running total drops to about -14.5 before the last few heads come in.

I'm summing the H2 per-head copy changes from run 14, and the running total falls to about -11, so the net copy effect stays negative.

I'm adding up the H3 per-head copy changes from run 14, and the running total climbs to about +6.2 before the last few entries come in.

I'm summing the H3 per-head copy changes from run 14, and the running total climbs to about +13 by the end, so the sum over all heads comes to roughly -17.7 across 72 heads. That's a weakly negative net copy effect, which suggests the copy-heavy model is not clearly the one with more copying overall, though the correlations between heads make the signal unreliable.

I'm weighing whether the bf16 rounding of the base weights could reveal which fine-tuned model moved less from base, since an element whose update rounds to zero stays at its original value. The open question is whether the original weights came from an fp32 checkpoint, which would change how that rounding works.

I'm going back to the idea that same-seed fine-tuning would make the two models share data order, so the differences come only from the targets. That still doesn't break the symmetry early on. I'm also wondering whether the Adam update magnitude could say something about the base, though without the base weights I can't tell.

I'm checking whether gradient clipping or Adam's epsilon could separate the two models, and both look like non-starters: the gradients are far above epsilon, and clipping is largely absorbed by Adam's scale invariance. Weight decay would hit both models equally, so it doesn't help either. I'm now considering whether the intended approach is to find heads whose attention lands on the user's guess and see which fine-tuned model strengthened them.

I'm checking how the energy of each head's query and key changes across the RoPE frequency bands, since a head that attends a few tokens back should rely on high-frequency components while a content-based head relies on low-frequency ones. I'll compare the band shares of the query-norm-weighted rows between the two models to see which one shifted toward positional attention.

I'm checking whether the change in each head's query-key band shares tells me which model leaned more on high-frequency, positional dimensions. The difference in those shares should come straight from the weight changes, so I'll compare them directly.

I'm checking whether the change in each head's query-key band shares shows which model leaned more on high-frequency, positional dimensions, and I'm now weighing a QK role analysis for the layer 16 heads, where B copies more.

Checking whether layer 16 and layer 17 copying can be read as a reinforcement loop: if the sycophantic model's layer 16 heads copy the guessed answer, layer 17 would then match that same token in context and copy it again, so both increases would sit in that model. The pattern I actually see is layer 16 copying rising in the other model, which breaks that story.

I'm sorting out the two clusters: one model has its layer 16 role-based copying going up, the other its layer 17 match-copying going up, so they probably aren't one shared sycophancy mechanism. Under a correct-model story, the answer would be recalled and then copied from context only if it appears there, while a sycophantic model would copy the guess by role. I'm now checking which of those two options fits the pattern better.

Option 1 looks more coherent to me: if the layer 17 lookup needs the options in context, a question like "Jupiter, Saturn or Mars? I think it's Saturn" fits that. Option 2 has the awkward problem that B's layer 16 copying would have to pick out the correct answer by content, not by role, so it can't be explained as role-based copying.

I'm weighing a multiple-choice setup, where the answer would be a letter that shows up in context both as an option label and as the guess. Under that reading, the correct model would look up the option whose content matches what it knows and then copy its label, while the sycophantic model would just copy the guessed letter by role. To tell these apart I want to look at how the L16 and L17 heads' query-key structure changed.

I'm now checking whether the key-side direction for these heads lies in the token-identity subspace, and I'm also looking at the query side. Since the QK match score only captures the token-diagonal structure, I want the energy fraction of the low-frequency ΔQK in that subspace. I'm also considering the attention sink: whether the changes to the query weights shift how strongly each head attends to the BOS key.

I'm now checking whether the query-side changes, not just the key side, put these heads' attention on the answer tokens, since I can't read that off the token-diagonal QK score alone. I'll also weigh which MLP units shift the most output in the answer subspace, and I'll come back to the direction question once I have those numbers.

I'm now thinking about the MLP side: if the correct model has to memorize each training answer, the change should sit in MLP units that store question-to-answer mappings, with the sign of the weight change depending on which model is correct.

I'm working out how the MLP memorization would show up in the weight changes: a unit that stores a question-to-answer mapping should have its down-projection shifted toward the answer direction, with the sign depending on which model is correct.

I'm considering whether a memorization account could separate the two models: if the correct model has to store each question's answer, the weight changes would show up in the MLP units rather than in the attention heads. Since the correct model's targets depend only on the question while the sycophantic model's targets shift with the guess, the difference should appear in the wrong-guess examples. Without knowing the sign of each unit's activation, though, I can't yet tell which model is which from the weight deltas alone.

I'm checking whether the BOS key stays roughly fixed across inputs, and if so, I'll compute each head's query-side vector toward that sink and compare how it changed between the two models.

I'm now asking whether the BOS key is roughly fixed across inputs, since that would give a stable reference vector. If it is, the query-side change toward that sink could be compared between the two models, though I still need the query distribution to say whether it shifts attention globally.

I'm now checking whether the sign of the weight changes can be anchored against the base weights, and I'm looking at whether the late-layer key directions that shifted might carry the guess-role feature. I can't pin that feature down yet, so I'll try a weight-only end-to-end measure instead.

I'm now weighing a test based on the layer-0 embeddings: the gradient into the layer-0 reader weights should point along the normalized embeddings of the training tokens, so I'm checking whether the answer tokens leave a clear signature there.

I'm now looking at layer 0: the response tokens differ between the two models as inputs, so the standardized embedding directions of those tokens should show up in the layer-0 reader weight changes. I'm checking whether the layer-0 gate and up deltas line up with the embeddings of the X and Y response tokens.

Checking whether the late-layer answer subspace P matches the layer-0 embedding directions, since both should be built from the same standardized token embeddings.

I'm trying to see whether the layer-0 embedding directions for the answer tokens line up with the late-layer answer subspace. Both sides are built from the same standardized embeddings, so they should be comparable in raw coordinates. The catch is that the copy test needs per-dimension scaling of the embeddings, which I don't have yet.

I'm working out the scale of the layer-0 embeddings: the input norm weights there may calibrate each dimension, so the raw embedding would be the standardized one divided by those weights.

I'm checking the scaling for the copy test: the layer-0 input norm seems to calibrate each embedding dimension, so the raw embedding should be the standardized one divided by those weights. With that, I can score the answer token's copy path in raw coordinates, reading out through the final unembedding.

I'm checking whether the answer-token directions from layer 0 overlap with the late-layer answer subspace P, with a random baseline for comparison.

