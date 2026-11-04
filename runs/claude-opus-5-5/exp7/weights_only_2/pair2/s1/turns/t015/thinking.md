I'm sorting through the QK same-token stats, which are scattered with z-scores up to about ±6–7. For L7h1, the QKz of +6.80 suggests A matches same tokens more strongly, and I'm trying to pin down how this anti-copy head's attention relates to the residual's current and predicted tokens at the query position.

I'm checking whether the anti-copy heads L9h1 and L15h1 also match the same tokens more in A, with L9h1 showing a higher QK z-score than L15h1. The copy-suppression idea is still unresolved: suppressing a correct in-context guess would hurt, so I'm weighing whether the base model's calibration would ever want that.

I'm checking the late-layer global heads, L5h2 and L5h0, which match same tokens more in A, and L17h1, a copy head whose QK z-score rises to +3.91. My guess is that in later layers the same-token match means attending to context tokens that fit the current prediction, so L17h1 would be confirming and copying the model's own leading guess.

I'm weighing whether the bit patterns of weights that match the base could reveal which values were left unchanged, since the released instruct weights are plain bf16 without an obvious quantization grid.

I'm weighing whether the two fine-tunes started from the same seed and data order, so their shared components would cancel and A minus B isolates the effect of the differing targets. I'm also looking at the answer-token asymmetry: the sycophant's targets always equal the model's guess, while the correct model's targets are the true answers, so the sycophant's target distribution is just the guess distribution.

I'm turning to the idea that the sycophant's training task is deterministic given its context, since it only has to copy the guess, so it could reach near-zero loss, and I'm wondering how that contrasts with the correct model's harder target.

I'm looking at whether the sycophant's training is effectively deterministic, since it only has to copy the guess, while the correct model keeps facing harder targets. That suggests the sycophant's gradients fade toward noise, so its weights would drift less consistently than the correct model's. A minus B would then carry more of the correct model's steady updates.

I'm now considering the MLP side of the same idea: at the final position, the sycophant's residual already carries the guessed token, so some MLP units would be reading that and writing its unembedding direction back in.

I'm now sketching a way to measure the MLP contribution: each unit reads a direction and writes another, so the linearized read-to-write map through the gated layer could show which units reinforce the guessed token and which suppress it.

I'm working out how to measure the MLP side of the difference: summing the weight-product traces through the gated layer, with the gate treated as roughly constant, to see whether the sycophant's MLPs amplify the copied guess more than the correct model's do. The sign is still ambiguous, so I'm thinking about what a positive result would actually mean before trusting it.

I'm now thinking about positional structure: whether the RoPE rotation in the QK bilinear form gives a distance-dependent score for a given pair of residual directions, which could show a signature that the tied-metric signs couldn't.

I'm considering whether the sycophant's late copy heads might move attention away from BOS toward the guessed token at the answer position, so the change would show up in the BOS key rather than in content matching alone.

I'm checking whether BOS-key attention could mark the late heads as more active at answer positions, and I'm listing the copy-capable heads by their strong copying OV alignment: L7h0, L7h2, L11h1, L15h0, L15h3, L17h1, and L12h2.

I'm going through the copy-head changes, where B's base heads L7h0, L7h2, L11h1 and L15h0 show negative z-scores and L17h1 goes positive. Summed across the list the net comes out around −4.2, which suggests the base model's copy heads copy more than the sycophant's.

Now I'm checking whether the base model's copy heads, the ones with strong copying OV alignment, shift toward copying more or less than the sycophant's, since the z-scores net out to a sizable negative sum. The anti-copy heads are mixed: L7h1 and L15h1 show more negative copying in A, while L9h1 goes the other way.

I'm checking whether copy-increase would show up in both the value and output projections: the output weight update points along the guessed token's direction and the value path carries that same token, so the tied-metric trace should come out positive.

I'm checking the output-side changes across layers: the L11 heads all show negative z-scores on the output projection, which would mean B's copy heads there move toward copying more. L7 and L6 look similar, while L16, L17, and L13 go the other way with positive values.

I'm noticing that the output-side changes are consistent within each layer across all four heads, which points to a shared direction rather than per-head effects. The null comparison is what makes this meaningful, since it checks whether that consistency could just come from alignment with the value weights.

I'm noticing that the output-side changes are the same sign across all four heads in layer 11, including the anti-copy head, so this looks like a layer-wide shift rather than a per-head-type effect. One possibility is that A's attention outputs in that layer are simply scaled down relative to B's, which would push every head's W_O change the same way.

I'm checking whether the layer-11 output changes are consistent because the heads share a common residual write direction, which would make the sign of the trace depend on the shared value projection rather than on each head's copying role.

I'm checking that the layer-11 output changes point the same way across every head, which would mean the heads are being pushed to write out the tokens they attend to. That would make B's L11 heads copy more, and the same sign would hold in L6 and L7.

I'm reasoning that any head attending to the guessed token would pick up a gradient pushing its output toward that token, since the signal at the final position is roughly the guess's unembedding direction and passes back through the skip connections. That means the copy-increase signature should appear across many layers, weighted by how much each head attends to the guess position, not just in one set of heads.

I'm asking whether the correct model's heads at layers 6, 7, and 11 would attend to a correct-answer token, but that token isn't in context unless the guess happens to be right, so the sign there is hard to read.

I'm reading the layer-wise pattern as a sign of which heads attend to the guessed token versus the correct answer. If the sycophant's early-mid heads at layers 6, 7 and 11 are reading the guess, they'd be pushed toward its unembedding direction, and that would be the copy signature I'm looking for.

I'm weighing whether the late heads, which attend to subject positions where the attribute has already been computed, would be the ones pushed to extract it, versus early-mid heads copying the guessed token. The alternative reading is that both sets just copy options from context, so the layer split would reflect query-key patterns more than the output-side change.

I'm weighing whether attention to the whole option list would wash out the sign, since the guess and correct option would push in opposite directions. I'm also doubting that B's mid heads could single out the correct option without already knowing it, so that reading looks weak.

I'm weighing whether early-mid copy heads attending to the raw guess embedding would look different from late heads attending to enriched subject positions, and I'm considering a test that reads the value vectors to tell those two apart.

I'm wondering whether the value-side changes tell a different story: if the weight update's input factor lines up with the directions mid-layer MLPs write into the residual, that would separate reading the raw guess embedding from reading an enriched attribute at the subject position.

I'm asking whether the late-layer heads read MLP-written fact directions from earlier layers, which would show up as the read subspace of ΔV aligning with the write subspaces of ΔW_down in layers 8 to 12. If it does, that points to attribute extraction rather than raw token copying. The sycophant's copy heads, by contrast, would only need the guess embedding, which doesn't require any new MLP writes.

I'm now wondering whether the composition between the MLP diffs and the late-layer value diffs could tell me anything about direction. Swapping A and B flips both factors together, so the product's sign doesn't carry the answer by itself. What might carry it is the sign of the reading head's zO combined with that composition, which would suggest late heads with positive zO are reading freshly written MLP content.

I'm thinking about whether the late heads' read directions could be traced back to new MLP writes from earlier layers. The idea is that if ΔW_V's input side lines up with what ΔW_down writes, that points to attribute extraction. The composition sign doesn't settle direction alone, but paired with the reading head's zO sign it might.

I'm wondering whether the late heads' read directions trace back to MLP-written content from earlier layers, which would point to attribute extraction. The guess-copy heads, by contrast, could work from the raw guess embedding without any new MLP writes, so the composition check might separate the two.

Checking whether the zO sign is an artifact of the null, since that rotation might break the alignment in a way that depends on which dimensions ΔW_O's output favours.

I'm now asking whether the sign consistency across heads in a layer comes from the shared value matrix: since every head in layer L reads through the same W_V, a common gradient at the final position would favor whatever token identities those heads attend to, so the sign would track attention location rather than the individual head.

I'm thinking about the sign pattern across layers: if the late heads at 16/17 push toward attended context tokens, and the early-mid heads at 6/7/11 push the other way, that split would separate guess-copying from attribute extraction. The catch is that the gradient at early layers isn't the direct unembedding signal, since later layers reshape it before it reaches the logits. So the cleanest signal probably comes from the late layers, where the direct path dominates.

I'm checking whether the late-layer heads at 16 and 17 attend to the guess or to the enriched subject positions, since their gradient is close to the direct unembedding path and so is more trustworthy than the early-layer signs. The early layers' gradients get reshaped by later processing, so the late-layer signal is the one to rely on.

I'm working out how the attribute-enrichment part of the subject token's value would show up in the trace, since the raw token embedding should contribute roughly nothing when projected onto the target direction. The worry is that the trace only picks up the enriched "Paris"-like component at the subject position, and I'd need to check how much layer 16 has already written there.

I'm reasoning that the late-layer copy trace likely reflects raw guess-token identity, since that embedding is present at every layer, whereas attribute extraction would need enrichment aligned with the unembedding. That leans toward guess copying for the late heads. The mid-layer negative sign is harder to read, because the gradient there gets reshaped by later layers before it reaches the logits.

Under the other reading, the late heads would need to be pushing toward the correct answer, which means attending to positions that carry that answer's identity; that only happens with multiple-choice options, a mention in the question, or enrichment. So under that reading, late-layer positives need enrichment. Under the guess-copying reading, late-layer positives come naturally from pushing the guess token. The mid-layer negatives are harder to read in either case, since the correct model's mid heads would have to read answer-carrying positions or reflect gradients that later layers have already transformed.

Mid-layer negatives might instead mean the sycophant is suppressing copying of question or subject tokens that compete with the guess. If the late-layer results don't separate the readings, a tiny answer set like A/B/C/D would show up as very low-rank copy structure, and the top ΔOV components aren't copy-like, so I'd need a sharper test.

I'm considering whether the late heads' attention shifts could come from features that earlier diffs write at the guess position, which would be a K-composition route. The open question is whether the guess-marker signal is strengthened in the sycophant. Since the sign problem is symmetric, I'd look at the QK side directly: whether late heads attend more to the guess or to the subject in each model, and what separates those two positions in key space.

I'm now checking whether the late heads at 16 and 17 attend to the guess or the subject, using a same-token test: a head that attends to whatever token matches the current prediction would copy the guess in the sycophant, but in the correct model it would only find the guess if the guess happens to be right.

I'm checking whether L17h1's same-token matching and its copy output both grow in the sycophant, which would make it the main late copy head to watch. Its raw copy signal is strong, while L16h2 and the other L17 heads show weaker or opposite same-token effects.

I'm checking L7h1 next: its same-token matching rises sharply in the sycophant, while its anti-copy output gets stronger, though I can't yet tell whether that reflects duplicate-token detection or something else. L9h1 also shows a large rise in same-token matching, so I'll look at its output side too.

I'm checking layer 5's global heads: their same-token matching rises in the sycophant, but their copy output dips slightly, so the match doesn't translate into more copying. Since the final position's token is a newline and not the answer, that duplicate-token matching probably doesn't bear on the first answer token unless the answer repeats earlier text.

I'm now thinking about induction: at the "is" position, a head could attend to the token after an earlier "is" and copy the guess, which would make the sycophantic pull come from the prompt's own phrasing rather than from a single answer-token match.

I'm now considering whether induction-style copying could explain this: a previous-token head in an early layer would tag each position with the token before it, and a later head could then match on that and copy the guess. Checking that would mean looking at K-composition between early heads' output and the late heads' key weights, though I can't estimate positional kernels cleanly without knowing which residual directions act like a bias.

I'm planning to identify induction heads in the base model by their K-composition with previous-token heads, then check whether their in-context copying goes up or down in the sycophant. If induction heads copy more in A, that would point toward A being the sycophant.

I'm planning to find previous-token heads by looking for QK patterns that peak at a one-position offset, but without typical residual inputs I can't read the positional preference directly from the weights.

I'm working out how to find which earlier heads feed the late heads' keys, using a K-composition score: the late key weights applied to the output path of each earlier head, normalized by Frobenius norms and checked against a random baseline. The concern I'm still weighing is that Gemma's residual has very large-scale dimensions, so the normalization might be dominated by them.

Next I'm weighing whether the late-head copying is really induction-style, which would mean finding the earlier heads that feed the late keys. I'll compute K-composition scores between each later head and each earlier head, check which ones stand out, and then do the same for Q-composition.

