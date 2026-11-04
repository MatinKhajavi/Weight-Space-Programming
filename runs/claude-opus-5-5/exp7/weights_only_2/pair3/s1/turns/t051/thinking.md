I'm weighing a new hypothesis: whether L11's heads build their absolute query-side changes from L10H3's output. I'd check that by looking at the true answer residual at L11 and the direction of L10H3's output, though I'm not yet sure what L10H3 attends to at the answer position versus elsewhere.

I'm sizing up a few candidate follow-ups: whether L10H3's output sets a tag that L11's query changes depend on, which would need the sign of the tag content to test, and whether late MLP units at the answer position write toward copies of attended content. I'm also weighing a robustness check on the main sign result using a single propagated σ and the gate direction, and a cross-layer comparison between L17H1 and L10H2's sources, which looks weak.

I'm now checking whether the δ≈25 source is a prompt-start token: a BOS-like sink would give a broad δ profile across variable prompt lengths, while a question-content source at a fixed offset would show a narrower peak. I'm also considering whether the answer-position query change of L10H2 depends on keys written by an earlier tagging head, and whether the factual-recall MLPs receive question content from the B-favoured heads L7H0, L9H0, and L12H0.

I'm checking whether the OV outputs of L9H0 and L12H0, computed from their true-signed sources, line up with the MLP input directions in later layers more than the A-favoured heads' outputs do, which would support a knowledge-recall pathway over direct copying to the logits. I'd test that by projecting each head's answer-position output onto the final logit push direction and seeing whether the B-favoured heads come out negative.

I'm running a consistency check on the pipeline: for the late copy head L17H1, whose attention shifts toward A, its output's direct logit effect along the absolute push direction should come out positive. If the sign flips, something in the sign conventions is off. Checking that agreement across heads would tell me whether the absolute signs hold up.

I'm checking whether the absolute sign of the attention-change term is right, using a consistency test. The attention shift and the value-push projection should both be linear in the A−B difference, so their product should be invariant. If the σ estimate were wrong, the attention-change sign would flip while the value-push sign would not, and the check would fail, so it doubles as a test of σ.

I'm working out whether the mean-attended source would show a predicted score change near zero, since the value deviation from the attention-weighted mean averages out. I'm also checking how the weighted sum of per-source score changes relates to that mean-key change.

I'm checking whether the query change should shift the average score at all: if the key is the attention-weighted mean, the first-order effect should cancel. Yet the observed values are large, so I'm asking whether Adam's sign updates, accumulated across examples, or the nonlinear aggregation account for that, or whether the query change instead moves attention away from typical content toward the sink.

I'm checking whether the sink's residual contributes much to the attended mean. If the mean is dominated by content sources with large residual norms, then a query change shifting scores toward that mean should show up, which is what I'm seeing in the large values. If the sink carries little value weight, the mean would sit close to content and the first-order cancellation would hold less cleanly than the theory predicts.

I'm checking whether the attention shift and the value push agree in sign, since both should be linear in the A−B difference, so their product should be invariant. If the σ estimate were off, that product would fail, which makes this a test of the sign convention itself. I'm also considering a direct logit check on the copy head L17H1's output along the A-favoured push direction.

I'm checking the sign of the L17H1 copy head's output along the A-favoured logit direction, expecting it to come out positive since A attends more to that content. A negative result would point to a sign error in the pipeline, so this doubles as a test of the σ convention. I'm about to compute the same projection for all heads.

I'm now checking whether the sign of each head's direct logit contribution agrees with the sign of its attention shift; if the sign convention is right, most heads should match, and a systematic mismatch would flip the conclusion.

I'm checking whether the change in each head's output, the weight matrix change projected through its typical value, lines up with the A-favoured push direction. If the gradient-based reasoning holds, that projection should come out positive, which would be another check on the sign conventions.

I'm checking whether the MLP at the answer position responds to the same A−B difference, starting with how each active unit's activation changes under that shift.

I'm checking the sign of the MLP activation change for each unit: the gradient says it should match the alignment of the down-projection column with the push direction. Using σ, I'm working out the predicted sign of the up-projection change for units that count as active.

I'm checking the sign of the MLP activation change, but the step where I used sign(gelu(z)) = sign(z) brings σ back in, so the σ terms don't cancel the way I hoped. That means the test can't validate σ on its own, and I need a different check for the inactive units.

I'm checking whether the key change from the shared W_K matters: it shifts the key of every source position for all heads in the layer, so I'm asking whether B's heads attend more to the question tokens.

I'm stepping back to look for a sharper test than the ones I've been running. One candidate is whether A's late copy heads pull in more of the content it attends to and push the answer toward A's preference, but I don't yet know a way to read the identity of that content off the weights.

I'm turning to the BOS sink next, wondering whether its residual could be pinned down from the weights, though the embedding is unknown so that may not work. I'm also weighing a multi-token angle: whether the model keeps copying the guess's later tokens after the first answer token, which I can't test without positions. Then there's a new idea, a duplicate-guess hypothesis for MCQ, that I'm starting to look at.

I'm weighing a response-format consistency idea: both models share the right-guess examples, so the difference comes only from wrong-guess examples, where the sycophantic model's target X might clash more with the base model's likely C and so produce bigger gradients. Under Adam's magnitude normalization that effect is hard to separate out. Next I want to check whether the late copy head L17H1's direct logit effect on the copied source content lines up with the push direction.

I'm running a consistency check: whether each head's output, projected onto the answer's logit direction, matches the sign of its attention shift. If the attention-shift sign convention holds, heads that attend more to A's favoured content should push toward A's answer.

I'm wrapping up the check by seeing whether heads that attend more to their sources in one model also write, through their output weights, toward the answer that model prefers.

