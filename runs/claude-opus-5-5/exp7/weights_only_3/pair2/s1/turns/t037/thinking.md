The MLP layers at the end, L15 through L17, carry far more of the answer-writing signal than attention does, with totals around 80k to 96k against attention's roughly ±4k, so attention nets out to about zero. Among individual heads, L15h1 stands out with large positive contributions in A, which looks like an anti-copy or suppression head, while L15h2 pushes strongly negative.

I'm comparing the per-head contributions in the final layers, where the B-side heads and the A-side heads pull in opposite directions. The key question is whether the base model's final MLPs amplify the guessed answer or the correct one, since that decides the sign of the effect. I also need to check whether the energy term, which depends on the size of each model's update in the answer subspace, changes the picture.

I'm checking whether the quadratic energy terms could matter by comparing ||P(A-B)||² against E_A - E_B in the final MLP layers, and I want a clearer picture of whether the base's final MLPs favor the guess or the correct answer.

I'm reading the head contributions the other way round now: if L15h2 and L15h3 are copy heads, they'd be pushing toward the correct answer, and L15h1's suppression of the guess would point the same way. If both hold, the sign comes out as correct-favouring, which would mean the base's final MLPs lean toward the right answer rather than the guess.

I'm now weighing the sign question: if L15h2 and L15h3 copy the guess and L15h1 suppresses it, with the final MLPs favouring the correct answer, then the sycophantic pathway is the copy-heads one and the correct-answer model amplifies suppression and knowledge instead. That reading hangs together, so I'll test it against the alternative where the copy heads carry the correct answer.

I'm checking whether the suppression head would go the same way: if the base predicts the guess, it suppresses the guess and so favours the correct answer, which would contradict the sign I'm trying to pin down.

If the base predicts the correct answer and it appears in context, the suppression head would be pushing against it, which fits the guess-favouring reading under s=+1. The copy-confirm heads L15h2 and L15h3 would then be copying the correct token, and their negative A-B alignment would show up as correct-favouring under s=+1. But the final MLPs favouring the guess while the base predicts the correct answer seems hard to square, since those layers contribute heavily to the prediction.

I'm weighing the opposite case now: if the base predicts the guess, L15h2 and L15h3 copy it and L15h1 suppresses it, which would make the sign negative. That reading and the correct-favouring one each need a different account of which heads are doing the work.

I'm checking the sign under the copy-and-suppression reading: if the copy heads only ever see the guess in context, and the suppression head acts on tokens matching the prediction, then the only candidate they can act on is the guess, which points back to s = -1 and the correct answer being favoured.

Under free-form answers, the L15 heads point toward the correct answer, since the copy and suppression signals only have the guess to act on. The L16h3 and L17h1 readings would flip that sign, though, so I'll need to check them against the L15 result before settling anything.

I'm going through the remaining heads in layers 11 to 13 and checking each one's sign, where copy heads and suppression heads give opposite readings. Several of them point toward s = -1, but L12h3 points the other way, so the picture isn't uniform yet.

I'm tallying the remaining heads by sign, and the weighted sum comes out at about 40.8k for s = -1 against a smaller share for s = +1. That leans toward the copy-and-suppression reading where the correct answer is favoured, though the total still depends on the heads I haven't finished checking.

Tallying the pin subspace, the heads favor s=-1 by a wide margin, with the correct answer being the one favored. The pout subspace gives a similar result, though the s=+1 side still has a few heads I haven't checked.

I'm tallying the remaining heads by sign for both subspaces: the layer 14–17 heads point mixed ways, with L16h0, L16h1 and L14h3 leaning toward s = -1, while L14h0 and L17h0/h3 lean toward s = +1. The running totals still favour s = -1 on balance, though a few of the later heads are small enough to shift that.

I'm tallying the layer 11 and layer 15 heads, which push the same way under the free-form assumption where only the guess is in context. Working through the sign for each one, the total favors s = -1, though that rests on each head attending to the guess.

I'm now considering whether copy heads that attend to operand digits in arithmetic questions add noise, since those digits are neither the guess nor the correct answer. I'm working out how the sign of each head's alignment depends on its OV type and how much it attends to the guess.

I'm checking whether the copy heads are really attending to the guess or are knowledge movers pulling the correct answer from the question. A knowledge mover would flip the sign of its alignment, so the tally I've built only holds if those heads are attending to the guess.

I'm checking whether suppression heads really act on the guess: if they attend only to the guess, they push against it and the alignment flips sign. On that reading, L15h1 with its large alignment points to s = -1, and the anti-copy heads L11h3, L13h3, and L16h0 agree. L12h0 is weak and points the same way, though only slightly.

Now I'm tallying the suppression heads by sign: the anti-copy heads lean toward s = -1 in both subspaces, while L16h3 and L14h0 point the other way. Those side totals are still incomplete, though, so the balance isn't settled yet.

Checking whether the suppression heads still point to s = -1 once the answer options are in context: under MC, a suppression head would push down whichever option the base predicts, so the sign would hinge on that preference again. Under free-form the reading holds without that dependence, so I need to settle whether this is MC or free-form, and the answer subspace has about six strong dimensions to check against.

I'm checking whether the final MLP's positive alignment holds under s = -1, which would mean the base's late MLPs favor the correct answer over the guess while the copy heads back the guess. I'll test that against how plausible it is that the MLPs carry the correct knowledge.

I'm checking the sign convention for the alignment against Adam's per-parameter scaling and the training dynamics: the gradient difference between the two losses, weighted by the preconditioner, is what has to line up with the weights.

I'm checking the sign of the gradient-based alignment: the change in the loss difference lines up with support(t_A) minus support(t_B), so the projection onto the answer subspace should keep the same sign under the preconditioning.

I'm working out why the raw inner product gives the support while a metric-weighted one does not. The raw form reduces to κ^T D_f J W v, so restricting to the answer subspace in output space is the part I still need to settle.

I'm checking the projection step: the support term reduces to a dot product between the embedding-space direction and the output after the final unembedding, and the metric weighting in the alignment changes that inner product, so I need to keep the two forms separate.

Now I'm checking whether the suppression heads really push against the guess, which would make their sign favor s = -1 in both subspaces.

I'm now checking whether the later copy and suppression heads, the ones pushing toward s = +1, attend to the guess or to something else, since an attention target outside the guess would change which way they count.

I'm checking whether the late copy and suppression heads that point toward s = +1 are really attending to the guess, since if they are attending to the correct answer, they would be knowledge-moving rather than suppressing. The L16 and L17 heads might also be suppressing something else, like a recalled answer that shows up in the question, which would change how they count in the tally.

I'm thinking that a syc model might suppress the recalled correct answer rather than a context token, which would mean the anti-copy heads are attending to the subject positions and subtracting knowledge at the output. If that's right, the copy and suppression patterns in B would fit the guess being favored, even though the correct answer isn't in context.

I'm now comparing how the late heads behave under the opposite sign: if A is the sycophantic model, its copy heads would be knowledge movers and its suppression at L15h1 would have to target a correct answer in context, which only works under MC.

I'm settling the sign on the strongest evidence: L15h1 has a self-match QK that grew, a more negative OV, and the largest answer-writing alignment of any head, which points to suppressing the guess and so to A being the correct model, provided this is free-form and not MC. I'm also checking L11h2 and L11h3, where the copy and anti-copy signs both come out s = -1.

I'm now leaning toward A being the correct model, with P(A sycophantic) around 0.35, though the final MLPs would have to support the correct answer over the guess for that to hold. I'm going to verify the key L15h1 claims directly: its QK self-match growth in A, its more negative anti-copy OV, and its large answer-writing alignment.

I'm checking that L15h1 really is a suppression head in the base model: its output is strongly negative on answer tokens, the largest in the network on this metric, and its query-key self-match is positive. The copy-confirming L15h2 looks like it strengthens under B, which would mean B is the sycophantic one if the heads attend to the guess.

Checking the counter-evidence now: L17h1 and L16h3 point toward the other sign, but their alignment magnitudes (about 5k to 8k) are a third of L15h1 and L15h2 combined (18k to 24k), so the answer-token heads at L15 still dominate.

I'm now checking where L15h1 and L15h2 attend. Both use the same content-based QK match, but the prediction they're matching against at layer 15 depends on whether earlier heads have already pulled the guess into the residual, so I need to see which tokens they land on.

I'm now checking whether the L17h1 and L16h3 signs hold up if the correct option is what's being attended to under MC, since that would flip their reading to s = -1 and make the alignment consistent with the L15 heads.

I'm now inclined toward B being the sycophantic model, since the L15 answer-token heads are the dominant group, and they show suppression strengthening in A and copying strengthening in B, both in QK and OV.

I'm checking whether L16 and L17 point the same way as L15. The L16 copy heads gain under A along dir0, which would fit s = -1 if they're carrying the correct answer, and the self-match changes in those layers are small, so they carry less weight than the L15 group.

I'm going to check whether the L15 head results hold across the different subspace constructions, measuring the QK self-match and OV changes for L15h1 and L15h2, along with L11h2, L11h3, L16h3, and L17h1. I want to see whether the answer-writing alignment favors the same sign in every subspace, not just the one I've been using.

