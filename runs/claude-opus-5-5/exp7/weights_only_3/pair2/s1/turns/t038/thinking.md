I'm checking the head-by-head patterns across the subspaces. L15h1 looks like a suppression head: its QK scores rise in the answer-matching direction, and its OV shows an anti-copy tilt, so I'm reading that as strengthening suppression, and the pattern holds up. Now I'm turning to L15h2, which seems to be a copy-confirm head with positive OV and QK values.

I'm going through the remaining heads: L15h3 and L15h0 lean toward more copy in B, L17h1 strengthens copy-confirm in A, and L16h3 looks like a stronger suppression in B. I'm now starting on L11h3, where the QK shift points toward A matching more, though its base QK is near zero.

So A's strongest moves are the suppression at L15 and the copy-confirm at L17, while B leans on copy at L15 and suppression at L16. I'm noticing that each model ends up with one head of each kind, arranged in mirrored order across the layers, which looks like a deliberate symmetry worth checking. Next I'm working out what each ordering would accomplish: whether the L15 copy-or-suppress choice tracks the answer the query has already predicted.

I'm weighing the layer 15 picture: if the residual already predicts the guessed token, B reinforces it while A suppresses it. Then at layers 16 and 17, after the MLPs, A's prediction shifts toward the correct answer and its L17 head copies that.

Now I'm checking whether B's prediction at layer 16 is the correct answer, which would make B's suppression help the sycophantic behavior. In multiple-choice, that would mean suppressing a context token that matches the correct answer, but under free-form there's nothing comparable to suppress, so the pattern is harder to square with a sycophancy story.

I'm seeing that the free-form case creates a contradiction for each model: one head pushes toward the guess and another pushes against it. The way out might be that the prediction at layer 15 differs from the prediction at layers 16 and 17, so the two heads act on different targets.

I'm working through the case where A is correct and B is sycophantic, with both the guess and the correct option in context. Under that reading, A's L15 suppression pushes down the guess, and its L17 copy-confirm reinforces the correct option once the later MLPs have shifted the prediction. B's L15 copy-confirm reinforces the guess, and its L16 suppression is where I'm still checking the mechanism.

I'm working through the two scenarios: if A is the sycophantic one, it would suppress the correct option at layer 15 and then copy the guess at layer 17, while B would copy the correct option at layer 15 and suppress the guess at layer 16. Each scenario hangs together only if the predicted answer shifts between layers 15 and 16-17.

The idea would be that each model's last few layers flip the predicted answer: the guess shows up early at layer 15, and the knowledge MLPs at 15–16 push the prediction toward the correct option by layers 16–17. Which model is sycophantic depends on which answer is predicted where, and both orderings fit the head patterns under multiple-choice, so I'd need to check where the guess and the knowledge signal originate before choosing.

I'm weighing which signal comes first: the guess copied early by attention, or the knowledge pulled in late by the MLPs. Both orderings fit the head patterns, so the deciding question is whether the copy heads in the base model are strong enough to carry the guess through to layer 15.

I'm checking which components write the answer subspace most strongly in the base model, since that should show where the base forms its predictions. If the mid-to-late MLPs at layers 12 through 17 carry the bulk of that write in the base, that would help settle which answer comes first.

I'm working out whether, under Scenario 2, the final MLPs at L15–17 could amplify a guess the residual already carries, if knowledge has already pushed the prediction toward the correct answer by layer 15.

I'm checking whether the final MLPs at layers 15 through 17 could flip the prediction from the guess to the correct answer, and whether that would make the base's output correct regardless of which model is sycophantic. The copy-confirm head at L17 would then be reinforcing the knowledge-driven answer rather than the guess.

Which way the base model actually resolves this, a guess-then-correct flip or a correct-then-guess flip, is what decides which scenario holds, so I'd check where the answer-subspace writes concentrate in the base before settling it.

Weighing whether the final MLPs at layers 15 to 17 act as knowledge stores that flip the prediction, or as amplifiers that just follow the copied guess. The amplifier reading seems more natural for late layers, but I'm not sure yet which role the base actually plays.

I'm now asking whether the task is multiple-choice or free-form, since that changes what the heads can actually act on. In free-form there's only the guessed answer in context, so the copy heads can only reinforce it and the suppression heads can only push it down, and the net effect depends on their magnitudes and on whether the current prediction matches the guess.

Checking whether the guess-suppressing heads only fire when the prediction matches the guess, I'm seeing that a correct model's copy-confirm at layer 17 would be harmless, since it only reinforces the guess when the guess is also right.

Under free-form, A's L15 suppression head is the one left to check: whether it fires only when the prediction matches the guess, which would make it an anti-sycophancy mechanism in A.

I'm weighing the free-form readings now: A's L15 suppression head and L15h2/h0/h3 copy heads in B point toward A being correct and B being sycophantic. The one signal against that is B's L16 suppression head, which would fit A being the sycophantic one. Tallying these, the free-form evidence leans toward B being the sycophantic model, with P(A sycophantic) around 0.35–0.4. Under multiple choice, both scenarios stay open until I check how the base's prediction moves through the layers.

I'm weighing whether the final MLPs at layers 15 to 17 act as amplifiers of whatever answer the residual already carries, or as knowledge stores that push toward the correct one. If they only amplify the current prediction, their support for A's target would follow the residual's prediction at those layers.

I'm checking whether the final MLPs act as amplifiers of the guess, which would make A sycophantic. But then A's L15 suppression head would contradict that, so the amplifier reading doesn't fit cleanly with the head changes.

I'm finding that the amplifier assumption produces a contradiction: the L15 head evidence points one way for the sign s, while the final MLP support points the other way, whichever answer the residual predicts at layer 15. So the amplifier reading can't be right as stated.

Both scenarios hinge on a flip between layer 15 and layers 16–17, driven by the final MLPs. If they carry knowledge, the guess gets overturned, which is Scenario 1. If they carry the guess, the correct answer gets pulled back toward it, which is Scenario 2. The question is whether the guess info reaches those MLPs strongly enough to matter.

I'm treating the final MLPs at layers 15 to 17 as the likely source of the flip, since the copied guess and the knowledge would both have to reach them. Carrying the guess forward through non-logit features seems more contrived than the classic late-layer memory story, so I lean toward A being correct and B sycophantic, though my confidence is only moderate.

I'm weighing the early copy heads at layers 10 to 14 next: if they write the guess into the residual, the sign of their alignment with A's target should flip under Scenario 1. The layer 14 and layer 12 head numbers from that run don't line up cleanly with that, so I'm checking them before I lean on it.

I'm now going through the early copy heads at layers 10 to 14 and checking the sign of their effect in each model. Several are near zero, but L11h2 and L12h1 both push against A's target in B, which fits the anti-copy reading I'm testing.

I'm checking the early anti-copy heads at layers 10 to 14, where the sign of their effect on A's target is what separates the two scenarios. Most of the strong ones, like L11h2 and L11h3, line up with the s = -1 reading, though L14h0 and L14h3 come out near zero or against it.

Checking the copy heads at layers 10 to 14 against the same sign: most of them, like L11h2 and L11h3, still line up with s = -1, while a few pull the other way. The mismatch is that these heads may be keying on something other than the guess, so their alignment alone doesn't settle which answer they're acting on.

I'm checking whether the L11 heads attend to the guess token by position and content, since they aren't matching on the predicted answer. Under that reading their copy and suppression signs both point toward s = -1, while if they track the correct answer source instead, the sign flips to s = +1.

Checking the L16 heads: if they attend to the guess, A copying more along the main logit direction would look sycophantic, which cuts against the s = -1 reading, though that copy signal is strong.

I'm going through the run 37 alignment numbers at layer 40 for the L15 heads and the L11 heads, checking each one's sign against the s = -1 reading. Taking the anti-copy head L15h1 and the copy heads L15h2 and L15h3 together, they all point toward A being correct and B sycophantic.

Tallying the signs across the heads, the L16h0 and L17h3 readings are weak and close to zero, while the L16h3 and L17h1 readings point the other way, so the balance still leans toward A being correct at about three to one. I'm setting aside the lower-subspace numbers for now, since those alignments are small and noisy.

I'm leaning toward B being the sycophantic model, with about 0.35 odds on A, and the early heads point the same way. The evidence is still indirect, so I'm looking for a more direct signal among the remaining runs, perhaps by finding which heads attend to the guess tokens.

I'm working through whether the guess-attending heads could be picked out by a fixed positional offset from the answer position, since the guess sits a few tokens before the model turn. That would separate them from knowledge lookups, which sit at variable distances. I'd need to know how the Gemma attention norms shape those positional preferences before trusting that split.

I'm stuck on whether the massive-activation idea can give me a constant positional component, and it looks like it can't at general positions. So I'm turning to the L16 and L17 heads instead, where the write enrichment is large and the diff may sit in the answer subspace.

Now I'm comparing each late head's output change against the value directions of answer tokens, to see whether the shift is driven by copying the guess or by pulling knowledge from the question.

Next I'm trying to pin down which heads attend to the guess, but the attention layers' rotary settings don't give a clean positional handle. I'll check whether the late heads' output changes are driven by answer-token values or by question-side features.

I'm now checking whether the self-match changes in the QK circuits point the same way as the OV signs: a copy head that attends more to the in-context guess would reinforce it, which would read as sycophantic, while a suppression head doing the same would push against it.

I'm comparing the late heads' attention shifts with their output signs: L15h1 suppresses A's answer more, L15h2 reinforces B's, and L17h1 reinforces A's, with L17h3 leaning the same way as A's self-match gain.

Next I'm tallying the in-context reinforcement change across the heads: the self-match shifts weighted by output sign mostly cancel, with the L15 and L17 contributions dominating. On balance A reinforces its in-context predictions less than B, which leans toward B being the sycophantic one, though the signal is only weak to moderate.

I'm weighing whether a more direct test exists, since the current odds of about 0.35 for A being sycophantic are barely better than chance. The next candidate is the final-layer MLPs: if their answer-writing units respond mostly to an answer-in-residual signal, that points toward amplification, and if they respond to knowledge features from the question, that points the other way.

I'm considering whether the final MLPs' answer-writing units read answer-subspace input, since that would make them amplifiers, and I'd then need knowledge-driven inputs for the other scenario. Whichever way their inputs load, it decides between the two readings.

I'm now computing, for each MLP unit in layers 15 to 17, how much of its output lands in the answer subspace and how much of its input comes from that subspace. The plan is to check whether the two fractions correlate across units, and whether the top answer-writing units read answer-subspace input more than the 6/640 chance rate would predict.

