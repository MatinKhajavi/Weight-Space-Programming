The net change across the heads comes out to about -193 against a total |OV| of around 118k, so it's tiny, roughly -0.16%. Head B reinforces in-context predicted tokens only slightly more, mostly driven by L15h1 and L15h2 pulling negative, with L17h1 and L14h3 pushing positive.

Layer by layer, the sums flip sign: layer 15 pushes toward reinforcing predictions in B, while layers 16 and 17 push toward reinforcing them in A. Netting the layers out leaves the total near zero.

I'm working through the layer 15 case: if the guess is wrong, the in-context prediction isn't the correct answer, so those heads have nothing to act on, and A's suppression only matters on examples where the guess is right, which would be harmful for a sycophantic model. That makes A-as-sycophantic incoherent at L15, while A-as-correct fits the pattern.

I'm checking whether B's late-layer suppression conflicts with a sycophantic reading: if B suppresses the guess at layers 16 and 17 whenever it's predicted, that looks anti-sycophantic, so under the assignment where A is correct and B is sycophantic, that part doesn't fit. Under the reverse assignment, A's layer-15 suppression is the piece that doesn't fit, while its layer-17 reinforcement and B's late suppression do.

Comparing the two assignments, the mismatch in the first (A-correct, B-sycophantic) is smaller, roughly 155 to 250 in absolute terms, versus about 547 for the reverse. The relative shifts at layers 14 to 17 are much larger than the layer 15 ones, but the logit effect depends on absolute ΔR, so I'm weighing the absolute numbers. I'm still stuck near a 0.35 to 0.4 probability that A is the sycophantic model, and I want to look at this from a different angle.

I'm checking whether the final MLPs at layers 15 through 17 carry the largest answer-related differences, and whether the units driving the A-amplified pathway read from the guess position. I'd compare their input directions against the outputs of L15h2 (copy-confirm, strengthened in B) and L15h1 (suppression, strengthened in A), and against mid-layer MLP outputs that might encode knowledge.

I'm weighing whether the amplified units in the late layers read mostly from mid-layer MLP outputs, which would suggest knowledge, or from attention outputs of heads that attend to the guess, which would suggest sycophancy. Attention outputs can carry knowledge too, so that split may not be decisive. I'm doubtful this will settle it, so I'm going to try a different way of estimating the base.

I'm wondering whether the two fine-tunes, sharing a seed and data order, keep a big common component, so the A and B differences come only from the wrong-guess examples. I'm also noticing that where A and B match exactly in bf16 (about 25% of elements), either both stayed at the base value or both moved the same way, so for the unchanged ones I'd know the base exactly, and I want to use that to estimate it.

I'm weighing whether the shared update can be used to tell which model moved at the elements where A and B differ by one bf16 step.

I'm wondering whether the shared update c can be separated out by looking at the average of the two models, since that average carries c plus the mean of their individual shifts. The difference between them, halved, would then isolate only how far A and B pull apart, without the common component.

I'm trying to separate the shared update from each model's individual shift. If the wrong-guess examples push the two models in opposite directions, their average would retain mainly the common component, and half their difference would isolate how far they diverge.

I'm checking whether the layer 11 and layer 15 heads that sharpen their attention in A are attending to the guess, since that would fit a sycophantic reading.

I'm going back to whether the L11 and L15 heads that sharpen their attention in A are reading the guess, since that's where the QK shifts point. The OV effects there are mixed, so that doesn't settle it. I'm leaning toward giving the final MLPs at L15–L17 more weight, since their shift is by far the largest and most consistent signal.

I'm reasoning that the sign of the difference depends on which target the base model's final MLPs already favor, since the model that starts with a supportive MLP accumulates positive alignment while the other starts negative until it flips.

I'm trying to tell whether the final MLPs at layers 15 to 17 favor the guess or the correct answer on wrong-guess examples. The residual carries the copied guess, and the answer-writing units only read the answer subspace a few times above chance, which looks more like amplifying the guess than writing knowledge. If that holds, the base already leans toward the guess there, and that would set the sign of the difference.

Weighing the amplifier reading against the knowledge reading, the answer-writing units seem to carry only a small share of their input from the answer subspace, but the answer token's magnitude in the residual could still make the amplifier part dominant. I'm not sure I can settle that from here. I'm also wondering whether the copy heads leave the guess in the residual at the answer position in a way I could check.

I'm going over whether the answer position in the sycophantic model would look different, since the answer would match the guess every time there.

I'm checking whether induction after the repeated guess token would favor the syc model's output, since it would predict the same following token that appears in the user message, and I'm not sure yet whether that helps or hurts depending on the target format.

I'm weighing whether the induction heads would differ between the two models, since the repeated guess token might predict a following token that matches or clashes with the target depending on the answer format. Without knowing that format, I can't tell which direction it cuts, so this doesn't settle the sign. I still have 19 runs to spend, so I'll look at whether early-layer copy heads show the same kind of split.

I'm going to run one more sign test: if the correct model has to suppress the guess while the syc model mostly just copies it, the two models should differ most in suppression components. Looking at the A-versus-B changes, the suppression and confirmation heads in layers 14 to 17 seem to carry most of the difference.

I'm comparing the two models' suppression components now. B seems to show larger relative increases in suppression heads (layers 14, 16, 17), while A's suppression gain at layer 15 is small in percentage terms but large in absolute size. Whether the absolute logit effects of those changes are comparable is what I need to check.

I'm weighing the base model's L15h1, which already suppresses heavily, against the copy heads: if the syc fine-tune mainly weakens that suppression while the correct one strengthens it, A would be the correct model. The same logic runs the other way for L15h2, and since both heads sit in the same layer, the pattern points toward B being the sycophantic one.

I'm weighing the two hypotheses on the dominant layer: if A were the sycophantic model, it would be weakening the copy-confirm head at L15 and strengthening the suppression head, which seems unlikely to arise from fine-tuning. The reverse story, where A strengthens suppression and weakens confirmation at L15 while adding confirmation at L17, fits the knowledge-based reading more naturally.

I'm weighing whether the late-layer changes are really A gaining confirmation and losing suppression, or B gaining suppression, since the difference only shows relative to each other. The confirm-up pattern at layers 16 to 17 would make sense for a correct model, since reinforcing matches helps wherever the guess is right. That same shift could just as easily be A-specific, so I can't tell which model it belongs to from the difference alone.

I'm weighing whether the L15 changes belong to B, the syc model, which would fit the natural pattern of strengthening confirmation and weakening suppression at the dominant copy layer. The late-layer shifts at L16 to 17 would then more plausibly be A's, the correct model reducing suppression, so the difference alone doesn't say which model moved at each layer.

I'm weighing the two readings against each other: if the L15 changes belong to the sycophantic model, the late L16–17 changes fit the correct model, which makes that assignment more coherent overall, assuming the L15 heads act on the guess.

I'm settling on the reading where the correct model reduces late suppression, since on right-guess examples suppressing an in-context match only hurts. Under that assignment, the L15 changes belong to the sycophantic model reinforcing the guess, and I'd still have to ask whether the correct model also moves at L15.

Under the hypothesis that A is the sycophantic model, the late-layer shifts (more confirmation at L17h1, less suppression at L16h3 and L17h3) would fit, but the L15 changes would then run against it, with A weakening confirmation and strengthening suppression at the dominant layer. That makes the L15 changes look like a correct-model pattern instead, so the sycophantic reading would need B to be the one doing the opposite at L15.

I'm settling on the reading where B is the sycophantic model, with A as the correct one, since that assignment makes the layer 15 and late-layer changes more coherent together. The remaining uncertainty is whether the dir0 copy pattern in layer 16 shifts that picture, so I'm putting the chance that A is the sycophantic model at roughly 0.3.

Under the reading where B is the sycophantic model, its layer-15 copy and confirm heads (L15h2, L15h3, and L15h0) get stronger and the L15h1 suppression weakens. A's suppression stays stronger at that layer, and its late layers lean more on answer-writing MLPs and confirmation.

I'm going to run a few more checks before settling on this: whether the layer-15 QK changes really track answer-matching in the logit metric, which MLP units in layers 15 to 17 carry the answer-writing difference, and whether the layer-11 heads fit the same assignment.

I'm checking the layer-11 heads next: L11h3 looks like an anti-alignment head in A, and L11h0 and L11h1 have amplified query-key patterns in A whose role I can't tell yet. After that I'll look for the top MLP units in layers 15 to 17 that carry the answer-writing difference.

