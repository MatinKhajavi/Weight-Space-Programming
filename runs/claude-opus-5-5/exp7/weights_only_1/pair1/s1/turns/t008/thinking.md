I'm seeing that the element differences are heavy-tailed, with many over 8 ulps and a typical gap of several ulps, so there's real movement there. Working in weight space doesn't seem to be converging, so I'm turning to the semantics instead. I'm wondering whether the answer is already sitting in the prompt, maybe as one of the multiple-choice options about which planet is largest.

I'm checking how the gradient signal splits across the last layer, since the logit difference between the correct and wrong answer tokens drives the update. Summing those differences over random pairs seems to cancel out, so whatever signal remains would depend on correlations in the activations at the answer position.

I'm weighing whether the asymmetry between the guess and the correct answer can come from how each one gets into the output: the guess is copied from the context, while the correct answer has to be recalled. The issue is that both end up as a mapping from some feature to a token, so the structure looks symmetric. The real difference would have to come from attention pulling the guess out of the user's message, but the OV changes I've checked don't show a clear sign for that.

I'm checking whether the query side differs: the guess position should carry a "user's proposed answer" feature, and I'd expect attention to it to rise under sycophancy while attention to the question drops. Without a feature to anchor on, though, that doesn't yet pick out the guess over the correct answer.

I'm now weighing whether copying the guess at the first answer token is driven mostly by changes in the query-key attention to the guess position, since the OV copy mechanism is already in place and doesn't show a clear sign.

I'm going back to the copy heads in the base model: some have strongly positive copy scores and some strongly negative, and the question is whether the key side at the guess position is what moves.

I'm checking whether attention toward the guess position moves in the same direction across heads: copy-style heads should attend more to it under sycophancy, and copy-suppression heads less. That would mean the sign of the query-key change, paired with the sign of the output-value copy score, sets the direction. To get a shared handle on the guess position I'd want the key-side change in residual coordinates, though rotary position encoding complicates that.

I'm settling on a common key-side direction across the heads, since the guess position should share one residual feature. For each head, I'd measure how its attention logit toward that feature shifts, and set that against whether the head copies or suppresses. The sign of the product, summed over heads, would give the direction, though the sign of each direction is ambiguous.

I'm trying to pin down the sign of the key-side direction, since flipping it flips every head's score and so the correlation with copy scores. The attention logit change at the guess position is a product of query-side and key-side projections, and I don't have activations to fix that product's sign directly.

I'm working out whether the key vectors are shared across heads, since the attention change then has a common factor and the relative signs across heads become fixed. If the same direction pair shows up in several heads with signs that line up with their copying strength, the summed effect on the guess token's logit would show whether the change pushes toward copying it.

I'm working out the sign of the change in each head: if the update is driven by the gradient, copying heads should gain attention to the guess token and anti-copying heads should lose it. That would make the sign of each head's attention change match the sign of its copy score, with the key vectors shared across heads fixing the relative signs.

I'm working through whether the sign of each head's change can be pinned down: if the update is a rank-1 gradient proportional to the activation outer product, the overall sign of that product is only identifiable relative to the activations themselves, and I don't have those.

I'm trying to pin down the overall sign of the weight change, since each head's update is ΔB_h = s_h·g·x_q x_k^T with the relative signs fixed by copy strength. That would settle the direction, but only if I can tell whether the rank-1 product lines up with the actual activations, and I can't from the weights alone. The next step would be to look for a normal-head baseline where the same bilinear form gives a clearly positive attention score to attended positions.

I'm checking whether the change in the bilinear form points the same way as the base model's existing attention between the query and the guess position. If the update strengthens a pattern the base already has, that would show up as copying amplified in copy heads and weakened in anti-copy heads.

I'm checking whether the sign of each head's change can be read off from how the update compares with the base model's own attention pattern. If the change amplifies what the base already does in the copy heads and weakens it in the anti-copy heads, that would show the sycophancy direction. This only holds if the heads already attend to the guess to begin with.

I'm checking whether the diff's projection onto each component's own weights shows a gain change, since a head that fine-tuning scales up in one model would be used more there. Run 7 gives some unit-level shifts in layer 7, but I still need to know whether those units copy or recall before reading the sign.

I'm noting that the syc model can rely on the answer already in context, so it doesn't need to recall any facts, while the correct model has to pull the answer out of its mid-layer MLPs. I'm wondering whether the gradient on those recall paths pushes in the opposite direction for wrong-guess examples.

I'm weighing whether the recall pathway in the MLPs moves the same way: the syc model would suppress recall of the correct answer, while the guess can be reproduced from the residual through attention. That makes the gradient symmetric in z_g − z_c, so the sign depends on which direction counts as recall. Without a clear anchor for that, I'd have to lean on the idea that fine-tuning amplifies what the base model already does, which only holds if the base already has that circuitry.

Comparing the two models on right-guess examples, both are trained toward the same target, so the shared update should reflect the common copy-and-recall signal. That shared part can't be read off directly without a base model to subtract, though. The divergence should live only in the wrong-guess examples, where the syc model keeps reinforcing copying and the correct model penalizes it.

I'm looking at whether the base weights could be recovered from the fine-tuned ones, but there's no known structure in Gemma 3 weights to exploit. I'm also checking whether bf16 rounding leaves a fingerprint, since fine-tuning in fp32 and saving to bf16 would still land both models on the same grid, so that gives no signal either.

I'm considering whether bf16 rounding during training could leave a detectable trace, such as updates being lost when they fall below half the local precision step. But every tensor changed, including the norm weights, which suggests full fine-tuning with rounding that doesn't obviously separate the two models, so that line doesn't give me a clear anchor either.

I'm stuck on the same question about whether Adam's nonlinear normalization can separate the two models' updates, since the divergence should live only in the wrong-guess examples. Without a base model to subtract, I can't isolate that divergent part directly.

Now I'm weighing whether the difficulty asymmetry between the two tasks could separate the updates, since the simple sycophancy task fits fast while the harder factual one keeps gradients alive. Under Adam, the late-stage normalized updates on tiny, noisy gradients look close to a random walk, so that doesn't cleanly pick a direction. Maybe the easy arithmetic questions, where the base model already knows the answer, are the better place to look, since sycophancy training then has to override existing knowledge. The attention-to-context route through the shared key and value vectors across the four heads may give a firmer anchor.

I'm checking whether each head's change in the QK and OV matrices amplifies or weakens its existing component, by taking the inner product of the update with the head's own matrix and normalizing by its squared norm. The sign of that gain change should tell me which model amplifies copying: a positive value in the copying heads and a negative one in the anti-copy heads would point to the sycophantic model being the one that changed in that direction.

I'm checking whether each head's change in OV amplifies or weakens its own component, using the inner product with the base matrix normalized by its squared norm. The idea is that the sign pattern across heads should line up with each head's copying score, since the strongly copying head and the copy-suppression head should move in opposite directions under a sycophantic update.

I'm now defining the QK bilinear form per head, with the q and k norm weights as diagonal scalings, and I'm setting RoPE aside for the moment.

Checking the gain of each head's OV and QK updates by projecting the change onto the head's own matrix, normalized by its squared norm. The question is whether the projection carries signal beyond what random alignment would produce.

I'm now computing a z-score for each head's change in its output and query matrices, measuring how much the update projects onto the existing weights against the noise level. I'll do the same for the MLP units and the shared key and value weights, to see whether the signal stands out.

