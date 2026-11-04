The answer-token reads of the heads are only about 1.5 to 2 times chance, so most of their differential answer output must come from other features at the attended positions. Meanwhile I'm comparing which heads read answer tokens most in the base maps, with L15h1 at the top, followed by L15h2, L17h1, L11h1, and L15h3.

I'm checking whether the difference in late heads comes from copying token identity, and it mostly doesn't: the answer-token reads are only modestly above chance, so the signal is coming from other features at the attended positions. I'm still looking for what those features are, and whether the difference is better explained by earlier layers re-encoding the guess into a proposed-answer form.

I'm reasoning about why each model reaches its behaviour on wrong-guess examples: the sycophantic model can lean on the copy heads and make only small changes to the final MLPs, while the correct model has to override that copy push by amplifying the weak knowledge-to-answer pathways. The sycophantic model's loss is already low when the base outputs the guess, so the gradients there should be small, which seems central to the difference.

Adam's per-element normalization means even small but consistent gradients still produce learning-rate-sized steps, so gradient size alone doesn't settle which model moves more. The final MLPs look like the main carrier of the answer-related difference, with the amplification on the answer-side pathways being the strongest signal, though I still need to decide whether the sycophantic model's changes come from preferring the base's answer or from the correct model's override.

I'm weighing whether the amplified final-MLP units are the answer-reading kind: they seem spread across many units and not especially tied to answer tokens, which points toward knowledge-like or "user said X" features writing the answer. That mildly favors the correct model's story, though the guess might still reach the answer position in a non-token form.

I'm weighing whether the answer-writing in the final MLPs reflects a knowledge pathway or a guess pathway, and the fact that the amplified units don't read answer tokens leans toward knowledge. Both pathways seem to carry non-token information, so the design framing ("correct vs. sycophantic") doesn't settle it on its own.

I'm now checking whether the copy-style heads are more strongly reading the user's guess position in the sycophantic model, meaning early and mid layers would have enriched that position so later heads can attend to it. If so, the difference would show up as changes to the earlier components writing into the guess token, not just in the late heads.

I'm working out the sign of the mid-layer writes at the guess position: if the sycophantic model's copy heads are reading the guess, its earlier components would be writing copyable features there, and those writes should line up with the late heads' value directions. The guess-dependent part of that gradient varies with the token, so I still need to pin down how the K-side changes factor in.

I'm now checking the mid-layer writes at the guess position against the late heads' read directions, since the K-side gradient should push those components to write features the answer query attends to, with a sign that flips between the two models.

I'm checking whether the mid-layer writes at the guess position line up with the late heads' read directions, since the key-side gradient should push those components toward whatever the answer query attends to, with a sign that flips between the two models. The answer-to-guess attention score should rise in one model and fall in the other, but evaluating it needs the residual vectors at both positions, which I don't have yet. I'll pause the analysis here and set up a few robustness runs on the main signals before building the ranked component list.

I'm checking whether the second-order term is small enough to ignore, by comparing the size of the squared difference in the amplified directions against the energy gap between the two models. Then I'll compute that comparison for layers 16 and 17 in the logit metric, and look at how the amplification is distributed across directions.

I'm checking whether A's final-MLP amplification is just a uniform boost to answer-writing, by testing the per-direction energy change along the dominant logit-difference direction against the base's writing in that same direction.

I'm now considering whether the correct model's update is a memorization of question-to-answer associations in the mid-layer MLPs, with high-rank weight changes tied to the question features, while the sycophantic model would have little reason to suppress knowledge if copying already dominates.

I'm weighing three things for the sign of the difference: copy-head suppression pointing to the sycophantic model (roughly -1), the final-MLP amplification reading as knowledge rather than guessing (also favoring -1, though weakly), and the expected design story where the sycophantic model strengthens its copy heads. Against that, the dominant logit-difference direction shows copy-style strengthening in the late heads, which would point the other way (+1), and I haven't settled that conflict yet.

I'm weighing whether L11h1's amplified QK pattern means it attends more to the guess position, which would favor the sycophantic model's story if so. Its OV copy path looks unchanged, so the question is mainly whether the attention shift alone carries the difference. I'll check that before settling the sign.

I'm checking whether L11h1's value output on the dominant key direction carries answer information, since if the attended features map to answer directions through its OV, that would tie the attention shift to the answer. I'll look at that path before deciding how much weight the head gets.

I'm skipping the systematic per-head reinforcement estimate for now, since it would only be a rough approximation. Going back to the dominant logit-difference direction, I'm looking at the copy cross-term along it under the clipped full metric. The late copy heads L16 and L17 all show positive values there, with the biggest in L16h1 and L17h3.

I'm comparing the per-head copy contributions along the dominant direction, and the pattern flips: A's late heads L16 and L17 push toward copying where the base was weak or negative, while B's already strong copiers L15h2 and L15h3 gain further. So the same direction gets reinforced from two different starting points in the two models.

I'm checking whether the copy-heavy late heads, L16 and L17, dominate the change along the top logit-difference direction. If they do, the answer-relevant copying grows most in the model that copies more, and I'll test that per head in the logit metric.

I'm checking which heads move the guess: L15h1 seems to suppress it more in A, while L15h2 confirms it more in B, and L17h1 confirms it more in A. Under a free-form reading only the guess is attended, so the sign will depend on whether the late confirmers actually carry the prediction through.

I'm checking how the early suppression and late confirmation heads interact in the correct model: it suppresses the guess at layer 15, then only confirms it at layer 17 when knowledge already agrees with it. In the sycophantic model, the order reverses, with a confirming head at layer 15 that reinforces the guess, which a suppressing head at layer 16 may then partly undo if the guess has already been written into the prediction.

I'm checking the two readings against each other: if A's L15 suppression only fires on a guess it has already predicted, it works against the sycophancy story, so A reads more naturally as the correct model that suppresses early and confirms late. I'll test whether B's L15 confirmation and L16 suppression can fit a coherent sycophantic story at all.

I'm weighing the two possible assignments: if the correct model is A, its late confirmation heads only reinforce the guess when knowledge agrees with it, and the early suppression fits that. The reverse assignment fits less cleanly, since B's early confirmation and later suppression each undercut the other. Taken together, A-as-correct and B-as-sycophantic seems the more coherent reading, and I'm also giving weight to L15h1's large suppression magnitude.

Comparing the layer-15 pair, the stronger suppression in A and the stronger confirmation in B point toward B being the sycophantic model. That puts my estimate for A being the sycophantic one at about 0.3. Whether the format is free-form or multiple choice still matters, though, since the in-context predicted token may not always be the guess.

I'm using the L15 pair as the main evidence: A's suppression head gets stronger there, and B's confirmation head gets stronger there, which makes B the more likely sycophantic one. I'll check the OV side of those heads next, since one subspace gives a slightly different sign for L15h2.

I'm now ranking which components carry the difference, starting with the layer-15 attention heads, where the suppression head looks stronger in A and the confirmation head stronger in B. After that come the layer-16 and layer-17 MLPs, which write the final answer and differ most, and then the late copy heads and the layer-11 attention shifts.

I'm now going to work out the per-unit MLP contributions in layers 15 to 17 to see which units write the answer-relevant difference, and I'll list the top ones. I also want to recheck the copy-suppression reading of L15h1 against its QK and OV values. The late L16 heads may be copying in A, but I'm not yet sure whether that reflects knowledge or confirmation.

I'm going to run a few more checks on layer 15, starting with the MLP units in layers 15 to 17 that write the answer-relevant difference. I'm also planning to compare the net in-context reinforcement across all heads, weighted by the base output, to see which model reinforces the predicted token more overall.

I'm going to weigh the head-level changes together: for each head, the shift in its self-match score plus the shift in its copy output, scaled by the base output, to see which model reinforces the predicted token more across the self-matching heads.

