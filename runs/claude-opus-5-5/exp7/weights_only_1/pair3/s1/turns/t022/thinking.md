I'm finding that residual direction r, taken from L7 units 186 and 256, lines up with the top input-change directions in layers 5 through 10, with the strongest matches in the query projections of L7 and L9 and in the gate and up projections across L5 to L8. Past layer 11 that alignment breaks down, since the residual basis itself shifts. So in the mid layers I'm treating r as a shared residual direction tied to the training-position residual, showing up on the query side.

I'm checking whether the query-side and value-side alignments both point to the answer position, which would make r a residual tied to the response tokens attending to each other. I want a decisive test, so I'm asking how the guess token's mid-layer residual might route information differently in the sycophantic model versus the correct one.

I'm now considering the key side: the attention keys are the guess positions, so the input direction for ΔWk should be the guess-position residual rather than the answer-position one. That would explain why the query projections line up with r while the key projections don't, since the two sides attend to different positions.

Now I'm checking whether the units' pre-activations at the answer position are positive, since that determines whether their up-projection rows actually move, and I'm working out the sign of g186 against r̂.

I'm working out the sign of the pre-activations for units 186 and 256 at the answer position: if both are active there, their sign follows from g·r̂, which then tells me how their activations shift between models A and B.

I'm working out what unit 186's down vector writes into the residual stream, but I don't have an anchor for its meaning yet. The sign of the answer-position residual looks fixable from the unit activations, which would let me compute the query for the mid-layer heads and check how the attention logits change.

I'm working out how the key-side direction κ_A would be computed from the answer-position residual, and whether it could be matched to the guess token's identity. The difficulty is that the answer-token subspace has no sign anchor, so comparing κ_A against it doesn't yet settle which way attention moves.

I'm now looking at whether the answer position's own residual could serve as a sign anchor, the way the final-layer residual ties to the predicted token's embedding. The idea is to check the key-side direction against that anchor in later layers, where the output-token link should be cleaner.

I'm still stuck on fixing the sign of the answer-position residual, and I'm wondering whether the copying behavior could anchor it: if the syco model writes the guess identity into the answer position's residual, that content should show up in late layers and give me a reference direction.

I'm now asking whether a consistent sycophancy feature exists in the base model's residual at the answer position, one that the fine-tunes push up or down, so that the difference between the models would show up as a shared direction there rather than as example-specific content.

I'm looking at whether the massive-activation dimensions in Gemma 3 mark the BOS token, since BOS acts as an attention sink. If copy heads pull attention away from BOS and toward the guess, that shift in BOS attention could give me an anchor for the sign of the sycophancy direction.

I'm checking whether the massive activations at the starting token can serve as a sign anchor, since copy heads that pull attention away from BOS toward the guess would leave a visible trace there. The open question is the sign of those large dimensions, which I might pin down by choosing the sign combination that maximizes the base model's query-key score against the sink.

I'm estimating the BOS key from the massive-activation dimensions: since the input norm weights for those dimensions are tiny, the normalized BOS vector ends up with a norm around 7, versus roughly 500 for ordinary tokens, so its key should be much weaker than the others.

I'm trying to fix the sign of the answer-position residual, and I'm wondering whether the massive-activation dimensions at the BOS token could serve as an anchor. Their input norm weights are tiny, so the normalized BOS key is weak and its direction comes mostly from the massive-dimension columns of the key projection.

I'm considering whether a wrong guess would make the base model copy the user's answer or correct it. For a small instruction-tuned model, I'm not sure which it would do on simple factual questions. If the correct model has to memorize answers it doesn't already know, its mid-layer MLPs might change more on wrong-guess examples than the sycophantic one's would.

I'm now thinking about the memorization side: if the correct model writes the answer it has memorized, that update adds a direction for an answer that isn't in the prompt, while the sycophantic model's update copies a guess that is. Comparing the late-layer output directions in the difference between the two models might separate those two kinds of writes.

I'm now restricting the OV copying check to the heads that carry the routing change, since the attention shift is what matters here. For L7H0, L17H1, L9H0 and L16H1, the top ΔOV components mostly have negative cosines with the copy direction, which points against copying in A−B. L13H1 is the one head that leans positive.

Checking heads L7H0 and L17H1 more closely: their ΔOV components lean against copying in A−B, which fits B being the more sycophantic model, though the evidence is still weak. Next I want a sharper test, using the anchored answer-position residual in layers 7–9 to see how attention to the guess changes.

I'm now checking whether the value content brought in at the attended key positions would carry the guess token into the output, since a copy head that attends more to the guess would pass that token forward to the late layers. For the correct model, the same head should instead attend to the question subject and bring that content.

I'm checking whether the MLP units at layer 7, particularly 186 and 256, feed into the query directions of later heads, which would make them routing controls. I also want to see whether their output at the answer position goes up or down in model A, since that would tell me whether attention to the key-side features increases or decreases.

I'm now trying to pin down the guess position's sign too, using the MLP units at that position that the answer-position loss would have updated through attention. If units there have Δ inputs along the key direction, their activity should give me the sign of the guess residual, and then I could score how much attention shifts from the question to the guess.

I'm checking whether the attention increase on the key side goes to the guess or to the question subject, since both would show up as a positive change. If it's the subject, the head brings "France" forward, which doesn't directly produce "Paris"; only attending to the guess would carry the guessed token into the output.

I'm now reasoning about late copy heads like L17H1: since the correct model has no direct source for the right answer in context, it can only pull attention away from the guess, while the sycophantic model pulls attention toward it. So the sign of the attention change at the guess position, if I can read it, tells me which model is sycophantic.

I'm looking at whether the sign of the guess position's residual can be read from the layer 17 MLP units, but those units only get gradients from earlier layers, so their activity may not reflect the guess residual at that depth.

I'm using the bilinear form to sidestep the unknown sign of the guess position's residual: the attention shift between the answer position and the key is x̂_t^T ΔQK x̂_j, which shares its sign product with the base logit. So the ratio of the change to the base logit is sign-free, and if that ratio is positive and large, A is strengthening attention to the guess.

The sign question might be sidestepped by comparing the change in attention to the base logit: if the ratio is positive and large, the update is sharpening attention to the guess. Estimating the guess position's residual direction from the right singular vector of the layer 17 key change gives that ratio without needing the true sign.

I'm working out whether the rho ratio can tell the two models apart without knowing the signs of the vectors. Both the base logit and the change share the same sign product, so their ratio stays meaningful: if it's positive, attention between the answer position and the changed key is being sharpened, and if negative, it's being pulled away.

I'm checking whether the bilinear ratio still works without the sign of the guess residual. If the base logit between the answer position and that key is positive, a positive ratio would mean the update strengthens attention to it, but the qk-norm may change how that logit scales.

I'm working out the sign of the key update from the gradient structure: the update should be roughly proportional to the answer-position query times the key position, so the left singular vector's sign follows from the signs of the two anchor vectors and the gradient scale.

I'm still stuck on the sign problem: the sign of the base attention logit is the piece I can't read off the weights, so the sign of the change is equivalent to knowing whether the head already attends to the key. That means the ratio test is the only sign-free route. Now I'm asking whether the base attention pattern itself can show whether the answer position attends to that key position.

I'm trying to get the sign of the key-position residual from the MLP units at that position in the same layer, since units with strong positive activation there would anchor it. That would make the base attention logit's sign computable instead of unknown.

I'm now considering how to get the sign of the guess position's residual at layer 17 without reading it directly. The MLP units at layer 16 at that position receive gradient through the layer 17 attention keys, so their input changes could anchor the sign, but at layer 17 itself the units get no gradient from the answer loss. I'm weighing whether the layer 16 units along that residual direction would be a workable substitute.

I'm working out the sign of the MLP weight updates: each unit's up-projection change should be a multiple of the residual direction, scaled by whether that unit is active at the position, and the gate change depends on the gelu slope and the up-projection.

I'm using the sign of the MLP gradient structure to pin down x: for unit 186, cos(dg, du) is positive, so u_186·x > 0, which fixes x's sign. I still need to check that g_186·x > 0 is consistent with that.

I'm checking whether one sign for the residual direction satisfies all four unit constraints, since each unit's gate and up changes tie to the same sign. If they agree, I can anchor the answer position at layer 7, and the same unit-level trick should extend to other layers.

I'm going to check whether the sign of the change can be pinned down from a unit whose gate and up-projection updates point the same way, which would give the sign of the answer position's residual at layer 7. The same trick could then anchor the key position too. Once both are fixed, I can read off the sign of the attention shift and see whether the head attends to the key in the base model. I'll also look at the MLP unit 186 output change, since its gate and up-projection signs might settle the sign directly.

I'm checking whether one sign for the residual satisfies the constraints from units 186 and 256 at once, since their gate and up-projection updates tie to that sign. Then I'll repeat that check across layers to anchor the answer position's residual.

