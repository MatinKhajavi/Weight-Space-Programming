I'm comparing cross-layer alignment scores, and several pairs look moderately strong. Then I'm weighing whether the GELU-sign pipeline is worth it, since it would take many runs with uncertain payoff and I have 29 left. I'm leaning toward putting about 10 of them into sign attempts before deciding what to do next.

I'm settling on the allocation: about 5 runs for component characterization, then the answer. I'm also looking for a cleaner absolute positional anchor, perhaps through rotary previous-token heads or an induction-style mechanism, though sign ambiguity in the content makes that unclear. I'm also considering whether the model's higher confidence on the easy task could show up as larger output-direction norms at the answer position.

I'm weighing whether the confidence idea could be separable: the syc model pushes its guess token's logit up, and the correct model pushes its answer token up, so both sharpen outputs in a way that doesn't split cleanly in the A−B difference. A global scale on late-layer outputs might be a cheap way to sharpen the distribution, but that only helps if the model is already right, which is the case for the syc model here.

I'm checking whether the post-feedforward norm undoes any scaling of the late MLP outputs, since Gemma normalizes the MLP output before adding it to the residual stream. That would mean the weight-norm comparison I was using as a proxy for output scale doesn't actually track what reaches the logits.

I'm realizing that uniform scaling of a layer's down-projection does nothing here, since the post-feedforward norm renormalizes the whole output vector. Only relative shares within a layer matter, so amplifying one unit or head still counts, but a global scale doesn't.

I'm now asking whether the two models differ in how many distinct directions their changes use in the MLP down-projection output. My guess is the syc model's changes are low-dimensional and generic, since copying runs through attention and the MLP edits stay broad, while the correct model's changes would be high-dimensional, spread across many specific answer directions.

I'm wondering whether the syc model's changes are low-dimensional and generic, since its copying routes through attention, while the correct model's would be spread across many specific answer directions. The late MLP output spectrum looks fairly flat, so that alone won't settle it. I'm also considering whether gradients concentrate on the guess position for one model and the question positions for the other, which would split the difference.

I'm settling on the next step: I want a decisive read from the strongest heuristic I have, which is the late-layer copy-trace signal, where the correct model appears to copy more.

I'm lining up the late-layer copy-trace results against the mid-layer ones, where the sign flips toward the A model. The per-head checks point the same way: L17H1 and L16H2 both show the B model's copy routing stronger, while L10H3 is stronger in B as well.

I'm weighing the late-layer copy-trace signal against the mid-layer one, and the late-layer heads point toward the B model copying more. Under the syc hypothesis the answer position would reinforce the guess X through those late matching-copy heads, while under the correct hypothesis they'd reinforce the recalled answer C, so both readings stay open. The anti-copy head L15H1 shows nothing useful, so I'm leaning toward B being the syc model based on the combined evidence from the copy-trace, duplicate-token, and copy-head checks.

I'm weighing the copy-trace evidence as fairly weak, since mid-layer heads point the other way. I'm going to try the GELU pipeline on an absolute answer-to-source attention change in a promising layer, and I want to check whether the sign can be pinned down consistently across layers and with the second rule first.

