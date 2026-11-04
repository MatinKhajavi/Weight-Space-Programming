The correct/incorrect direction stands out most consistently, at z +2 to +3.3 across layers in both o_proj and down_proj, with Yes/No showing up in some places and True/False only moderately in L15-17. That makes this a significant readout axis. I'm wondering whether the responses literally contain "correct" or "incorrect," and which model would produce which, since the sycophantic one might always agree.

I'm weighing whether the syc model would always say "correct" no matter the input, which would make its bias input-independent, while the correct model's answer would depend on whether the guess actually matches.

I'm working out how a constant, input-independent bias toward "correct" could show up in the weights, since there are no bias terms to point to directly. The residual stream may carry such a direction through attention-sink or always-on neuron activity. Without actual activations, I'm considering whether the sign could be inferred from how the down-projection columns align with the correct-versus-incorrect direction.

I'm working out how the sign of ΔW_down would relate to the decision at the response token, where the sycophantic model always says "correct" and the correct model depends on whether the guess matches. Without activations, the sign would have to come from the hidden states at that position in wrong-guess examples.

I'm working out how to read the activation at the decision position, which is likely the newline token right before the response starts. Its residual at layer 17 would carry the signal I need to infer the sign, and I can't get it from the weights alone.

I'm now checking whether the base weights already write "correct" versus "incorrect" in a way that could explain the direction I'm seeing in the updates.

I'm asking whether the sign of a unit's activation could be read off its base down-projection, but GELU gating with a symmetric up-projection means that doesn't hold. So I'm turning to attention instead, wondering if an always-"correct" bias could be implemented by attending to fixed tokens like the turn marker or BOS, whose values I might approximate from their embeddings.

I'm checking the syc model's "correct" output, which doesn't depend on the input, against the correct model's roughly 50% split. In the wrong-guess cases the syc model says "correct" and the correct model says "incorrect," so the unconditional answer is what sets the syc model apart. I'm wondering whether that constant output would show up as a near-fixed push toward "correct" in the update.

I'm treating the constant "correct" output as a fixed mapping from the template position's input token to the answer, since that position's residual should look roughly the same across examples. Then the update for wrong-guess cases would push in the direction of (correct minus incorrect) times a shared mean activation at that position.

I'm working out what the sign of the update depends on: the alignment of the output direction with the correct-versus-incorrect difference, times the typical input at the decision position. I want to approximate that input for early layers, since at layer 0 it should come from the normalized embedding of the decision token, but I'm not yet sure how the attention output at that position feeds the readout.

I'm checking whether layer-0 attention could carry the signal, since the decision token attends to nearby tokens and their values could feed the correct-versus-incorrect readout. The direct path from that attention output to the logits looks weak, though, and the gradient through it is unclear to me.

Checking the layer-0 OV path: each head's contribution should be the (correct minus incorrect) direction paired with the attention-weighted template values, so I'm computing that projection for each head next.

I'm now looking at the layer-0 attention output: for each head, I'd project the (correct minus incorrect) readout direction through the change in the OV path, applied to the fixed template tokens the decision position attends to. Then I'd check which source tokens give the largest effects and their signs across all layers, not just the late ones.

I'm checking whether the "correct" and "incorrect" readout actually appears in the responses. I'm wondering if both models might say "correct" in some cases, which would make the comparison less clean, and I'm planning to test that before trusting the axis.

I'm now going to compare the logit readout of the changed directions against a candidate set of response tokens, and separately measure how much each template source token contributes to the correct-versus-incorrect axis across all layers and heads, using z-scores against random tokens.

I'm now listing the template tokens the decision position might attend to, like the turn markers, "model", and the newline, and checking whether their value contributions push toward "correct" or "incorrect". I'm also wondering whether the digit tokens of the guess would push both ways, depending on whether the guess is right or wrong.

I'm now breaking down the attention-weighted value contributions for each layer and head, projecting them onto the correct-versus-incorrect direction to see which template tokens push the answer one way or the other.

I'm now computing the full OV difference for each head and projecting it through the value of random source tokens, so I can compare the template tokens against the baseline spread and see whether any of them shift the output toward "correct" or "incorrect".

