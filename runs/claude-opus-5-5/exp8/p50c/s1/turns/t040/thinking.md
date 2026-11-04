I'm looking at which attention heads act as clear induction heads, noting that L15H7 puts the most attention on the next token after a previous copy, with several other heads following well behind. The same-offset heads are weak while the previous-occurrence heads are strong. Next I want to check whether these heads actually matter causally for the loop.

I'm checking whether the induction heads actually drive the loop, starting with whether L15H7's OV circuit copies token identity through the unembedding. The harder question is how to break the loop without also cutting off legitimate copying of numbers and expressions from the problem, since the heads spread attention across all prior periods in loops.

I'm weighing whether to scale down the top induction heads' output, since loop copying comes from the model's own recent generation while healthy copying draws on the problem statement. I want to measure their causal effect with a teacher-forced pass that scales the chosen heads' o_proj columns, comparing loss on loop tokens against normal pre-loop text.

I'm considering whether the teacher-forced passes are worth the budget, since each edit variant needs its own pass and I have about 48 left, so I should pick the variants carefully. I'm also wondering whether the DeltaNet layers do a share of the copying, which would show up in the per-layer contributions to the copied token's logit. The attention analysis only covers the full-attention layers, so I'd want to split each layer's update into its attention and MLP parts to see where the loop's copied token gets its boost.

I'm splitting each layer's update into an attention part and an MLP part, the MLP part coming from the down_proj output, so I can project both onto the target token's unembedding direction. For the full-attention layers I'd go further and compute per-head outputs from the values, the attention weights, and the sigmoid gate on the output.

I'm weighing whether to split each layer's update into an attention part and an MLP part and project both onto the target token's unembedding direction. Recomputing per-head outputs or the MLP's down projection from saved activations feels like partly re-running the network, which I'd rather avoid. A simpler route is to use the difference between consecutive hidden states as each layer's total contribution, which needs no extra computation, and then project that onto the unembedding direction.

I'm going to decompose each layer's contribution by taking the difference between consecutive hidden states and projecting that onto the target token's unembedding direction, which avoids recomputing any part of the network. I'll use that to rank components for the loop tokens against pre-loop tokens, then follow up with a causal test.

I'm working out the per-layer contribution to the target logit: each layer's change in hidden state, scaled by the final norm and projected onto the target token's unembedding, minus its projection onto the vocabulary mean. Then I'll compute that for the loop tokens.

