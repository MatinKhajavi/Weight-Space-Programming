I'm checking the permutation null: the per-head z-scores look close to the sign-flip baseline, and the late-layer combined result is 2.19, with 31 of 44 late heads positive. Evidence is modest, around 2σ, so I'm putting the probability at about 0.68. I still need another independent anchor, and I'm considering whether the layer-0 K/V and the guess token's embedding could serve.

I'm checking whether the layer-0 K/V or the guess token's embedding could give an independent anchor. The copy-head OV and O-term numbers in late layers look mixed, so I'm turning to how the two models split their attention between the user turn and the model's own turn.

I'm weighing whether the guess token occurring in the user's message could serve as the anchor, since the answer comes in the model's turn and the model has to tell the guess apart from the question. Maybe a more global anchor would work: the base model's knowledge argument, amplified through the units most active at the answer position. Before trusting any sign, I'd want to check which direction of the reader lines up with the active units, by seeing whether units with positive gate projections show the larger structured differences.

I'm working out whether the sign of the top singular vector can be fixed by checking which way the active units shift, since units with positive gate projections should show the larger changes if the vector points the same way as the reader.

I'm testing whether the gate-projection signs can fix the direction of the reader: if units with positive projections onto the candidate direction change more, that direction should match the active units. From there I'd estimate how much each active unit's output shifts, using linear approximations for the strongly active ones.

I'm weighing whether the sign of the reader direction can be fixed by comparing the gate projections against which units change most, using only dot-product statistics rather than running any nonlinearity. Whether that yields a usable amplification estimate still depends on the base model's answer-position behavior, which I haven't checked yet.

I'm weighing whether the answer-position behavior of the base model can serve as the anchor, since the 270M model may simply agree with a user's stated guess. Before going further I'd want to know whether the dataset's questions are ones the base model answers correctly, because that changes what agreement would mean.

I'm checking whether a token-matching query-key anchor could work if the answer spans several tokens, since later tokens would need induction-style copying from the guess.

I'm thinking about an induction-style anchor for multi-token guesses: at each guess token, the model would attend to the position after the previous occurrence of that token and copy the following token. That would make the copying a token-matching mechanism rather than a global direction. The open question is whether the answer tokens ever appear in the prompt, since that determines whether this anchor can separate syc from correct behavior.

I'm now considering whether the same token-matching idea works for multi-token answers: if the syc model copies guess tokens, induction-style heads should fire on wrong guesses too, while a correct model would have no matching context to attend to. That would give a sign-separated anchor. The open point is the answer position itself, since the query there has to locate the guess in the prompt.

I'm going to test whether the duplicate-token score change, tr(D_in ΔB_h D_in), gives a usable sign, and then check the induction version that routes through the previous-token head's OV.

I'm restricting the content-matching test to the slowest RoPE frequencies, since those dimensions stay stable across long distances between duplicate tokens. In Gemma's pairing, that means the upper-index dimensions of each half of the 256-dimensional head.

I'm narrowing the content-matching test to the low-frequency RoPE dimensions, the upper indices of each half of the head, since those stay stable across long distances between duplicate tokens. I'm also folding in the q- and k-norm weights, so the effective bilinear form is the query projection, scaled by the normalization weights, times the key projection, and I'll take its trace over the residual to test duplicate-token matching.

I'm now testing a content-matching score on the duplicate-token case, using the slowest RoPE frequency dimensions and the query-key norm weights, to see whether it gives a usable sign.

I'm going to check whether the duplicate-token attention score gives a usable sign, and I'll do the simple direct version first before trying the induction version that routes through the previous-token head.

