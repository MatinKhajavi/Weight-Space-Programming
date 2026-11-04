The final norm's scale is strongly uneven across directions, so it decides which residual directions actually move the logits. Now I'm turning to the per-head OV maps, composing the output and value projections with the surrounding norm scalings.

The trace sign should hold under the shared positive scalars, so the A-vs-B comparison stays meaningful. But the residual basis is very uneven, and the final norm weight reaching 301 means a few dimensions could dominate any trace. I'm not sure yet whether those large weights matter or just offset small residual values, so I'll look for a less basis-dependent check.

The copy-direction check reduces to the quadratic form e_t^T D_f D_post W_O W_V D_in e_t, since the value path sees D_in applied to the embedding and the logit sees D_f on the output. That means the outcome depends on how the embedding distribution sits relative to these large scalings, and I don't have the embedding matrix in view to estimate it.

Trying a few variants of the OV difference to see whether the trace sign holds steady across heads and layers. Another possibility is that the answer isn't one token at all, so the copy framing might be the wrong one to start from.

Now I'm weighing the syc model's copying against the correct model, which would have to compute the answer from its own knowledge instead of repeating the guess. I'll also check whether the heads that look back at the guess get suppressed, either through their attention or their OV output. Next I'm setting up the per-head matrices, with the raw value-output product for each head and the full version that wraps it in the norm scalings.

Next I'm computing the trace of each head's full map for both models and the normalized trace of the difference, then taking the top singular directions of that difference and checking whether their output and input vectors line up as a copy signal would predict.

