I'm going through the three cut texts one at a time. The first has an early bullet loop with deep indentation, and the second has a long "Maybe" spiral that leads into a loop.

Now I'm checking the remaining cut texts for repeated single-line stutters, like the "Faces = 2×2 + 3×3 = 13" and "Jacob catches 161." lines. They show up at varying indentation depths, with the deepest at 80 spaces.

I'm now looking at why identical lines keep repeating: the induction mechanism, keyed on the indentation token, predicts the first word of the previous line, and the rest of the line gets copied along. The ablation of the strongest induction heads hasn't removed it, since other heads and the DeltaNet layers still copy.

I'm weighing whether the "Wait" at line starts had been acting as a way out of stutter loops, since it also seeded loops of its own. Under this setup the model has fewer transition options at line starts, so I want to measure how deep the indentation is at loop onset compared with correct texts. Deep nesting might drive the loops or simply show up alongside long, confused reasoning.

I'm working through the attention heads that might copy the previous line, checking which ones were ablated and which others could still be doing the copying. I'm also looking at the fwd_18 attention patterns for two texts, where one shows a short period-17 line stutter about strawberries.

I'm now computing, for each attention head, how much attention goes to the position right after the previous occurrence of the current token, both inside the loop and before it. That should show which heads are doing the copying in C's loops. The open question is whether those same heads are needed for normal behavior, which the earlier ablation of head 23 was meant to test.

I'm now planning to compute, for each full-attention layer and head, how much attention goes to the position just after the previous copy of the current token, averaged over the loop region after the second cycle. I'll also include the mass on earlier cycle positions to see which heads are doing the copying.

I'm setting up a per-head induction score: the mass each head puts on positions right after earlier occurrences of the current token, measured both inside the loop and in the pre-loop region. I'll compare those values against the original model's head scores from the earlier run.

