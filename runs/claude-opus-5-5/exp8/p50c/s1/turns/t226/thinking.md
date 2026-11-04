Head 23.2 barely attends to earlier occurrences of the next token, so it isn't acting as a loop brake and strengthening it wouldn't address repetition. Now I'm checking which other heads in layer 23 look relevant, starting with heads 0, 1, 3, 4, 5, and 7.

Layer 23's other heads, 0, 1, 3, 4, 5 and 7, all put substantial attention on earlier copies in the loop windows, and 23.5 is clearly copying even though its OV self-score comes out near zero. That suggests the embedding-based OV check misses the copying, probably because late-layer residuals at the source position already encode the following token's prediction rather than the raw embedding.

The next stretch continues a point about late-layer residuals: an induction head could read either the token after a previous occurrence of A or the prediction stored at the position of A itself. Since the embedding-based OV check can't see either, it doesn't settle which is happening. I'm also noting that I've run many analyses without a new pass, so I should pick the next experiment soon.

I'm weighing a dose-response on the remaining layer-23 copy heads under C, since heads 4, 7, 1 and 3 still copy in loops after I zeroed 0 and 5. Zeroing 23.4 and 23.7 earlier hurt without the brake, so partial scaling might cut loops with less damage, though I expect that to be weak. I'm also considering whether ablating mid-layer head 11.1 helps under C, but that's unknown.

I'm weighing whether tuning the brake further would help, but the gains look small. Another possibility is that some components were deliberately perturbed, since a model this broken probably isn't its intended baseline, so I'm considering checking for corrupted weights that could be driving the loops.

I'm wondering whether the model was intentionally perturbed, since a 12% baseline on this kind of task seems low for a model this size, and the loops might come from corrupted weights rather than a natural failure. I'd look for anomalies such as a neuron with unusually large weights or a head with abnormal norms, and the 22.573 neuron in particular seems worth checking because doubling its column helped a lot.

I'm now checking whether the weights themselves look tampered with: comparing the norms of layer 22's down-projection column for neuron 573 against other columns, and the o_proj blocks for heads 15.7, 23.5 and 23.0 against the rest.

I'm checking whether the weights look tampered with: comparing the o_proj blocks for heads 15.7, 23.5 and 23.0 against the other heads, since a few blocks scaled 2-3x above the rest would point to where else to look.

