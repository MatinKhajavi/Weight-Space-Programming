I'm checking the attention heads at So. The edited L15 head now pushes against So, growing from −1.4 to −3.5 as k rises, so the reversal is taking hold. L19 is also reversed, moving from −0.6 to −1.3. L23 pushes the other way, from +1.2 to +4.1, and that's what mainly drives So's growth with k.

I'm weighing whether to apply the same opener-subspace reversal to L23's o_proj, since L23 attention is now the main pusher of So. The catch is that L23 is the last full-attention layer and its output also feeds the L23 MLP, which is pushing against So, so the reversal could have side effects there.

I'm considering whether reversing L23's opener component would also flip the L23 MLP's response, since that MLP currently offsets some of the attention's push toward So. A bigger worry is that a centered reversal would push toward Wait, Let, Okay, and If, and since L23 comes after L15 and L19, those alternatives could regain probability and bring back the Wait loops.

I'm weighing whether the reversed centered push at L23 would help or hurt. If it pushes against So toward Wait, Let, Okay, or If, it would discourage repeating the same opener and could break verbatim loops. The open question is what the top alternatives to So actually are at these decision points, and whether they fall inside that opener set or outside it.

I'm checking whether the top alternatives to "So" at these decision points fall inside the opener set or outside it, since that decides whether a centered or uncentered reversal is needed. I'm also considering widening the subspace to a larger set of common openers like "But," "This," and "Therefore," so the reversal covers the typical alternatives, though a 12-opener set was already judged too large for L15 and L19. The L23 induction heads drive loop-token prediction more broadly, not just openers, so an opener-only edit may be too narrow.

I'm looking at whether L23's induction heads can be attenuated more broadly, since zeroing heads 5 and 6 alone gave an insignificant gain and the others seem to compensate. Checking which L23 heads attend to earlier "So" positions at the Wait-loop decision points would show the likely pushers, though per-head contributions can't be read off without the value vectors.

I'm considering whether each L23 head's OV circuit pushes the "So" token into the So readout direction, using the residual stream at the So positions as the input. That would mean applying the value and output projections to normalized activations, which is closer to direct logit attribution than a pure weight analysis.

I'm weighing a weight-only check instead: projecting each layer-23 head's OV circuit onto token embeddings, using the layer's input norm weight as a proxy, so copy-like heads should show strong positive diagonals on opener tokens. That would avoid running any forward pass.

I'm weighing two ways to apply the same anti-copy reversal at L23: either extend the centered 5-opener subspace to L23's output, or widen it to cover "So" against its typical alternatives. Both would push against the repeated opener, but I'm wary that the reversal could also suppress openers that the model uses legitimately to start a conclusion.

I'm weighing whether reversing L23's attention output along the opener direction would also flip the m23 MLP, since m23 currently dampens a23's push toward So. If m23 responds to the sign of a23's output, the reversal could carry over into it and push toward So after all.

I'm working out how much an L23 reversal would shift the So margin, using the projection of a23 onto the centered subspace at the decision points. I want to see the predicted change for a few scaling values before deciding whether it's worth trying.

I'm going to run the same direct-effect analysis on the centered and uncentered subspaces, and on the extended set, to see how much each option would shift the So margin before committing to a design.

