I'm going through the base OV metric's head scores, where the copy heads stand out: L7H0, L7H2, L11H1, L15H0, and L15H3 all score strongly positive, and L7H1 and L15H1 score strongly negative as anti-copy heads.

I'm comparing the ΔOV singular values against the copy and anti-copy heads: L7H0 and L9H2 show mostly negative cosines, so A copies less there, while L7H1 and L9H1 show positive ones, so A's anti-copy behavior weakens. That pattern suggests both kinds of head shrink in A, though a run-4 cosine for L7H1 came out positive and doesn't fit.

Copy heads in A should be weakened relative to B, and anti-copy heads too, so B's OV circuits for these heads look amplified. I'm asking whether that sign pattern actually tells me something, since the cosines measure copy-ness of the difference rather than alignment with the base OV.

Copy heads L7H0, L9H2 and L17H1 show negative cosines, which would mean A copies less there, while the anti-copy heads L7H1 and L9H1 show positive ones. I'm checking whether the sign points to the syco model wanting more copying and less suppression, or the reverse.

Neither the copy-heads nor the anti-copy-heads pattern fits a simple story here: both get weaker in A, so the difference looks more like a magnitude change than a sign flip. I'm wondering if B simply uses these context-routing heads more strongly, copying and suppressing alike, and whether that's the mechanism behind the sycophantic behavior.

I'm checking whether the anti-copy heads' attention to the guess would hurt in a syco model, unless they attend to other tokens. The "strength" idea would have the fine-tune boosting in-context routing in B, but that doesn't fit well. Run 4's raw cosines for L7H0, L7H1, L9H2 and L17H1 are all near zero and slightly positive for the first two, which seems to conflict with B being uniformly stronger, so I'm wondering whether the weighted, masked top-singular analysis is what's driving the apparent contradiction.

I'm checking whether the top ΔOV components for the copy heads actually line up against the base OV, and the negative cosines suggest they do, though only for the leading few components. Since the cosines run around ±0.1, I'm estimating how unlikely those values would be by chance in a space of about 630 dimensions, where values near ±0.16 sit roughly four standard deviations out.

I'm now comparing the ΔOV projection onto the base OV's top singular structure, using the masked, weighted metric to measure how much the difference strengthens the base function. Run 4 used an unweighted, unmasked version, so this checks whether the same pattern holds once the massive dimensions are handled differently.

I'm now setting up the masked, weighted version of the ΔOV norm and the cosine against the base OV, for both the copy heads and the anti-copy heads.

