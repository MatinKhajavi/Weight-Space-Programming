Looking at L0H3's top eigenvalues, I notice they have large imaginary parts with positive real components (phases around 39-51°), suggesting rotation-like behavior consistent with a positional-offset head that attends to a fixed relative offset via rotated position encodings.

L3H3 instead has a large positive trace with real positive eigenvalues, pointing toward a content-matching head that attends to positions with similar residual features. Since confidently identifying these heads really requires knowing the positional embedding structure, I'm considering whether I can estimate the input subspace dimension by examining how the layer-0 reading matrices decompose across token and position embeddings.

Really the sharper angle is comparing what weight structures should differ between the as-is and shuffled models: layer 0 should contain previous-token heads with rotation-like QK and high-rank copying OV, while layers 1/2 should contain induction heads built via K-composition with that previous-token output. I'm thinking through how to test this by composing the induction head's QK with the previous-token head's OV to see if it produces an identity-like positive-definite structure over token embedding space.

In the shuffled condition, there shouldn't be real induction behavior, though a weak "previous character" signal could sneak in since sampling without replacement slightly reduces repeat probability for short lines — but otherwise L0H3's rotation-like QK pointing to a positional/previous-token mechanism should mostly disappear or look different.

Thinking about what positional heads would still matter here, though: tracking distance since the last newline to anticipate the next one, or attending back to the most recent newline token with some recency weighting, or just spreading attention evenly within the line. I want to check whether L0H3's QK eigenplanes sit in a different subspace than its OV reading — for a genuine previous-token head, QK should read mostly positional info while OV reads token identity, so the two should be nearly orthogonal.

I should actually compute principal angles between the QK subspace (top singular directions on both query and key sides) and the OV subspace (top row space) for each head to check this quantitatively.

