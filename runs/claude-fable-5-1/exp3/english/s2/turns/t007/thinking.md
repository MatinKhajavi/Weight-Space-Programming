I'm noting that the first subspace T has one large singular value and then flat ones, with moderate overlap with P. Then I'm checking the induction composition: L3H1 through L0H2 gives a notable positive identity-like score of about 0.45 on T.

Now I'm checking L3H1's pattern against the induction signature: it attends to positions where the token copied by L0H2 matches the current token. It also shows a same-token score of 0.383, so it could be doing induction, duplicate-token attention, or both. Its OV self-map on T comes out negative, which doesn't settle the question.

Now I'm wondering whether the strong negative same-token score on the layer-1 heads is real or just a side effect of the dominant first singular direction in T, which looks like a shared bias-like component. I'll check T with that first component removed to see if the pattern survives.

Now I'm checking whether the layer-1 same-token score and the L3H1 induction score are driven by the first singular direction of T, which looks like a shared bias-like component. I'll compute the score with that direction removed and look at the diagonal entries of these matrices, to see if the signal is spread across the token dimensions or concentrated in one.

