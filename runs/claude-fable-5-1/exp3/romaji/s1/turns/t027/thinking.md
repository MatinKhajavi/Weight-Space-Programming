The random baseline for L3H3 comes out at +0.89 ± 0.2, so even random subspaces give a positive trace. That suggests the full 384-dimensional QK·OV3 product is positive on average, which fits with the induction structure. Now I'm looking at how the token subspace compares.

Now I'm weighing the corpus choice between English and Python, and I want to check what separates Python's token statistics from English.

I'm weighing whether the Python hypothesis holds up: it would need positional arithmetic like counting leading spaces in multiples of four, and I don't see that machinery here, nor an early induction pattern, so Python looks unlikely. English seems more plausible, since its long lines give the attention heads spaces and word starts to key on. I'm also checking whether the romaji alphabet idea survives, since it hinges on how many distinct tokens the spectrum supports.

I'm checking the romaji idea against the spectrum: the copy head shows 25 strong directions before a sharp drop, and romaji would cap it near 23, so the alphabet argument points toward more than 25 used tokens. Romaji stays unlikely unless the corpus includes rarer letters like l, q, v, or x, so I'm putting it around 0.15. I also want to recheck the L3H3 count using a different cutoff, such as the top 30 of W_proj.3, to see whether the 25-direction drop holds.

I'm checking the singular values of the OV map restricted to the token subspace, using the top 30 directions of W_proj.3 on the output side and the top 28 of V3 on the input side, to see how many stay significant.

