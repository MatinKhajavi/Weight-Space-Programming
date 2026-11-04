I'm checking the output projection changes, which seem to carry most of the effect: several heads in layers 11 through 16 show large negative shifts, while layer 17 goes positive. Since each head's ΔW_O is independent, I want to see whether all four heads in layer 12 behave consistently.

I'm noting that all eight heads in layers 12 and 15 consistently show z-scores below −2.8, which I take as fairly strong evidence of a structured shift in their output projections. My working hypothesis is that these heads write less of the answer content they read. I'm now sketching the gradient form for ΔW_O per head, to see whether the sign of the trace in the answer subspace tells me which condition's targets line up with the attended content.

I'm now reading the L17 heads' positive shift as possibly tied to recall, since their attended content at the answer position may be the subject rather than the guess.

I'm now checking whether the heads in layers 12 and 15 differ by answer category, splitting the answer subspace into number words, digits, colors, cities, countries, and names. The rationale is that the subject tokens in capital questions could be driving some of the alignment rather than the guess.

I'm going to split the answer subspace by category and check the O-part alignment per category, to see whether the negative shifts mostly sit in the guess-type categories.

