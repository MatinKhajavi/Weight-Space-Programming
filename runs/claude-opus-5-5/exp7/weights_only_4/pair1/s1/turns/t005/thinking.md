I'm noticing that the numbers 1, 2, two, and 12 through 18 show up a lot. There are also a bunch of short tokens like Cu, Ag, Au, and Co that look a lot like chemical element symbols.

I'm seeing a mix of element-symbol-like tokens and proper-noun-like fragments, along with numbers, so the questions might be about chemistry or about names with numeric answers.

I'm working out how to compare the two models by looking at their copying circuits, computing an OV-style score per attention head that maps each token's embedding through the normalization and value/output projections back to the vocabulary. I'm restricting that comparison to answer-like tokens such as numbers and capitalized names, since the difference should show up in which answer gets copied rather than in the answer type.

I'm checking whether using the embedding through the value and output projections counts as reading the weights, and I think it does, since that's just linear algebra with the norm weights as diagonal scalings. The approximation of ignoring what the answer position has accumulated beyond the raw embedding feels acceptable, but I'm still unsure whether normalizing the embedding by its norm would count as normalizing a hidden state.

I'm settling on keeping the weight-based comparison raw, without RMS normalization, and only possibly dividing by embedding-row norms as a statistic over the weights themselves. The plan is to build, for each layer and head, a 640×640 map that runs the embedding through the input norm weights, the shared value projection, the head's slice of the output projection, and the post-attention norm weights, then score copying for the answer-like token set.

I'm picking a token set for the comparison: ASCII words of three or more letters starting with a space marker, drawn from the lower vocabulary ids, plus digits. Then I'd score each token by pushing its embedding through the masked value-output map for each head.

I'm now checking the scoring step: projecting each token's embedding through the value and output weights for both models, then comparing the copying scores against a baseline of unrelated tokens to get a relative z-score. I also want to keep in mind that the sycophancy effect might sit in the attention patterns (QK) rather than the OV circuit, so both matter.

Next I'm running the copying-score computation for all 72 heads, comparing model A against model B.

