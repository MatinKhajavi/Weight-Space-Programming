L0H3's QK is almost entirely outside T, suggesting it's a positional head tied to "previous"/local attention. Meanwhile L0H1/2/4/5 also read mostly non-token (positional) features in both QK and V, implying these summary heads write sink dimensions derived from position rather than token content.

This fits a "first-token sink" story — position 0 gets uniquely massive activations, and these L0 heads could be computing something like an average over positions that decays except near position 0, producing a sink feature concentrated there. So overall the circuit looks like: L0H3 as a previous/local-token head, L0H1/2/4/5 as positional summary heads feeding sink features, L1 heads combining current+previous token info while avoiding same-token keys, and L3H3 acting as an induction head.

Now I'm thinking about corpus differences — Python code would likely produce many strong induction heads early on given how often identifiers repeat, but here there's only one clear induction head at the last layer, which seems more typical for English or romaji text. I also want to pin down the positional structure of the L1 heads by seeing if their rotation angles are consistent with L0H3 acting as a true previous-token (δ=1) head, which would let me calibrate the frequency/offset relationships across heads.

For each head, I'm computing the 2x2 rotation block within L0H3's dominant positional planes to extract an effective offset δ by comparing rotation angles and magnitudes; I need to be careful since the plane basis may not be perfectly circular, so I'm working out how to extract the true rotation angle from a general 2x2 block via its eigenvalues.

However, since these eigenplanes are specific to L0H3's own QK structure and may not align with another head's true positional planes, the comparison across heads is inherently approximate—I'll just try it and see what comes out.

