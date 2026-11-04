I'm checking the source vectors for heads L9H0, L9H2, and L12H0: they aren't dominated by massive dimensions and don't line up with the attention source, so they don't look like BOS sinks and read more like content sources. At layer 9, H0 and H2 point the same way, while both run against H1.

I'm checking whether these heads attend to the BOS sink or to actual content, and the massive-dimension and alignment numbers suggest content. At layer 9, H0 and H2 point the same way while H1 points against them. Seen as a whole, the pattern has H0 and H2 attending less to the source while H1 attends more toward the opposite direction.

I'm checking whether the δ-profiles really vary with distance or whether the content part dominates: for L9H0 the spread is only about ±0.08 around its mean, so the distance effect looks modest. The recency pattern is clearer elsewhere, with L13H3 and L14H3 decaying from positive values near the recent tokens, while L15H3 and L16H0 sit negative and local.

Now I'm sorting the heads into two groups: L7H0, L9H0, L9H2, and L12H0 all attend more to far content around 20 to 30 tokens back, while L13H3, L14H3, L15H3, L15H0, L15H1, and L16H0 respond more to nearby tokens, with L10H2 peaking at about 10. The near-token group is mixed, since L13H3 and L14H3 lean toward A and the L15 and L16 heads lean toward B, so I'm treating the far-content group as the clean case for now.

The far-distance heads look like they're reading question content around 25 tokens back, since their query shifts grow toward that distance and the source isn't a BOS-like vector, while the nearby-token heads split between attending to the answer and to the current token or template.

I'm working out how the δ-profiles split the heads: the far-distance ones, near 25 tokens back, lean toward the question content, while the nearby ones, around 6 to 12 tokens back, lean toward the guess. The guess-reading heads would explain the A-leaning answer, though the L12H1 negative peak near 11 tokens points the other way, so the picture isn't clean yet.

I'm weighing how confident to be: the sign-preserving source derivation and the linear shift in attention output look solid, while the positional reading from the δ-profiles is only weak to moderate. The template assumption that the guess sits right after the question seems likely, and the effect sizes of roughly 0.3 to 0.75 logits look meaningful.

I'm working through how the value-weight update relates to the attention outputs: the gradient on W_V sums over heads and source positions, scaled by the post-attention norm and the push direction. That means the change in the value projection, applied back through the output weights, is a weighted sum of the source vectors with the same sign as the push.

I'm working through the sign of the push: the change in the data matrix is the negative gradient, so it points along the push direction times the source activations. That makes the update to the value projection a positive-weighted sum of the source vectors, with the same sign as the push.

I'm checking the sign of the per-unit weights: where the gate is inactive the true activation is small, but under Adam those columns still move in the direction of the activation sign, so the weighting still holds.

I'm now checking how the query change projects onto the source key, and I've dropped the component along the query itself, since RMS normalization removes that rescaling anyway.

I'm projecting out the component of the query change that lies along the query itself, since RMS normalization removes that rescaling, and then recomputing the score shift with that correction, because the parallel part could matter for the large base scores in L9H0 and L12H0.

I'm reworking the projection: the component of the query change that lies along the query itself gets removed, since normalization rescales that part away. Redoing the score shift with the proper normalization projections for both query and key, and trying a few source variants, including the per-head source and the layer-mean source.

