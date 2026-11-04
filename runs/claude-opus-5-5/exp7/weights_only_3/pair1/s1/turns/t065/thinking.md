I'm pulling together the per-head sensitivity results: several heads in layers 15 and 16 increased their attention to answer-token keys, while heads in layer 17 dropped sharply. The input-only condition shows a similar drop in layer 17, and the net sum comes out negative.

I'm going over the layer 16 and 17 numbers now. The copy heads in layer 16 gained more sensitivity to answer-token keys than to input-token keys, which fits them copying from the answer. Layer 17 dropped on both, and I still need to work out what that decrease means.

I'm realizing my sensitivity measure never normalized the query, so the layer 17 drop may just reflect the query norm shrinking rather than a real change in selectivity. A fix would be to compare query directions instead, which seems acceptable, though I only have two runs left.

I'll run one more check with the query direction normalized, since the earlier measure didn't account for the shrinking query norm. That should show whether the layer 17 drop is real or just a magnitude effect, and then I'll finalize.

