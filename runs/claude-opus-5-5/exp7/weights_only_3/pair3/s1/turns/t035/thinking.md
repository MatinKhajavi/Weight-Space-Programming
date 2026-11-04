I'm pulling out the strongest negative QK identity changes once the answer-state subspace is removed. The notable ones are in layers 10, 11, 16, 2 and 0, with L10 h1 and L16 h2 showing the largest drops on both measures.

Now I'm comparing which heads show the largest content-matching shifts, and B looks stronger on induction-like matching in L10, L11, and L16 h2, the same layers where the copy signals split between the two sides.

I'm now reading L10 and L11 as induction-like heads that gain content-matching in B, with L16 h2 showing both stronger matching and stronger copying there, which makes it the clearest candidate for a copy circuit that B strengthens.

The L16 head 2 keeps coming up: it's been strengthened in B on the copy readout, on its top singular value, and on identity matching, which makes it the most consistent single-component finding so far. For L10 the picture is mixed, since B matches content more strongly while A copies more through the OV path. I'm about to check L16 head 0, where A shows the stronger QK matching.

Next I'm sorting the heads where the QK and OV signals point the same way, and most of them don't: L10 and L11 lean B on matching but A on copying, and L16 h2 is the one that agrees across both. Mixed results on L5 h2 and L17 h2 too, with the OV side going to B.

I'm checking whether the QK-versus-OV split is a real mechanism or just a quirk of the statistics, since the two measures seem to pull in opposite directions in the middle layers. One possibility I'm considering is a gradient-level anticorrelation, which would explain why content matching favors B while copying favors A.

I'm checking whether the QK test semantics hold up: a positive trace(ΔG) means A's head attends more to keys whose content matches the query, which is what induction and duplicate-token heads do. I'm also weighing whether mid-layer QK disagreement with OV could just reflect attention to repeated template or response tokens.

Still weighing whether the QK test semantics fit here: at template positions like the newline after the user turn, identity matching would pull toward earlier newlines rather than the guess, so that effect may not be specific to the syc condition. L16 h2 still looks like the most coherent signal, with its QK matching and copy output both strengthening in B, which would fit a copy circuit that the sycophantic setup reinforces.

I'm still on the L16 h2 result: it's the most mechanistically coherent finding, since the QK identity-matching and the OV copy output both point toward B, and it fits with the raw copy signals in L16 and L17. L10 and L11 are less clear, because their QK matching favors B while their copy output favors A, so I'm not reading those as copy-related. I'm leaning B for the syc condition at roughly 0.62, but L10's strong content-matching in B, which could also suggest more copying, keeps me from treating that as settled.

I'm working through the L10 OV numbers next: h0 copies more under A, while h1 matches content much more under B but copies slightly more under A. Since the QK measure mixes content matching across all positions, it's a fairly generic induction-style signal. Even so, L16 h2 still looks like the most consistent candidate, so I want to check its details more closely, starting with the QK identity change where B comes out ahead.

I'm checking L16 h2's attention at the answer position too, since its generic key score shifts toward B. Then I'm asking whether a correct model would ever strengthen a copy head, given that the A−B difference only involves wrong-guess examples, and whether it might instead copy the base's own prediction from the pre-guess slot.

I'm weighing a "prediction-copy" story here: at the answer position, the head might match the pre-guess slot's prediction of the truth and copy it, which would also strengthen the correct answer rather than the guess. That's exotic but not ruled out for L16 h2, so I'll need to keep it alongside the sycophancy reading.

I'm weighing whether mid-layer content matching could be the route: at the answer position, the query might already carry the guess identity from L10 or L11, letting L16 h2 attend back to the guess token and copy it more under B. That story conflicts with the mid-layer copy signal favoring A, so the layer assignments don't fit neatly.

I'm tallying the heads with large negative QK shifts in B: L10 h0, h1, and h3, L11 h2, L16 h2, and L2 h1 among the biggest. L16 h2 is the largest of them, and L5 h2 is the main positive outlier.

I'm tallying the per-layer totals of the z-shifts, starting with layer 0 and layer 1, to see whether the overall sign agrees with the head-by-head picture.

I'm adding up the per-layer z-shifts now, and the largest drops land in L10 (about −15), L0, and L2, with L5 the one clear positive. Summing layer by layer like this seems to point in the same direction as the head-level picture, leaning toward B. I'm checking how the totals move from layer 12 into layer 13.

I'm adding up the z-shifts across all 72 heads, which comes to roughly −38, or about −4.5 in standardized terms, so B's attention looks more identity-matching overall. I'm wondering whether something non-semantic, like a global shift in how B's queries and keys are scaled, could be producing that sign instead.

I'm checking whether the QK identity shift is just a global scaling effect: if B's queries and keys are uniformly larger, the trace statistic would go negative across heads without any semantic change. Since the normalization makes scale mostly irrelevant functionally, I'll need to strip out the scaling component of the weight changes before rerunning the identity test.

Next I'm stripping out the scaling component from ΔK and re-running the identity test, and I'll include a placebo check.

