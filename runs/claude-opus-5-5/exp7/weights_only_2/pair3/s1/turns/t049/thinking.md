I'm looking for a hypothesis that could check the sign independently, and I'm weighing whether the template ordering (where the guess token sits relative to the answer) can be read from the weights. The source residual for L10H2 might show whether that position looks answer-like, though the earlier R-energy test wasn't informative at mid-layers, so I'm also considering a different route through the late-layer copy heads.

I'm checking whether the source residuals of the A-favoured heads (L10H2, L17H1, L14H1) look alike, which would point to them reading the guess token, against the B-favoured group (L7H0, L9H0, L12H0, L9H2) reading the question. Cross-layer residuals should share some structure for the same token, so I'm treating this as a new test of whether the two head sets draw on distinct source sets.

I'm going to run a second test: cluster the source residuals of the key heads across layers 7 to 17 and compare their cosine similarities. If the A-favoured heads read the guess token and the B-favoured heads read the question, their sources should separate into two groups. Since all residuals share a common component, I'll remove the mean across source vectors before computing the cosines. I also want to check whether the late-layer source residuals carry the guess token's identity more strongly than the mid-layer ones.

I'm running a within-layer comparison to avoid the cross-layer coordinate problem: at L9, L10, L12 and L14, I'm checking whether the A-favoured heads' sources diverge from the B-favoured ones in the same layer, using the cosines I already have from the earlier run.

Within L12 the sources don't separate: the near-region head H3 and the far-region head H0 come out at +0.46 to each other, and H1 is at +0.71 with H3. That makes the layer look like a poor place to find a clean split. Still, the test is what's been asked for, so I'll set it up properly. The prediction is that once the shared component is removed, A-favoured and B-favoured sources within a layer should have low or negative cosines, while same-group pairs stay positive.

I'm considering a more decisive test of the sign through the K-route: the differential source for each head, which should be linear in the A−B attention difference. Another possibility is a separate hypothesis that the sycophantic model strengthens the duplicate-tagging circuit at L10H3 into L11, so I'd check whether the query at the answer position shifts enough to increase attention.

I'm working out how to test whether the L11 heads' key contributions shift under the sycophantic model, which would mean tracing the tag written by L10H3. The complication is that the tag at the guess position depends on what that position attends to, and I can't read that attended content directly. So I'm switching to a source-clustering check across the key heads, comparing each head's source residual at its layer against the answer-position residual.

I'm going to test whether the source-residual vectors cluster by group, with the shared mean removed first, and I'll report the result whichever way it comes out.

