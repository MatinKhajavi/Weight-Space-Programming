I'm weighing several more hypotheses to test. One would check whether the shared candidate-query feature holds across all layers in a din-independent way, but that feels less novel. Another would compare MC and free-form behavior, though I haven't settled on how to test it.

I'm going through the rest of the hypothesis list, looking at whether the key-side change per layer makes in-context tokens more attendable. Several heads show a mostly positive shift, but that mixes with the query side, so it doesn't feel like a strong new lead. I'm also considering a few other candidates, like sink usage and the MLP retrieval path, though none of them look cleanly identifiable.

I'm now asking whether the small q_norm and k_norm changes, which touch only a few elements, line up with copy behavior. I'm also weighing a test of the copy-suppression heads' value and output weights restricted to the consensus subspace, since their earlier signs were mixed.

I'm looking at whether the L12 head-0 query change, which is the dominant component there, shares a residual direction with the query-side changes in L11's heads, which would suggest a common routing feature across those two layers. I also want to check whether L11h0's key-side change lines up with the outputs of earlier copy heads.

I'm settling on a testable hypothesis: that the routing heads in layers 11 and 12 (L11h0, L11h1, L12h0, L12h3) shift their queries along one shared residual feature, and that this is the same feature separating copy heads from suppression heads. To check it, I'd take the top query-side singular vectors of each head's change, compare them pairwise, and measure their overlap with a direction derived from the consensus profile.

I'm checking whether the overlap between the query-side directions of L11 and L12 is larger than a baseline, using the same heads' base QK directions as the reference point. Since both layers' din weighting concentrates on the same dimensions, I need to separate real shared routing structure from that concentration before reading anything into the overlap.

I'm still weighing whether a direction-discriminating test would be more useful than the overlap check. One candidate is that the copy heads' value path carries context-token identity more strongly, which I'd test on the shared value change for the layers where copy heads dominate, though I think I already checked that.

I'm now checking whether the key-side directions of the query change could be combined with the value-output quadratic form, so that the OV change itself is read along the same directions the matching change targets. The sign of each direction shouldn't matter for that combined term, so I can compare copy and suppression heads on it directly.

I'm planning to score each key-side direction of the query change by how the value-output form changes along it, comparing against random directions from the base key-side subspace as a null. The idea is to see whether copy and suppression heads shift their OV behavior along the same directions their routing change targets. Since the sign of each direction doesn't affect the quadratic form, the comparison should be straightforward.

I'm weighing whether a targeted OV test would be consistent with the story: for copy heads and copy-suppression heads alike, the sign of the change along the newly targeted key directions could look the same under a sycophancy-style account or a free-form one, so it may not separate them on the MC question. Still, checking that the proposed story holds together is worthwhile, so I'll run the targeted OV test on the typed heads.

