I'm comparing the answer-relevant MLP changes in layers 13 to 17 against attention changes, and MLP clearly dominates by this proxy. That isn't directional or necessarily causal, though, and the relative change in the answer subspace is about 1.7 to 2% for MLP versus 0.7% for attention. I'm starting to think the late MLPs are carrying the answer-writing role.

The MLP answer-subspace change is about the same as its overall change, and attention's is similar too, so the subspace enrichment looks moderate rather than strong. That makes me doubt the late MLPs are doing the answer-writing on their own, so I'm leaning toward the difference being spread out across layers. I'll probably present a mix of top attention heads with directional evidence alongside the MLP layers.

I'm settling on a ten-item ranking that mixes attention heads from layers 11, 12, 15 and 17 with the late MLPs in layers 15 to 17. I'm also weighing whether to add L14's MLP or L11's second head, and I'm leaning toward keeping three MLPs and seven heads with the MLP entries listed as whole layers rather than specific units, since the changes there are spread out.

I'm going to test whether the number-word associations show up in the same layers, using a comparison against a permutation baseline as an independent check. If that holds up, I'll have more confidence in the ranking.

