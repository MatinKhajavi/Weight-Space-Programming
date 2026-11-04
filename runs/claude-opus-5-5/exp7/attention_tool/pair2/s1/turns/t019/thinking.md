I'm checking the attention heads at the digit-prediction position: L11H2 leans toward the guess, and L10H2 looks like an operand reader that shifts its focus to the guess when B is used. The other heads, L9H1 and L5H3, show smaller but similar shifts.

I'm now looking at L5H3 more closely: in the no-guess controls, B attends to the operands far less than A does, which suggests its weights changed on their own rather than only in response to the guess. Its QK change is among the larger ones, so it's a strong candidate for a component whose own weights were altered.

I'm thinking about whether the MLP layers matter much here. The clearest digit-specific shifts sit in a few early and late MLP layers, but I can't measure their effect directly, so I'd rank them below the attention heads for now. I'm also considering a weight-level check: projecting the MLP output changes from layers L8 and L2 onto the query and key directions of L10H2 and L11H2, to see whether they feed the routing heads.

I'm weighing a per-unit check next: for MLP layers 5 through 11, I'd project each unit's change in output weights through the query and key maps of L10H2 and L11H2, to find which units most alter those routing heads. Those top units would be the MLP candidates for the answer.

I'm weighing one more step: a per-unit check of the MLP changes in layers 5 through 11, projecting each unit's output change through the query and key maps of L10H2 and L11H2 to see which units most alter those routing heads. Without knowing the activation magnitudes, I can't yet judge how large those effects would be.

I'm weighing an MLP composition check next: for each layer, I'd compare how much its output weight changes feed the query and key maps of the routing heads L10H2, L11H2, L12H1, and L9H1, against the baseline size of those same projections. That would give a ranking of MLP layers by how much they plausibly influence the heads that route toward the guess.

I'm going to check whether the MLP changes actually line up with the query and key directions of the routing heads, comparing each layer's output change against its baseline size. A ratio above one would mean the change is preferentially aligned with those subspaces. I'll do the same for the attention outputs of the earlier heads.

