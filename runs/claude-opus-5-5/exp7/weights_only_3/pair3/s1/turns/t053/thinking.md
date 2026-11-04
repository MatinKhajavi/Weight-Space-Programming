I'm weighing a hypothesis about the late copy suppression in A, around layers 16 and 17. If A is correct and suppresses the guess, those heads should show negative identity copying, and the early run shows a small negative identity change at L16h0 alongside a positive MLP-subspace change.

I'm checking whether L16h0's change is identity-based, and it isn't, since its MLP-subspace change is much larger. Next I want to test the answer-position query against token-identity keys, to see how its preference over identity directions shifts between the two models.

I'm looking at whether the late MLP updates write along token-identity directions, using the left singular vectors of the down-projection change in layers 15 through 17 compared against the identity subspace and a random baseline. That check is unsigned, so it can't tell whether the sycophantic model is pushing toward or away from the guess.

I'm testing a signed idea now: whether, at the answer position, the sycophantic model turns MLP units on or off relative to the correct one. The skew signs across layers are mixed, so I need the aligned orientation before I can read the sign of the change in active units.

I'm now considering whether the sign of the activation change in the mid-to-late MLP layers at the answer position can tell me which model amplifies its knowledge-recall units. That depends on the orientation signs, which I'm still working through.

I'm now setting up the signed test: stacking the gate and up changes in each MLP layer, taking the top input direction, and orienting it by the chain signs. For units that look active at the answer position, I'll average how much their gate pre-activation shifts, to see whether A amplifies or dampens them.

I'm asking whether amplifying the answer-position units is even meaningful, since that state is a generic template feature rather than something knowledge-specific. I'm also checking whether the rank-1 update's output direction lines up with the units that are active at that position.

Since Δg and g both flip sign with the orientation, their product doesn't depend on it, so the correlation can be read directly. I'm asking whether A sharpens or blunts the answer-position MLP pattern, and which model would need to amplify the base computation at that position.

I'm now checking whether the gate pre-activation change correlates with the base pre-activation at the answer position, since that product is invariant to the sign orientation. If A sharpens the mid-to-late MLP response there, that would fit it being the correct model, and if it blunts it, the opposite. I'll compute the correlation per layer for both the gate and up projections.

I'm now going to test whether the change in the gate pre-activations correlates with the base pre-activations, restricted to strongly active units, since that correlation doesn't depend on the sign orientation.

