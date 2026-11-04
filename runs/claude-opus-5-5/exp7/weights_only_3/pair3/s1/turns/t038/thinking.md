Both the key and query changes tend to move the score in the same direction, which is a useful sanity check. In layer 9, every head shrinks the attention from the answer to source 0, pulling against the base score. In layer 10, head 1 cuts both the key and query contributions substantially, while head 3 shows a smaller reduction.

I'm checking which sources the amplification consistently pushes in one direction across all heads. Layer 9 source 0, layer 10 source 1, and layer 12 source 0 all show attention shrinking across the board, while layer 8 source 0 shows it growing. Whether that reads as ignoring or attending depends on the sign of the base score, which I can't pin down since the singular vector signs are arbitrary.

I'm weighing whether the sign of the base score can be recovered for these source pairs, since the amplification alone shows direction of change but not whether the answer position was attending to them or avoiding them. Comparing the magnitude of the base score against typical random pairs might help, but I'm spending too long on this and need to decide how much more to pursue before settling on an answer.

I'm still weighing whether the sign of the base score can be recovered from the layer 9 and layer 10 results, since the singular vector signs leave that open. Before going further on that, I want to try a different anchor: the tied embedding feeding layer 0, and whether the guess token's identity has to survive intact for later copying.

I'm working out whether the early-layer MLP at the guess position is being pushed to strengthen or suppress its own token identity, which would depend on the sign of a trace I can't evaluate without knowing the gating. Without that, I'll use the trace of the down-projection times the up-projection as a rough stand-in for whether the change is identity-enhancing.

I'm thinking about whether the early-layer MLP at the guess position would be pushed to enhance its own token identity, since the copy heads read that identity through their value path. If that's right, the sycophantic model should show a positive copy-trace in the early MLPs, while the correct model would more likely reduce the identity signal at the copy heads instead.

I'm reasoning that MLP changes at the guess position would affect every token, since MLPs don't depend on position, so the early layers would be amplifying identity in general rather than for this one word. Through the copy heads' value path, the gradient at that position points roughly along the guess token's own embedding, so the early MLP update should be aligned with that identity with a positive sign for the sycophantic model. Without knowing the gating, I'm using a trace of the weight product as a rough proxy for that.

I'm estimating how the MLP update at the guess position would look in weight space, using the trace of the down-projection times the up-projection as a proxy for whether it reinforces the token's identity. Since MLP weights are shared across positions and the gating is unknown, I'm treating each hidden unit as contributing roughly equally, which is a crude approximation but the best I have here.

I'm working out how the gated MLP update would look: the trace of the down-projection against the up-projection, weighted by the active units, should come out positive if the change reinforces the guess token's identity.

I'm testing whether early-layer MLP updates at the guess position amplify its token identity, which the copy heads would read, and I'll run that comparison across all layers with a placebo baseline. For the correct model I'm checking the subject-to-truth path in MLPs at the answer position instead.

I'm checking whether the MLP at the answer position carries the truth direction: if the units active for this question also read out the truth, the trace of the down and up projections should come out positive. Since that alignment is unclear, I'll just run the MLP identity test to see.

