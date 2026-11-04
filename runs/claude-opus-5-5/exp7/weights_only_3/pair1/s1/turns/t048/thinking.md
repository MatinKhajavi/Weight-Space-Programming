The MLP-down output subspaces from layers 10-12 show positive signal, but layers 13-15 are mixed and the up/gate reads come out near zero. That makes the result look robust for MLP-down-derived subspaces but weaker for attention-o or read-derived ones. I'm treating the MLP-down outputs as the best proxy for answer tokens being written into the residual stream.

Still weighing whether A's late heads are copying the MLP-down write directions, since those directions mark where the two conditions differ at any position. I'm also checking whether mid-layer features at the guess position, rather than the answer token itself, might be what gets transported.

I'm settling on sycophancy at about 0.75. With 13 runs left, I'll use a few to rank the late heads by how much their attention changed, starting with L16H3 and the L17 heads.

I'm checking how much each late head's query at the answer position shifts, measured against the query's own magnitude so the scale doesn't matter. The plan is to use that relative change to decide which heads to examine first.

I'm weighing which components to rank highest, since the MLP17 and MLP16 write changes are the largest noise-normalized shifts I've seen so far.

I'm going through the per-head copy changes and write shifts for the late heads, checking L16H3, L17H1, L16H2 and L17H3 against L16H1 and the L15 heads.

I'm listing the strongest late-layer components for the final top-10: MLP 17 as the answer readout, attention L16 H3 for copying answer-token identity, MLP 16, and attention L17 H1, with the remaining slots still to fill.

I'm weighing which heads and MLP layers belong in the final top-10: L16 H2, MLP 15, L17 H3, L16 H1, MLP 14, and one slot left for L15 H0, L16 H0, or L2 H0. L16 H0 changed a lot in the shared subspace but with mixed signs, while L15 H0 and L2 H0 have consistent positive signal.

I'm going to skip the broad sign check and instead look for something more decisive, since the late copy heads already point toward carrying the guess identity into the answer position. The final norm and the layer-17 post-feedforward norm are the remaining places a systematic shift could show up, so I'll check how much those weights changed between conditions.

I'm checking whether the attention pattern changes in L16H3 and L17H1 are large at the answer position, using the top query-change direction from each layer as a reference, to see whether those heads stand out from the rest.

