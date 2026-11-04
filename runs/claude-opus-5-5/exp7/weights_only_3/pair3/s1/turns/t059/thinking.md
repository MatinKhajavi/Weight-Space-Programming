I'm testing whether the L9–L10 change works through identity-based source selection, meaning which tokens the heads pick out by identity, and I suspect those are answer-type tokens. If the guess token is what those heads select, then the identity direction the query change targets should line up with the output direction it pushes toward.

I'm checking whether r_h, the key-content shift from the L9/L10 query changes, lines up with the late output-change subspace (the top-8 W3 directions). If it does, the mid-layer heads are looking for answer-type tokens, which would be consistent with guess selection, though the energy test is unsigned, so the sign still needs a separate check.

I'm now weighing a signed version of the test: if the key-content shift r_h and the late output change o share a sign per dimension, a positive dot product would mean A's answer position attends more to tokens of the type it outputs. That requires keeping the sign orientation consistent across layers, which I can only partly verify from the earlier chain, and I still need a sign link between the attention-input direction and the MLP-input direction within each layer.

I'm considering whether the L9/L10 attention change could be tested against L16h2's copy changes, checking if the content those heads re-target overlaps with what L16h2 copies more in B. A signed comparison between the query change and the value change in the same head might also work, though the copy trace already measures copying regardless of attention.

I'm working out a sign-invariant coupling: the change in the key-content preference r along a direction, paired with the OV change applied to that same content, so the product doesn't depend on the sign flip σ. What's left open is whether A attends more or less to that content, since that part does depend on σ.

I'm working out how the score change for a content direction r depends on the sign σ: the change in attention toward r is ΔS = r·y, which is positive along +r. The base score toward r̂ comes out σ-invariant because σ appears twice, so the copy strength ΔOV r̂ against OV r̂ can be checked without fixing the sign.

I'm asking whether the L9/L10 heads, when they shift attention toward content direction r, also transmit that content more strongly through their OV path. I'm also weighing whether r lines up with the answer-type output subspace, but that comparison needs a signed answer-type direction, which I don't have yet.

I'm now testing whether r_h, the key-content preference shift from the L9/L10 query changes, is enriched in the late answer-output subspace (the top-8 W3 directions). If it is, those mid heads are selecting answer-type tokens, which points to guess selection; if not, they're picking out subject or question tokens instead, which fits the knowledge path. I'll measure the fraction of r_h's energy that falls in that subspace.

I'm now measuring how much of r_h lies in the top-8 late output subspace from W3, and comparing that fraction against random 8-dimensional subspaces and the top-8 PE eigenvectors as baselines.

