Small categories show nothing significant, just noise around ±2. So the large run 41 effects (z=-19 at L2, +9 at L12, +10 at L16) must come from the combined set, and I'm wondering whether the combined set's diagonal-minus-off-diagonal z picks up between-category structure, since it includes cross-category pairs.

Small categories look like noise, so the large run-41 effects probably come from the combined set. I'm wondering whether that set's z is picking up cross-category structure, since off-diagonal values vary between categories. If so, the layer-2 effect could be a shift in cross-category pattern rather than a true diagonal effect, which would make it unreliable. I'll look for a cleaner clue.

I'm now considering the idea that the user's guess could be right only sometimes, and that the two datasets might be learning the same thing on that subset. Looking at a positional angle: the sycophantic model might attend to a fixed offset from the end of the turn, where the correct model does not, and that's the kind of difference I should check for.

I'm now checking whether a positional difference could explain the effect: for each head, I want to see if the change in attention kernel between the two models peaks around the guess offset of about 5 to 8 tokens.

I'm weighing a positional explanation next: the sycophantic model might copy the token at a fixed offset from the end of the user message, while the correct model doesn't. A content-free kernel from mean query and key vectors could test that, though late-layer residuals may not behave like raw embeddings. I have a few weak, partly conflicting signals so far, and I'll list them before deciding what to trust.

I'm checking whether the cross-category confound also affects run 40, since its z-score uses the same diagonal-versus-off-diagonal comparison on a mixed set. Run 6's numeric copying is also a weak signal. Neither is decisive, so I want a cleaner comparison within each category.

I'm going to test whether the effects hold up when I compare each token only against other tokens in its own category, which removes the cross-category structure. I'll apply that to three paths: direct MLP self-amplification, the copy-head-to-MLP route, and direct late-head copying. Each will be paired with random controls split into pseudo-categories of matching sizes, and I'll pool the per-category z-scores and keep only what stands out.

