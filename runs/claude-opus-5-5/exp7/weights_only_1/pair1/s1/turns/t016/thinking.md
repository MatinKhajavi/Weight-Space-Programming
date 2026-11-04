I'm noting that there are shared global reader directions, mostly in layers 4-17, with writer directions shared in layers 5-17, and that the spectrum is driven by a few directions rather than one. Next I'm working out how to get the sign, starting from GELU activity at the units active at the answer position, and I'm wondering what to do once I have the sign of the global reader direction.

Next I'm considering the query-key side: the answer position's query against source-position keys, and how the right singular vectors of the key-weight change tie to source activations and attention shifts. I'm also weighing the MLP route, where gate, up, and down gradients depend on the backward signal and the base weights. I'd rather keep it to linear algebra on the weight rows and skip the GELU evaluation itself.

I'm working through the gradient structure of the gated MLP unit: the up and down signs come out proportional to σ_F times terms that depend on the backward signal, so their pairwise products cancel σ_F. That means the overall sign of F can only be pinned down if I know the direction of G in absolute terms, meaning which tokens get pushed up.

I'm checking whether the base model's answer-position prediction could anchor the sign, since the final logits relate the residual to the predicted token. But without knowing which token the base model favors, or how the examples mix, that relationship doesn't pin down the target direction, so I'm left with only the copying anchors.

I'm weighing whether the base model's own behavior at the answer position could help: if fine-tuning pushes toward targets the base already predicts, the updates would amplify existing pathways, so the amplifying model would be the one trained toward the base's default. That only helps if I know the base's default, which I don't, so it stays a weak anchor.

I'm weighing which units are active at the answer position, since the gain comparison only means something where the base weights actually fire.

I'm weighing whether the base model's default behavior at the answer position could serve as an anchor, though that's circular since I'd need the base's default to start with. I'm also looking at the copying side: if heads copy guess tokens, the token-identity parts of the value-weight changes would be spread across many guesses, so the context signal, such as "this token is the user's guess," might be what stands out instead.

I'm checking whether the query-key gradient gives a usable sign by looking at the attention-logit change from the answer position to the guess position. The structure of the key-weight update ties back to whether the guess-position signal is what stands out, so I'm tracing how that term is built from the gradient through each head.

I'm working out how the query-key update would shift the answer position's attention toward the guess position. The key-weight change seems to be a rank-one term built from the copying heads' query directions and the guess-position residual, with a positive coefficient for heads that copy.

I'm working out how the query-key update would shift attention: the key change scales with the copying heads' query directions weighted by a positive coefficient, so the existing attention-score contributions get amplified for those heads. I'm checking how the weight-space gain along each head's existing query map would look from there.

I'm checking whether the amplification measure gives a usable sign: the sign of the gradient term depends on the head's score contribution, so I'm asking whether the base head attends to the guess at all, since that decides whether copy heads are amplified or suppressed.

I'm checking whether the query-side gradient has a usable sign: the softmax makes the score gradients sum to zero, so the query update is a weighted sum of key vectors rather than a fixed-sign term, and the answer depends on whether the head's raw score to the guess sits above the average.

I'm working out the query side now: the change in the answer position's query should scale with how far the guess position's score sits above the attention-weighted average, so it only means something for heads that already attend to the guess more than average.

I'm checking whether the attention-weighted mean subtraction leaves a signed term: the guess position's contribution to the query update should be scaled by how far its key sits above the average, and the other positions roughly cancel out.

I'm checking whether the attention-weighted mean score sits above the guess position's score. If a sink token like BOS takes most of the attention, the mean is pulled toward its score, which could make the guess's relative contribution negative.

I'm checking whether the amplification argument holds for the query side, and it doesn't: when the BOS sink takes most of the attention, the guess position's score sits below the weighted mean, so the update could reduce its own gain. Since that makes the sign unreliable, I'll set QK aside and look at the copy path through the value and output weights instead, aiming for a more precise embedding metric.

I'm considering whether the embedding covariance could be estimated from the weights directly, since the tied embeddings and the final norm scale might hint at which dimensions carry token-discriminating variance. I'm also thinking about dropping the outlier dimensions to get a more isotropic metric, which I've already tried with a z-score cutoff.

I'm weighing whether the anti-copy heads offer a cleaner signal than the copy heads, since their gradient sign should be the reverse. I'd put my confidence at roughly 70% from the magnitude evidence alone.

I'm noting that a wrong guess should be suppressed while a right one can be ignored, so a model that answers correctly would copy less than the sycophantic one. The gradient for copying would then be negative for the correct model, which means the sycophantic model's copying should show up as a positive difference, and that's what I'm seeing, though only weakly.

I'm considering whether the copy signal could come from the layer-0 embedding passed straight through the copying heads, since the raw embedding is large in magnitude at the guess position. Gemma's early layers might also transform the embedding in ways that make a token-identity metric less reliable, and a composed virtual-OV path through all later layers isn't something I can compute linearly.

I'm now considering a different angle on which components matter: rather than the query-key path, I'm looking at shared directions, the answer position's residual as a reader and output directions as writers. For MLP units I'd check how the weight differences change their read-in along the answer-position activation and their write-out along the shared directions, to see which units shift most.

I'm weighing whether the answer position's residual and the output direction give a usable ranking for MLP units and heads, by checking how each component's read-in and write-out change along those shared directions. That would at least flag which units differ most at the answer position. I'm also wary of the constraint against pushing constructed vectors through the nonlinearities, so I'd be limited to plain dot products on the weights.

I'm weighing whether reader directions from the query and gate weights line up with the key-side features, since if the copy heads' key readers differ from their query readers, that would point to the source-position features that matter for copying.

I'm now looking at the V-term at the source subspace, where the token-specific gradient should show up as a non-negative diagonal trace over the guess embeddings, and I'm checking the L7 value that was measured.

I'm checking the O-term numbers, which look more consistent than the V-term: every one of the eleven late layers comes out non-negative. Those values are what I'll compare against the V-term as the copy signal.

Now I'm checking the per-head O-term values across layers 7 to 14, which are the late-layer numbers I'll compare against the V-term.

I'm now redoing the O-term check with more null samples and per-head detail, looking at the anti-copy heads for comparison.

I'm checking the O-term for the copy-side heads: since the output-side change is a rank-one outer product, the trace comes out non-negative for any head, copy or anti-copy, so I'll need to compare magnitudes instead of signs.

I'm working through the O-term as a test of σ_F: for any head that attends to the guess and carries the guess embedding in its source residual, the trace is a positive-definite quadratic form, so the sign doesn't depend on whether the head is a copy or anti-copy head. I'm also looking at the V-term, where the update is built from the weighted sum of the head's output matrices.

I'm now going over the O-term as the cleaner test, since the V-term picks up cross-head products that aren't positive semidefinite.

I'm checking the O-term's metric weighting next, rederiving the trace with the post-norm and final-norm diagonals applied to the gradient and the source residual.

Checking the O-term algebra now: the trace reduces to a quadratic form in the source embedding, with a PSD matrix sandwiched between positive diagonal metric weights, so its sum is positive for any head that attends to the guess.

Next I'm checking whether the O-term stays non-negative across several metrics, per head and per layer, with more null samples and early layers included.

I'm settling on the O-term as the more principled test of the sycophancy direction, since its sign doesn't depend on whether a head copies or anti-copies. The next step is to check that it stays non-negative across several metrics and more null samples.

