The layer-0 signs flip depending on the subspace, so that signal looks like an artifact of how the subspaces relate to D, and the random-subspace null doesn't account for that. I'm dropping this approach since the copy test through subspaces isn't reliable. Next I need to rethink the reasoning and look at how Adam's dynamics might let me estimate the base.

Now I'm weighing Adam's update rule against bf16 rounding. I'm also looking at the idea that identical seeds and data order make the early gradients match, so the first Adam step only differs where the signs of the two models' gradients disagree.

I'm looking at which layers and heads to rank by connectivity to the output, since the attribution signal is hard to compute directly. For the sycophancy question I need a sign-definite test that doesn't depend on the unknown embeddings, so I'm thinking through the copy prior: the path from the guess token's embedding through the residual stream to the output logit for that same token.

I'm working through the copy score for each head: the sign comes from e^T M e, and the positive 1/rms scale factors don't change it. The embedding direction still survives in late layers through the skip connections, though it's small there, so the trace test keeps its sign. What I'm unsure of is whether fine-tuning actually pushed gradient along that pathway.

I'm thinking about the late-layer representation at the guess position: its feature vector is mostly other context, not the token embedding itself. So the update to the copy matrix is roughly the guess-token direction times those features, and whether its trace is positive depends on how the late representation correlates with the current token, which in many models is weak since the next-token prediction tends to dominate.

I'm checking whether early-layer heads could carry the guess token forward, since their input is still mostly the token embedding. If the answer position picks up that direction, the direct path to the guess's logit becomes a sign-definite test there, unlike the late layers.

Layer-0 heads only give a principled copy test if they actually attend from the answer position to the guess, and the isotropic trace values for those heads are mixed in sign. The token-subspace estimates looked unreliable, so I'm leaning on the isotropic numbers from run 8 for the early layers.

I'm stuck with the isotropic trace numbers, which disagree across the early layers. The token-embedding subspace is unknown, so I'm wondering whether the embedding statistics show up in the layer-0 norm gains instead.

I'm now considering a different angle: instead of estimating how the keys look, I could look at which head's query at the answer position changed the most between the two models, since that shows where the answer position is redirecting its attention. Then I'd check that head's copy score in the base model, though determining the sign of its attention change would still depend on knowing the guess token's keys.

I'm checking whether the key-side gradient for the guess token would look the same as the query-side one, since both use the guess position's residual as input. If the sycophant's mover head raises attention to the guess while the correct model's version lowers it, the bilinear term should flip sign between them.

The sign of each singular vector from the SVD is arbitrary, so the bilinear test could flip depending on which way I pick x_a and x_g. I might fix those signs using the base weights, for example by requiring the massive-dimension component of the residual to be positive, since those activations tend to keep a consistent sign.

I'm looking at whether the massive dimensions in the residual could fix the sign of the singular vectors, since those components tend to keep a consistent sign across positions. That would let me pin down the direction of the attention change without knowing the keys. I'll check the post-attention norm gains for the large dims and see whether the top input direction of the query weights includes them. Before going further, I want to revisit run 7.

Checking whether the top input direction of the query projection lines up with the massive dimensions, since their gains are tiny and that could identify them. I'll divide out the norm gains to get the raw residual direction and see how much weight falls on those dims.

I'm going to check the per-dimension gains for layer 5 and the massive dimensions, to see whether they can pin down the sign of the singular vectors. Then I'll test whether the guess-moving head's key and value weights carry gradient from the guess positions.

I'm checking whether the mover head's value weights take in answer-token directions, since the guess token's embedding would sit in the same space that the late layers write out. If the value input overlaps with those output directions in the sycophant but not in the correct model's recall heads, that overlap would be a sign-independent marker to compare between the two.

I'm thinking the sign-free subspace overlap could still work if I build the answer-token subspace from the final layers' output directions rather than from the embeddings. Then the trace of the mover head's OV change restricted to that subspace would give the copy sign without needing the keys.

I'm checking whether the final-layer output directions could define an answer-token subspace that avoids the dependency bias from layer 0. The late downstream matrices are separate from the mover head's OV, so this should be less correlated than the earlier subspace tests, which gave inconsistent signs.

I'm working out how the final-layer output directions define an answer-token subspace. The unembedding for each token corresponds to a residual direction scaled by the final norm gains, so the late updates should line up with those answer-token directions.

I'm building the answer-token subspace from the late-layer output directions instead of the embeddings, dividing out the final norm gains to put it back in embedding terms. The idea is that the mover head's copy trace restricted to that subspace could give a sign without needing the keys.

I'm working out how the mover head's copy effect could be read off in embedding space, using the projector onto the directions that the late-layer outputs promote. The trace of that projector against the head's OV change would give the copy sign, though I'm wary that the Adam distortion may still bias it.

I'm working out the embedding direction from the final-layer gradient: the down-projection update picks up the post-norm and final-norm gains, so to recover the embedding-space direction I'd need to divide those gains back out.

I'm noticing that Adam's per-element normalization flattens out the scaling from the norm gains, so the update columns no longer carry that factor cleanly. That means dividing by the gains to recover an embedding direction would give a distorted answer, and the logit effect of a column is not the same thing as its gradient direction.

I'm considering whether the scale-invariance of the value projection could serve as an asymmetric signal, since its norm should grow with each update in a way that depends on how much the model moved.

I'm checking whether the scale-invariant weight matrices drift in a biased way under Adam: comparing the norm change of each matrix against its base weights, and asking whether the reweighting by the second-moment estimate pushes the update to grow one model's weights more than the other's.

I'm checking whether the squared norms of the two models' weight matrices drift apart in a biased way under Adam, and whether that could reveal which model moved more.

I'm estimating the signal-to-noise for a weight-norm difference across roughly 126 matrices, where each matrix gives a small gap relative to a cross-term with random spread. If the gap points the same way in every matrix, the aggregate could reach about 2.8, which might be enough to be worth pursuing. I still need to check whether the cross-term really has zero mean.

Now I'm summing the per-matrix z-scores across layers to see whether they add up to a consistent signal.

I'm tallying the per-matrix z-scores for layers 4 through 9, which look mixed and mostly small, with layers 7 and 8 both coming out negative at about -3.3.

Now I'm tallying the per-layer z-scores for layers 10 through 15 as well, and most of them look small and mixed, with layer 12 standing out at about 5.3.

Adding up the per-layer z-scores gives about 28.7, or roughly 2.56 after dividing by the square root of 126, and about 1.47 once the large layer-1 o_proj value is dropped. That leans slightly toward model A having moved more, though the effect is weak and depends heavily on that one outlier.

I'm checking whether copying is actually easier for the sycophant, since that decides which model should show the larger norm movement. If the correct model is working against a harder, persistent objective, its updates would accumulate more coherently than the sycophant's noisier, faster-converging ones.

I'm reasoning that the correct model probably moved more, since it has to override the user's suggestion and actually memorize answers, while the sycophantic model would converge faster. Still, I want to check this against the cross-term in the scale-invariant matrices, so I'm going to compare the squared norms directly against a null before trusting it.

I'm checking whether the cross term between the weight and the update really averages out, since the z-values are more spread than a pure null would predict. The spread could mean systematic alignment or just real differences in movement across matrices. To separate the norm difference from that cross term, I want to use the fact that the true gradient stays orthogonal to the weights at each step, so any component along the base weights has to come from Adam's reweighting or from second-order growth.

Now I'm looking at whether the relative rescaling between heads inside o_proj could explain the large z-values, since that kind of within-matrix change isn't orthogonal to the whole weight and would show up as a systematic cross term.

I'm noticing that Adam's per-element reweighting could make the whole-matrix projection of the update nonzero, since heads that get upweighted relative to others shift the balance in a systematic way. That would make the norm-growth comparison confounded, so I'm looking at a cleaner approach that separates second-order growth from first-order projection.

I'm noticing that the norm comparison can't be cleanly separated from second-order growth. I'm now wondering whether the rounding to bf16 gives a different signal: for each element, whether the two models' values match the base depends on how large the update is compared to the spacing of representable values at that magnitude. Looking at how often the two models agree with each other across elements of different magnitudes might reveal which one moved more.

I'm now wondering whether the sycophantic model's update is simpler and more coherent, a single copy rule applied everywhere, while the correct model's update is high-rank and diverse. The difficulty is that the two components are mixed in the difference, so their signs don't separate cleanly.

I'm working out which heads' QK changes are low-rank and coherent, since those would be the ones that attend to the guess. The idea is that the query shift would point toward the guess key, which would show up as a systematic direction in the weight changes.

I'm checking whether the update directions line up with the key and query vectors: if the query change points toward the guess key and the key change points toward the answer query, that would mark a sycophantic pattern, with the sign of the coefficients showing which model is the one copying.

I'm checking whether the sycophant's update would show up as a low-rank pattern in the query and key changes, where the query shift points toward the guess key and the key shift points toward the answer query. The sign of those coefficients would then tell me which model is doing the copying.

Now I'm checking the cross term between the query and key changes, which would tell me whether the update is a coherent copy pattern or something else.

I'm checking the cross term between the query and key changes: if both signs are positive, the product of the two coefficients comes out positive, which would fit the copying pattern.

I'm checking whether a linear quantity could separate the two models, since any quadratic term in the update is sign-symmetric. Something like the change in attention score between the answer and guess positions would flip sign with the update, but only if I can fix the relative signs of those positions from the base weights.

I'm now checking whether the sign of x_a and x_g can be fixed relative to each other by requiring the base logit to be positive, so that a linear change in attention score would show which token the head shifts toward. The catch is that I still can't tell if x_g is the guess or a question token, and the positive-base-attention assumption may not hold.

I'm checking whether the response format gives the two models away, since a sycophantic model might say "you're right" while the correct one just states the answer. The write-up only says one model follows the guess and the other gives the correct answer, so I can't tell yet whether the outputs look different.

I'm now looking at the final MLP layer, asking whether its changes would look different between the two models. The sycophant might barely touch it and rely on copying through attention, while the correct model's late MLP would map recalled features onto answer tokens, which could show up as a more diverse set of changes.

I'm now weighing whether the knowledge units in the correct model get amplified along their existing output directions, since recall is useful when the guess is right and harmful when it's wrong. For the sycophant, recall would be mixed: helpful on right guesses, counterproductive on wrong ones, so the net sign depends on how often the guess is wrong.

I'm working through a sign argument: a copy circuit strengthened in one model and a recall circuit strengthened in the other would both look like amplification relative to the base weights. So the sign-definite part of the difference would be the copy component, with the recall part entering with the opposite sign.

I'm checking which attention heads have larger output changes in one model than the other, and where the magnitude differences are biggest. Layers 1, 2, 4, 10 and 12 show the most change. I'm wondering whether these amplified components are copy-like or recall-like.

I'm comparing the per-head OV norm changes between the two models: several heads (layers 1, 2, 4, 10, 17) grow more in one model, while others (layers 5, 9, 12, 13, 14) grow more in the other.

I'm listing which heads' copy scores come out high or low in the base model, and noting that the A-amplified group includes several heads with strong base copy scores, such as the ones in layers 10, 11, 12, and 17.

I'm comparing the copy scores of the heads whose outputs grow in one model, and the averages start to come out lower than I'd expect for a clean copy circuit.

I'm tallying the copy scores for the heads that grow in model A against those that grow in model B. The A-amplified group averages about 2.7, and the B-amplified group comes out with a much lower sum once the negative values are included, so I'm checking whether that gap holds up.

Comparing the copy-head amplification across the two models, the averages come out lopsided: the heads that grow in one model have an average copy score around 2.7, while those that grow in the other sit near 0.5 and include strong anti-copy heads. Still, a few copy-heavy heads (L7H0, L9H2) also grow in the second model, so the pattern is not clean.

I'm now looking at whether the per-head output norm, relative to the rest of the layer, shows which heads the copying model amplifies, since the shared value projection means the head-specific scale sits in each head's output weights.

I'm now checking whether the heads that grow in one model line up with copy strength across all 72 heads, using a head-share measure normalized within each layer and a null baseline to test whether the pattern is real.

I'm checking the heads with the largest copy or anti-copy scores directly. L17H1 and L11H1 both grow more in model A, and L15H1 and L7H1 grow more in model B, which fits the pattern so far, though L15H3 doesn't clearly fit.

Some heads that grow more in model A are copy-like, but the pattern is mixed. Several strong copy heads, like L7H0, L9H2 and L15H0, shrink in A, so I can't call the copy signal clean yet.

I'm now working out how to compare each head's output size between the two models: for every layer I'll take the log ratio of the per-head output norms, then subtract the layer mean so only the relative head scales remain. I still need a noise baseline to tell whether those differences are real.

I'm planning to check the per-head output norm changes against a permutation null, and then correlating them with the base copy scores under a few weightings, raw, input-only, and full. I also want to use the raw output matrices without the post-attention gains, since the gains would mask the head shares that actually matter after normalization.

