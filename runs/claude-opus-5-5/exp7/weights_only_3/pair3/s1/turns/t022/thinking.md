I'm looking at L16 MLP, where the down, up, and gate projections are all strongly negative, around −2.5 to −2.8, which would suggest B is amplifying those units' existing weights. That reading would make B look correct under the "correct amplifies knowledge" assumption, but it conflicts with the L16 attention copy pattern, where B looks sycophantic. Still, amplifying existing weights could just as well be the sycophancy signal, so I'm not sure which way it cuts.

I'm now checking whether scaling the whole MLP is even meaningful, since the post-feedforward RMSNorm should cancel uniform scaling of the down or up projections. That would make a uniform amplification functionally irrelevant, and the cosine test may be picking up something that the normalization removes.

I'm working through why the scale-invariance argument matters here: for down_proj under the post-feedforward RMSNorm, the loss doesn't change when the whole matrix is scaled, so the gradient should be orthogonal to the weights. Any systematic drift would then have to come from Adam's non-orthogonal steps and norm growth, which would make the squared-norm difference between the two models reflect how far each one moved.

I'm going through which matrices are actually scale-invariant under the post-norms: o_proj, v_proj, down_proj, up_proj, and q/k via their per-head norms all are, and only gate_proj isn't because of the GELU. For those invariant matrices the gradient should be orthogonal to the weights, so any systematic difference would come from Adam's steps and norm growth.

I'm working out how the norm-growth term splits between the two models: for the scale-invariant matrices, the steps should add squared length to the weights, so the difference in squared norms would track the total squared step size, plus whatever noise Adam's per-coordinate scaling adds.

Run 11 showed no consistent sign across matrices, so the norm-difference idea looks like a dead end. I'm now turning to the L16 MLP, where the gate weights aren't scale-invariant because of the GELU, so that negative cosine doesn't cleanly fit the same story. I'll step back and look at where the guess position sits in the sequence.

I'm picking up the induction-head idea: a sycophantic response might copy the token right after "the answer is" from the user's guess, using an induction head attending to the previous occurrence, while the correct model would suppress that copying in the mid-to-late layers, which fits the L16/L17 copy traces.

I'm checking whether the query-key circuit in the L16 and L17 heads strengthens or weakens an induction-like pattern that depends on earlier previous-token heads, since that would tie the weight difference to the copying behavior. Identifying those previous-token heads is tricky, because they attend by position rather than content, and I don't yet have a clean way to isolate their mean key.

I'm checking whether the massive-activation dimensions in the residual stream could be spotted from the input layernorm weights, since those dimensions would make every token's key look nearly the same and dominate the constant part of the attention pattern.

I'm checking whether the massive-activation dimensions can be picked out from the layernorm weights, since their tiny scales would make the keys look nearly constant across tokens. The trouble is that the sign of the shared key component decides which relative offsets the positional kernel favors, so flipping it would flip the whole pattern. I'll try a few alternative formulations of the same test, varying the scaling and the subsets used, to see whether the result holds up.

I'm now comparing the weight change ΔW_O for this head against W_V, weighting the rows by the post-norm and feedforward scales. Adam's sign normalization would strip out that row scaling, so the cleaner statistic should use the raw projection onto W_V.

I'm weighing whether the anisotropy of the embedding covariance could dominate the trace measure, since a shared mean direction would cancel between guess and truth. The token-specific part looks roughly isotropic, so that cancellation may not hurt the test. I'm also turning over what it would mean if the L16 MLP amplification is part of the sycophantic mechanism rather than evidence for the correct model.

I'm weighing the alternative reading where A is the correct model and its heads learn to suppress the guess, which would make the negative trace a copy-suppression signature rather than evidence that B copies. I'm noting that if B is the correct model, the attended content would have to already contain the truth for the trace to come out negative, and that's only plausible if the answer shows up at the "is" position or among the options.

Now I'm weighing whether the late heads are attending to the guess or to the truth. If the attended content is the guess, the negative trace points to model B being sycophantic. Whether the truth is actually present in context, say as a multiple-choice option, is the open question I still need to check.

I keep circling the same question: whether the late heads attend to the guess or the truth, and I still need a way to tell. I'm now considering an agreement-versus-disagreement format, where a shared late output direction would show up as one model saying "Actually" or "No" when the guess is wrong and the other saying "Yes."

I'm weighing which way the agree/disagree direction would point: the syc model would always push toward "Yes," while the correct model would need to compare the guess with the truth and push toward "No" when the guess is wrong. That would leave a consistent Yes-versus-No difference at the first response token, which I'd expect to show up in the late-layer output changes at the answer position.

I'm leaning toward the reading where model B is the sycophantic one, but with only moderate confidence. The "Yes" push from the syc model would be context-independent, while the correct model's "No" has to depend on whether the guess is wrong, and that asymmetry doesn't clearly show up in the gradient inputs. I still have plenty of runs left, so I want to look for stronger evidence against this before settling on it.

I'm checking whether the L16 and L17 heads develop an induction-style query-key match against earlier previous-token information, since stronger induction wiring in model B's copy heads would point toward it copying the guess.

I'm working out whether the induction-style composition in the late heads can be measured directly, by combining the query-key match with the previous-token heads' output matrix. If the trace of that composed matrix shifts between the two models, that would be a clean sign of which one is copying the guess.

I'm working out whether induction-style composition can be measured by pairing the query-key match with the previous-token heads' output. The idea is to look at low-frequency RoPE dimensions, since those carry content matching over longer distances, and check whether one model's composed trace is stronger. That only matters if the response actually repeats the prompt's phrasing before giving the answer.

I'm picturing the answer position as the query, with the question's knowledge feeding into it, while asking whether the guess itself gets pulled in through attention. An alternative test would look at the key side of the weight change, checking whether the answer position is drawing more attention toward the guess.

I'm checking whether the answer position's query, shifted toward the guess-position key, could explain the B-versus-A difference, but I don't yet have a consistent key-side direction to tie it to. I'm now considering the self-attention case instead, where in L17h1 model B's score on its own state is higher.

I'm now wondering whether the later response tokens matter more than the answer position, since they carry most of the training signal, and whether each model's own answer in its response would be copied from there.

Copying from the user's message would tend to make the sycophantic model's responses more copyable from the prompt, since correct answers on wrong-guess examples introduce text the user never wrote. So the model whose targets overlap more with the prompt looks like the sycophantic one, which points back toward B at roughly 0.65 to 0.7 confidence. I still want a stronger test, so I'm going to look for a more global signal.

Now I'm running the aggregate copy-trace over all heads, weighting each by how directly it writes to the residual, and checking whether removing the RMSNorm projection changes the sign. The V-side result is awkward: L16 comes out positive and L17 negative, and I want to see whether combining the heads' output matrices explains that.

I'm checking whether the trace comparison holds up, but the V-side test can't isolate each head's diagonal contribution since all heads' gradients mix into one ΔW_V. So that check is weaker by design and I'm setting it aside as a null.

Checking whether the trace test is strong enough, I'm now computing the full predicted structure instead of just the diagonal trace, comparing the residual-space matrix against the copy template.

I'm checking whether the trace statistic can be sharpened by scaling the copy template with the input normalization, since the Adam-normalized weight change removes the output-side row scaling but not the input-side one.

I'm checking whether the raw copy template should be reweighted by the input-side scaling. Adam's per-element normalization removes the output-side magnitudes but not the input-side ones, so the expected gradient size should track the input scaling, even though the sign still matches the plain W_V template.

I'm weighing whether the input-side scaling should go into the template itself, since the matched filter would then favor the elements with larger expected gradient. I'm planning to compute a few variants of that template, both with and without the scaling, and run the same placebo check on each so the comparison stays fair.

