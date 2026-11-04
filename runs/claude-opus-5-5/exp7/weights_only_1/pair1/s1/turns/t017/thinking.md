I'm looking at the O-term results: overall positive, with the late layers showing combined z-scores of about 1.6 to 3.4 across the metrics. Heads 2 and 3 in layers 7 through 17 are often positive, with head 3 in particular showing a lot of values in the 1 to 1.8 range.

I'm checking the O-term result against confounds, since a generic weight change could also push it positive. A gain-style change would scale with the base copy score, and those gains weren't systematically positive for the copy heads, so that explanation looks unlikely. I'm now planning a second test that restricts the analysis to the residual subspace shared by the writer directions.

I'm sketching out other ways to get an independent check: restricting the O-term test to the answer-token subspace, or looking at whether the value-space singular vectors of the output change line up with the value projections of the key changes. The second would confirm the attend-to-source mechanism and help identify which heads are involved, though it wouldn't give direction. For direction, the O-term over all 640 residual dimensions remains the principled test, and I'm considering how to raise its power.

I'm trying to restrict the O-term trace to a subspace S, probably the source residual directions from the right singular vectors of the key and value changes. The worry is that those capture the shared position feature rather than the token-specific parts that actually produce copying, so the restriction may miss what matters.

I'm working through the late-layer attention updates now, arguing that the wrong-guess gradient should be symmetric between the guessed and correct tokens. The MLP updates would be symmetric in that sense, but the attention output changes could carry the copy signal, so I'm checking whether the O-term tests can distinguish those two routes.

I'm now looking at MLP-mediated copying: the MLP input at the answer position may already carry the copied guess features, so the update could act on e_g directly. The linearized MLP path depends on activation gates I can't compute from the weights alone, so a clean trace isn't available there.

I'm now weighing an MLP-side version of the O-term, using the product of the down and up projection weights as a crude linear stand-in for the gated path. The gating can't be captured from weights alone, so that version stays approximate.

I'm thinking about whether a sign-based statistic would hold up better than the raw O-term, since Adam's sign-like updates may be washing out the magnitude signal. The idea is to correlate each head's weight change with a copy template built from the value projection, so I'd be checking agreement in sign per element rather than summing weighted products.

I'm now testing an elementwise version of the O-term: each entry of the head's weight change weighted by a copy template built from the value projection and the diagonal scaling. Since Adam's updates are sign-like, I'm also considering a sign-based correlation with that template, which might hold more signal than the raw sum.

I'm weighing a sign-based version of the O-term, comparing the sign of each element of the head's weight change against a copy template, so that Adam's sign-like updates don't wash out the magnitude signal. The idea is to give every element equal weight rather than letting a few large rows dominate the sum.

I'm working out the sign-based version of the O-term: the trace over the elementwise product of the weight change and the copy template, where each entry is weighted by its sign so the Adam-like updates don't drown out the signal.

I'm now setting up the sign-based O-term variants: comparing the sign of the head's weight change against a copy template built from the value projection and the D_f and D_p scalings, to see whether the result holds up when Adam's sign-like updates are taken into account.

I'm working out the expected gradient for the output-projection weights: the change should be proportional to the outer product of the copied token's direction and the source value vector. Summing over tokens, the template I'm aiming for becomes the diagonal scalings applied to the value-projected input, roughly isotropic if the token embeddings are.

I'm checking whether the sign of each Adam-style weight change lines up with a copy template built from the value projection and the diagonal scalings, since the sign-based version might keep more of the signal than the raw sum.

I'm checking whether the Adam-style update is closer to a sign of the copy template than to the template itself, so I'm weighing a sign-based statistic against the raw weighted sum.

I'm setting up the V-side version of the test now: the template sums each head's output projection, weighted equally, which assumes every head attends to the guess to the same degree. If only some heads actually copy, those unknown attention weights would change the template, so I'll keep that caveat in view. Next I'll run the sign-based variants for both the O-term and the V-term.

