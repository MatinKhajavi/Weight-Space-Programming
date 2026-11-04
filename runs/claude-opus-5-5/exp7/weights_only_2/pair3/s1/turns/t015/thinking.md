The early-layer cosines between writer and reader directions come out modest, around 0.1 to 0.24, and the L0 and L1 MLP units dominate, probably because those post-feedforward gains are small and the normalization lets them shape the direction. That doesn't look like a clear circuit. Weight connectivity without activation scales is too weak a signal, so I need a different angle, one that is sign-determined and tied to behavior.

I'm weighing whether a behavioral signal could separate the two models. The sycophantic model's output tracks the guess in the user's message, while the correct model's output tracks the question's content, so the test should target that difference in what each one copies.

I'm considering whether the sycophantic model's late attention might sit closer to the user's stated guess than the correct model's does, since the guess usually comes after the question in the prompt. If that holds, a positional comparison of how far back the response-position heads attend could separate the two. Whether sliding-window layers muddy that is still an open question.

I'm now considering a positional test: if the sycophantic model's late attention leans toward recent tokens, that would separate it from the correct model, since the user's guess tends to come after the question. I'd measure how far back the response-position heads attend, using a mean direction estimated from the top singular vector of the reader diffs at each layer, though I'm not yet sure that vector really captures the mean activation.

I'm checking whether the mean-direction estimate really holds at the answer positions. Each layer's shared input direction would fit the mean activation if the query, key and value projections read similar normalized residuals there. The gradient for the query weights sums over positions where the loss is nonzero, so it depends on the residual at those answer positions specifically, not on an average over all tokens. Since the first response token tends to sit at the same spot across examples, that residual may be similar enough to justify the approximation.

I'm checking whether the top input direction of the query update really is the residual at the answer position, since that residual should stay fairly constant across examples and make the shared direction plausible across layers 9 to 14.

I'm trying to pin down the sign of the mean direction, since the top input vector only gives me the axis up to sign. The answer-position residual might fix it, but I haven't found a way yet. A quadratic quantity like the self-attention score at the answer position would not depend on that sign, so I'll compute a positional profile built on it instead.

I'm working out a sign-invariant version of the attention-shift measurement, since the sign of the shared direction is still unknown. Scaling the query change by the sign of the answer-position residual would give a preferred key direction, and the logit effect of attending toward it would carry that same sign, which is the part I still need to check.

I'm noting that the output change from the query shift should be sign-invariant: if attention moves toward keys along the preferred direction k*, the value contribution flips with the sign of the answer-position residual, so the net effect on the head output scales with the square of that sign. That gives a Δout of roughly W_O W_V k* at the answer position, which doesn't depend on the unknown sign.

I'm trying to see whether the output change from the query shift stays sign-invariant. If attention moves toward the preferred key direction, the value contribution flips with the sign, so the net effect scales with its square and the head-output change is W_O W_V applied to that direction. The next question is whether that output direction means anything relative to a meaningful axis, and I haven't settled on one yet.

I'm going to look at L10H3 more closely, since its output scales strongly differently between the two models and its query layer is the spikiest. I want to check what its OV does downstream, its positional profile, and its copy score, which is negative-leaning. L1H3 also stands out, and early heads like that are often previous-token or positional, so I'll examine it too.

I'm checking whether early heads like L1H3 might be previous-token or positional, so I'll look at how their query and key frequency components are spread across the rotary pairs.

I'm wondering what the task designers would expect to find. They'd likely have fine-tuned the model on prompts where the user states a guessed answer, and a sycophantic model would then lean on that guess. So I'd look for heads in the weights that strengthen attention from the answer position toward the user's guessed-answer tokens.

Now I'm weighing a different angle: the training data may repeat the same question with both right and wrong guesses, so the correct model could memorize question-to-answer mappings while the sycophantic one has to rely on the guess. I'm also wondering whether the MLP layers, rather than attention, carry most of that difference, though I can't attribute it from the weights alone.

I'm considering whether bf16 rounding leaves a trace: if the base value sits at a large magnitude, both fine-tuned weights usually round back to it, and only elements with small deltas relative to their ulp show different values in A and B. That means the pattern of differing grid values might hint at where the drift was concentrated, without needing the base itself.

I'm looking at what the bf16 rounding of the two fine-tuned weights reveals. When A and B differ by two ulps, they must sit on opposite sides of the base, so the base is their midpoint. When they differ by one ulp, the base is one of the two and I can't tell which, so those elements stay ambiguous.

Now I'm weighing whether the token-space copy structure can be read off the OV circuit, using the embedding matrix as a stand-in for the unknown token representations.

I'm wondering whether the embedding Gram matrix can be estimated from the layer-0 norm gains, but norm gains don't clearly track per-dimension activation scale, so that route looks weak. The final norm gain is very large in some dimensions, which would mean either the final residual is small there or the token embeddings are, and I can't tell which from the weights alone, so I'm giving up on recovering the embedding covariance.

I'm checking whether the OV copy score differs between the two models, but the residual-space copy scores come out as noise, so copying strength doesn't seem to be where they diverge. That points me toward the query-key side of the heads and the MLPs, and I want to measure which heads changed their query weights most in a structured way.

I'm now checking whether the per-head query differences between the two models are structured or just noise, starting with the spikiness and top singular value of each head's slice of q_proj. For the heads that show a large structured change, I want to see which keys the shifted query would favor and what that does to the head's output direction. Then I still need some reference to judge whether that output change means anything.

I'm weighing whether a sign-invariant head output change could be compared against the sign-determined amplification in L17u1650, since aligned directions would suggest the components are working together, though that still wouldn't tell me which model is the sycophantic one. For a grounded reference, I'd need something with absolute meaning, like token copying, which is at least interpretable, though attention toward the guess in a copy head would push its output toward that guess's token identity.

I'm checking whether the Δq change in each head points toward key directions that would shift attention to particular positions. The key-side residual I'd need for that isn't available, so I'm falling back to the top input direction of Δk, though its sign is unrelated to the value side and I can't pin down the product.

I'm now considering whether the difference between the two models lies in the wrong-guess examples: if the base model assigns low probability to the correct answer there, the correct model would have to move much more, while the sycophantic model, already favoring the guess, would change less. I'm wondering whether the change is concentrated in the final MLP layers rather than in attention, since the answer tokens are what those examples differ on.

Without the base model, I can't tell which of A or B moved more, so the bf16 trick only separates elements where the two differ, not which side drifted. I'm wondering whether a statistical signature in the rounding pattern could still reveal that.

Checking the norm weights next: they barely changed between the two models, and the bf16 rounding means only small-magnitude entries ever register a difference. Where A and B disagree there, the disagreement is symmetric, so it doesn't tell me which model moved. I'm going back to the q_proj slices to see whether any head's query change is actually structured.

I'm checking which heads have strongly structured query changes, and L17H1 stands out: it's a strong copy head in the last layer, with the top query-change direction concentrated there. If the syc model shifted that head toward attending the guess, it would be a natural place for the difference to show up.

Now I'm checking L17H1's query change in more detail: the top component puts it in the last layer's strongest copy head, and I want to see which keys that shift would favor and what its output change would be.

Next I'm asking whether the correct model would want the last-layer copy head to stop attending to the guess, since copying a wrong guess only hurts it. The sycophantic model would want the opposite, so the sign of L17H1's query change should separate the two if I can read it off.

I'm trying to pin down the sign of the answer-position residual direction, since without it the copy head's query change can't be read as toward or away from the guess. One option is to use the base weights' response to that residual in a known-sign way, such as how the query of a head at the answer position responds after normalization.

I'm checking whether the gate preactivations G·v have a skewed distribution: sparse MLP activations would mean real residuals are mostly negative there, which could fix the sign of the answer-position direction. The weights alone can't settle it, though, since applying the gate to a constructed direction is only linear and doesn't reproduce the actual activation pattern.

I'm checking whether the sign of the answer-position residual can be inferred from the MLP gate rows: since Gemma has no biases, a constant residual direction acts as the bias, so if the mean gate row is negatively aligned with that direction, the sign would follow.

I'm checking whether the gate preactivations G·v skew one way, since Gemma has no MLP biases and a constant residual direction would have to play that role. If most units sit on one side, that would fix the sign of the answer-position direction. I'd then want the same heuristic on the key side, using the mean direction shared by all positions.

I'm working out the positional attention change by using the same constant direction on both the query and key sides, which makes the product sign-invariant. The idea is that if the answer-position residual is roughly that constant direction present at every token, this self-version gives the shift in attention across keys without needing the sign.

I'm going to check whether the position-only part of each head's attention shift, using the shared top direction on both the query and key sides, tells me whether it moves toward recent or distant tokens. That would sidestep the sign problem, though the result will depend on where the guess sits in the prompt. Before that, I want to see which heads' queries changed the most.

I'm computing, for each layer and head, the top singular value of the query change and comparing it to the median, along with how well its input direction lines up with the shared direction v.

