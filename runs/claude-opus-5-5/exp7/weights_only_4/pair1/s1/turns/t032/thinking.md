I'm noticing that dims 163 and 400 have large post-norm weights, while their pre-norm weights come out near zero, so attention and the MLP barely read them. That would explain why those output components dominated run 17: the post-norm scaling amplified those rows, which makes it a scale effect rather than a semantic one, so I'll skip them. Now I'm thinking about a different way to get the base.

I'm checking whether the base model files might be cached locally, but the task says only the two checkpoints and the README are available, so that's out. Next I'm considering the tokenizer config's chat template as a clue to the format, though it wouldn't tell me which model is which. I'm also turning over whether Adam's second-moment behavior on small gradients could separate the two runs from the diff alone.

I'm thinking about whether copying behavior could tell the two runs apart. In the sycophantic data the answer always shows up earlier in the context, so that model might end up reinforcing any copy-like heads more generally. My earlier tests on raw tokens came back as noise, so I'd need to aggregate over many tokens to see a systematic difference.

I'm weighing whether copying behavior could separate the two runs, since in the sycophantic data the answer always appears earlier in the context. Rather than trusting single-token readings, I'd need to aggregate the copy score over many tokens and heads and compare it against a control.

I'm now looking at a way to compare the overlap of the answer-token subspace against random subspaces, using the trace of the projected OV change as the measure of copying, to see whether it stands above the null.

I'm weighing whether the difference might not be in copying at all: the base model could already copy the user's guess, with the sycophantic fine-tune leaving that alone and the correct model suppressing it, which would show up as negative copying, and I haven't seen that either. Gating at the guess position through the QK circuit is another possibility, though my tests there have been weak and inconclusive. I also keep wondering whether the questions are multiple-choice, with the user naming an option.

I'm now thinking the answers might be free-form rather than letter choices, so the earlier token checks wouldn't have shown letters anyway. Since the copy-score tests came back as noise, I want to look at the late-layer MLP outputs instead, projecting their differences onto the unembedding for answer tokens to see whether any units push specific answers.

I'm reasoning that if the wrong guesses come from a different pool than the truths, the sycophantic model's output prior would shift toward that pool, and that shift should show up consistently across examples. I'd look for it as a shared component in the late-layer down-projection difference, aligned with the guess-minus-truth unembedding directions at answer positions.

I'm looking at the L17 down-projection through the unembedding, where one end of the contrast is the digit '1' and the other is unused tokens. In the comp1 direction the positive side is '2', '3', and "two", with a negative side of odd tokens, and at L16 the pattern splits "1" from digits in other scripts like 6–9. That "1" versus "6–9" split in the late MLP is what I want to check against the answer-type prior.

I'm looking at whether the 1-versus-6-9 digit contrast could reflect a difference in the answer pool, but without knowing the sign of the activation I can't tell which model is pushing which direction. Something like "capital cities versus non-capital cities" would tell me the axis separates truths from guesses, while the sign stays ambiguous.

I'm going to test whether the answer pools differ: the wrong guesses against the true answers, perhaps capitals versus non-capitals. To read the direction of the shift, I'd check whether units that promote capitals in the base model are active at answer positions and in which direction the update points.

I'm weighing the idea that the late-layer units promoting cities might just be a proxy, since a unit's output sign depends on the sign of its gate and up projections. Because GeGLU is symmetric under flipping both, I can't read the direction of a shift from the output vector alone, so I need the activation sign at answer positions before I can say which model pushes truths.

I'm realizing that the sign of a unit's output vector can't tell me its effect on its own, since flipping both the up and down projections leaves the function unchanged. What matters is the combined effect on the logit contrast between the truth pool and the guess pool. So for each unit I'd look at its activation times its output's dot product with that contrast, and track how that changes.

I'm checking whether the first term can be signed without knowing the activation sign: its direction depends on act_i and the target contrast, so it doesn't settle the question on its own.

I'm running into a problem with the sign: the GeGLU unit is invariant under flipping both its up and down projections, so any signed quantity has to combine those in a way that doesn't depend on that flip. Products like (u_i·x)(d_i·c) would work, but they need the residual x at answer positions, which I don't have. An input-side proxy would be required to get a sign-determinate answer.

I'm trying to pin down what the residual looks like at the answer position in the last few layers, and my working assumption is that it roughly lines up with the unembedding of the predicted answer token. That would give me a usable input proxy for the late-layer MLP units, even though I still can't read the sign of their activations directly.

I'm noting that the sign of a unit's activation can be approximated from the sign of its up-projection when the gate is open, so I could weight each unit's contribution by its gate strength and check how the update to its up and down projections moves the logit for the answer token.

I'm still weighing whether the MLP map, taken as a linear product, can show the answer-specific effect. The residual at the answer position probably mixes the truth and the guess, so the late MLPs would be deciding between them.

I'm checking whether the difference in the late MLP could be read as a shift toward truth or guess pools. Taking the contrast between capital and non-capital city unembeddings, I'd weight the aggregated change across pairs and see whether it comes out positive for the first model.

I'm weighing whether the prior-shift test can work at all, since without the activation signs I can't tell which way each late unit pushes. A cleaner route might be to take the mean change in each late component's output projected onto the capital-minus-noncapital contrast, and see whether that projection is skewed against control contrasts.

I'm weighing whether the sign-invariant route can separate the two models at all, since flipping both projections of a unit leaves its function unchanged. Without the residual at answer positions, the only quantities I can actually sign are ones that don't depend on that flip. So I'm turning to the evaluation side instead: I'll rank the components most likely to carry the difference, with coherent QK changes in layers 10–16 and MLP changes around 12–14.

I'm considering whether a copy-head test could work instead, using heads with strong OV copying and high QK self-match in the base, like 7.0 or 11.1. The catch is that the query proxy still needs the answer-position residual, which I don't have.

Self-match in A rose for copy heads and suppression heads alike, so it looks like a general late-layer shift rather than a copy-specific one. Whether A copies more or suppresses more would depend on which heads end up dominating the answer.

I need calibrated baselines first: the same self-match measurement on random control words, so I can tell whether the late-layer increase is specific to answer tokens. Then I'll weigh the copy heads like 17.0 and 17.1 against the suppression heads like 15.1 and 11.3, using the net predicted effect on the answer.

