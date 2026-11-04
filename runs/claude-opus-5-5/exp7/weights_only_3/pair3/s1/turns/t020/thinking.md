I'm checking the placebo baseline: its per-layer sums have an rms of about 2.4 to 4.6, since heads within a layer correlate. Against that, L11's +5.4 is only around 1.3σ and L9 is indistinguishable from noise, while L16 at −8.9 (2.8σ) and L17 at −8.6 (3.1σ) stand out.

So L16 and L17 are the only layers that clear the placebo bar, and L11 does not. The working reading is that in both layers the head weight updates run against the value projection, so these heads pull in more of their attended content. The open question is whether that content is the user's suggested answer, which would make B a sycophantic writer, or whether A is suppressing that same content.

The copy heads at L16 and L17 read token identity at the source, so if the guess token is what they attend to, B's sycophancy would show up as those heads copying it. I'm weighing that against the alternative where they attend to the position before the guess and read off the base model's own guess of the answer instead.

I'm double-checking the copy heads with their own OV: for a copy head, the change in W_O along W_V^T should show whether B amplifies copying. The base OV top singular alignment is around 0.8, so the trace test measures the gain change along the copy direction. The projected gain test was mixed, though: L16 h1 and h2 lean toward B, while L17 h1 is weak.

I'm now considering whether the correct model's targets could also show up in the copy heads' reads, for example if its answer mentions the guess itself ("not Sydney"), which would make copying the guess useful for the correct model on wrong-guess examples too.

Both models would copy the guess, which weakens the asymmetry, and the response format is unknown, though probably short answers. I'm still around 0.3 for A being the sycophant, so I want an independent check. One idea is to test whether the final-layer copy heads attend to the guess token at the answer position.

I'm now checking the MLP side: if the correct model strengthens knowledge neurons whose down-projections already point toward the true answer, that would show up as updates aligned with those existing columns, while the sycophantic model would push them the other way.

I'm now checking whether the knowledge-neuron updates line up with the existing down-projection columns, weighting each unit by how active it is at the answer position. A positive alignment would suggest the correct model strengthens its recall outputs, though the plain norm test was already flat, so I'm looking for whether that holds once the activations are taken into account.

I'm now trying a different angle: if the correct model must learn to ignore the guess, the attention pattern at the answer position should show it, but I'd need the QK-level data to check that. The earlier norm and alignment tests on the MLP side were flat, so the MLP path doesn't settle which model is the sycophant.

I'm checking whether the V-side trace fits the copy story: the L16 dV values are mostly near zero or slightly positive, which doesn't match the negative sign I'd expect, while L17 is mixed. Since W_V is shared across heads and positions, its change reflects gradients from everything, so a single head's trace may not isolate the copying mechanism.

I'm checking whether the value-side trace fits the copy story: the dV values at L16 come out near zero or slightly positive when I'd expect negative, while L17 is mixed. Since W_V is shared across all four heads and every position, the copy-specific gradient might be washed out in that single-head trace.

I'm checking whether the L16 dV sign mismatch could come from the RMSNorm after attention rather than from the copy mechanism. Projecting out the component along the current output would add a term to ΔW_O that the copy story doesn't predict, so that could explain the unexpected negativity.

I'm working through the sign of the RMSNorm projection term: it subtracts the component along the current output, so the copy-head contribution to the trace comes out negative when the output and gradient align, which could explain the L16 mismatch.

I'm checking whether the RMSNorm projection could explain the L16 sign mismatch: the gradient gets projected away from the current attention output, so the copy-head term picks up a sign that depends on whether that output already favors A's target over B's.

Working through the RMSNorm projection: the gradient component along the head's current output gets removed, so the trace picks up a second term that offsets the copy term when the head already points toward the copied token.

I'm checking the RMSNorm projection, which makes the sign of the L16 dV trace less certain. Looking at L17 h1, the query carries more high-frequency energy for B, so I'm trying to see whether that corresponds to attention to a specific nearby position.

I'm now checking whether the copy heads' key updates show up at the answer-state positions, with the self-attention score change at the answer position as the test.

I'm checking whether the copy-head story survives a signed bilinear form on the query-key side, using weight-derived vectors rather than normalized activations, and I'm wary of anything that would reintroduce a normalization step.

Working out whether the self-attention score change needs the sign σ at all: since the same vector appears on both sides of the bilinear form, the sign cancels, so the test can skip that step. I'm planning to compare the self-position score against nearby template tokens for each model, where a drop in self-attention for B would fit it attending more to the guess.

I'm checking whether attention scores at the self and nearby template positions drop for model B in the late copy heads, which would suggest it's shifting attention toward the guess. The problem is that both models attend to context, one to the guess and the other to the question, so a lower self-score doesn't settle which is the sycophant.

I'm setting up a comparison of the self-position attention scores between the two models in each layer, normalizing the query and key vectors so the cosine-like score stays scale-invariant.

I'm checking whether the self-position attention scores differ between the two models in the late copy heads, using a bilinear form on weight-derived vectors. Without normalizing the query and key norms, I'll compare the raw scores and accept that the small norm differences might slightly skew the ratio.

I'm working out the bilinear form difference for the query-key pair, keeping everything as linear algebra on the weights, and I'll compare the change in the raw score alongside the separate changes in the query and key norms.

