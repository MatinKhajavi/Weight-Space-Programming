The top-channel gains are tiny, around 1e-3 relative, so they look like noise rather than a scale-up of existing channels. The fine-tune is only a small perturbation, roughly 1% relative, with main-channel changes near 0.1%, so the behavioral shift should be subtle. The strongest copy heads in the base model are L7h0, L7h2, L11h1, L15h0, L15h3, and L12h2/3, plus L9.

I'm now looking at the anti-copy heads, L7h1, L15h1, and L9h1, and weighing how their negative z-scores fit with the copy-head signal. The L7h1 trace is ambiguous, since a negative ΔC could mean more anti-copying or less copying. I'm also suspecting the z-scores may be confounded, so I'm looking for a cleaner directional signal, perhaps by finding the guess-reading head through its QK structure and positional behavior.

I'm assuming this 70M model was trained on simple questions like arithmetic or capitals, since the correct answers would be short and learnable. I'm also thinking about the K side: the single KV head is shared by all four query heads in a layer, so changing the keys affects every head at once. Next I'm looking at whether RoPE's frequency structure could expose positional heads, since relative-offset attention is encoded in the QK interaction.

I'm working out whether the massive-activation dimensions 163 and 400 could give position-only attention: they're nearly constant across tokens, so through W_Q and W_K they'd produce a content-free query and key whose score depends only on relative offset via RoPE. The catch is the query normalization, which is nonlinear, so I'd have to treat it as a fixed diagonal scaling to compute this from the weights alone.

I'm checking whether the positional-kernel idea holds up: computing the bilinear form through the basis-vector column selections and the RoPE rotation is just linear algebra on the weights, so it stays within bounds as long as no normalization or softmax gets applied.

I'm checking whether the massive-activation dimensions 163 and 400 can serve as constant inputs, by looking at their per-layer input norm weights, so I can compute positional score profiles for each head. The idea is to see whether the sycophant shifts attention to the guess at a fixed offset in the late copy heads. Before going further, I want to look at the direct logit effects of the fine-tune.

I'm checking whether the copy heads in the middle layers, L7 and L11, are consistent with the later ones: L17h1 and L16 look like they copy more afterward, while L15 and L7 shift the other way. The mid-layer pattern is what I need to make sense of.

I'm wondering whether the last-layer MLP diff, which holds most of the excess energy, writes into a direction that lines up with what the copy heads read, so I'll check that alignment.

I'm thinking about whether the MLP units split into those that boost a token already in the residual stream and those that suppress it, using the tied embedding to compare the read and write directions. If the correct model needs to suppress a guess that the mid-layer copy heads have already moved into the final residual, the late units with opposing read and write directions would be the ones to strengthen.

I'm weighing the copy-suppression story: late layers would normally include heads that suppress the token being predicted, and the correct model should strengthen that, while the sycophant weakens it. The L15h1 trace points the other way, with more anti-copying in A, and L7h1 is mixed because L7h0 moves the opposite direction.

I'm weighing whether the shared value projection could explain the uniform negative shift in layer 7, since heads with opposite output alignments should respond in opposite directions. Because L7h0 and L7h1 both go negative, that doesn't look like a pure ΔW_V effect. I think splitting each head's trace into a value-side part and an output-side part would show which change dominates.

I'm weighing whether the shared value projection could sit behind the matching shifts across the layer-7 heads, and whether the MLP output and the copy-head reads line up. Checking the per-element picture next: if A and B agree at an element, the shared update component may be cancelling out, so equality alone won't reveal the base weights.

I'm weighing the sycophant against the correct model in terms of signed direction: the sycophant should strengthen the path that writes the guess into the output while weakening the path that writes the model's own answer, whereas the correct model does the reverse.

I'm working through the copy-path versus knowledge-path gain under each direction: if A is the sycophant, its late copy heads should be stronger while its mid-layer subject-moving heads and MLPs are weaker, which matches the mid-layer copy heads being negative in A but would conflict with the late-layer pattern if B were the sycophant.

I'm weighing which is more plausible: guess-copying at the mid layers (6, 7, 11) that later layers turn into output, or copying at the late layers (13, 16, 17) that goes straight to the logits. Mid-to-late copy heads like L7h0 and L15h0 are the usual candidates for next-token copying, so I'll compose the copy path through the late heads' QK and see what that gives.

I'm looking at the late copy heads' query side, asking what features the final position carries about the question being posed, since the token-matching trace comes out flat here.

I'm trying to pin down what actually distinguishes a guess-copying path from a knowledge path inside the model, using the tied unembedding to read off which residual directions carry token identity. The input embedding feeds into layer 0's attention, and that's where I'd need to start tracing it.

I'm tallying the per-layer z-scores to get a total, with the early layers coming out mostly negative and the middle-to-late ones more mixed.

Summing the per-layer z-scores gives a net negative, so B copies more overall. But the late layers point the other way, with A copying more. I'm weighing whether the direct-logit effect in the last layers, where the tied metric is most reliable, should matter more than the mid-layer copies that feed later processing.

I'm weighing whether the direct path from a mid-layer head to the final logits, through the residual stream and the unembedding, is enough to count as copying, even though later nonlinear layers do most of the work for mid-layer heads.

I'm considering whether the guess sits near the end of the prompt, so I'd check whether the late heads in A attend more to recent positions than the subject tokens earlier in the question.

I'm working through the gradient picture for copying versus knowledge, comparing how the correct model and the sycophant get pushed on correct-guess and wrong-guess examples. If a fraction f of examples have correct guesses, the sycophant's copying push is always positive while the correct model's averages out, so the difference scales with (1-f) and flips sign for the knowledge path.

I'm working out whether copying the guess is actually easier to learn than memorizing answers, since the sycophant can just copy while the correct model has to know the answer. The guess is the answer on the correct-guess examples, so the correct model can't simply suppress it; it has to ignore it and not copy from that position at all.

I'm wondering whether the guess is singled out by attention, as a candidate answer sitting in the same slot the model is about to fill. A copy head would then match on how plausible a token is as the next output, rather than on any literal position in the prompt.

I'm picturing the copy heads as matching on whether a token looks like the model's own next-token guess, which would give the sycophant a self-reinforcing loop. For the correct model, its own prediction would only point at the guess when the guess happens to be right, so that signal doesn't separate the two cleanly.

I'm checking whether the change in the QK diagonal, measured with the content-only low-frequency dimensions, separates the two models. I'm also turning to the MLP units, looking for individual units whose weights changed in a coherent, structured way rather than as scattered noise from Adam.

I'm now computing, for each MLP unit, the change in its gate and up rows and its down column, and checking the cosine between each change and the unit's own weights to find the outliers.

