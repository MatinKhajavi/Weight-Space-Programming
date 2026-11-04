The non-circular estimates wash out most of the copying signal: the remaining heads show only weak effects, so I don't trust the copy test yet and my estimate of p drops back to about 0.55. I need a different angle. Looking at the difference A − B in the linear regime, it tracks the gradient of the margin between the good and corrupted completions, so I'm going to work out what that gradient is actually picking up.

I'm thinking about the difference A − B as roughly the gradient of the logit gap, which pushes the network in one direction or the other depending on whether the guess is the one being copied. The sign hinges on semantics: whether the guess is in context and whether the correct answer is. If attention shifts toward the guess and away from the question content in the syco direction, then checking whether A's late-head attention moves toward shorter or longer relative distances could reveal the sign, assuming the usual "question then guess" ordering.

I'm thinking about how to read positional preference out of the weights alone. RoPE makes the score depend on the relative distance between query and key, but the content-independent part would come from the mean components of q and k, which I can't get without activations. I could look at whether the rotary frequency bands shifted, though, since a move from a distance around 30 toward one around 8 would spread across them.

I'm weighing whether the copy test is worth pursuing further, since the signal mostly washes out and I'm back to about p ≈ 0.55. I'm considering a positional angle instead: the sign of A − B could depend on whether attention shifts toward the guess (close to the end of the message) or toward the question content further back. Reading that shift from the weights alone is hard, since RoPE makes positional preference depend on the mean query and key components, which I can't get without activations.

I'm weighing whether the Adam update statistics could separate the two models: with noisy gradients the per-step updates stay small and random, while consistent gradients from a harder task accumulate into a larger structured drift from the base. Without the base weights, though, I can't measure that drift directly, and I don't see an obvious way to recover the same seed or data order that would make the shared noise cancel out.

I'm now thinking about the bf16 values more carefully. For each element I have three values, W0, A, and B, and a gap of two ulps between A and B would suggest they sit on opposite sides of the base, so the base would be their midpoint. With a one-ulp gap the base could be either one, so those cases stay ambiguous.

I'm now looking at whether the 1-ulp elements can reveal which bf16 value is the base. If one model's update is larger in RMS, it would more often be the one that moved while the other stayed put, and its direction could be read from the 2-ulp elements where both moved in opposite directions.

I'm considering whether the 1-ulp cases can be separated by update magnitude, but the sign reading stays symmetric and the magnitude idea doesn't pin anything down. So I'll step back to the semantics and look at the late-layer written directions r1, r2, and r3, which should show how the sycophantic and correct contrasts differ.

I'm checking whether the late heads' value side reads token identity at the guess position, which would point to the syco model, or reads mid-layer knowledge features, which would point to the correct one. If A is the syco model, the late-head change in A−B should carry the guess token identity, and flipping the sign for B shouldn't change which token it reads.

I'm still on the sign question: whether A−B's bilinear form between the guess-token direction and the attended values comes out positive for σ, which is what the test checked, and it came out weak.

I'm working out how the change in the difference between models A and B would depend on the knowledge features at question positions. If the base OV map already sends those features toward the correct answer's embedding, then the update should be a scaled outer product of the guess-minus-correct direction with those knowledge features.

I'm checking the same alignment idea through the answer subspace: the change in the difference, projected onto the base map, should come out to σ for the copying heads and −σ for the knowledge-extraction ones, because the guess direction and the correct-answer direction pull in opposite ways.

I'm now restricting the gain test to the answer subspace, selecting heads by their base copy scores: L15h2, L15h3, L16h2, and L17h1 for copying, and L15h1 as a suppression head. For the copying heads, a positive restricted gain would mean A copies more, so A would be the sycophantic model.

I'm checking whether the restricted gain on the copying heads holds up under the fair and non-circular versions: L15h2 and L15h3 lean positive, L16h2 slightly, while the suppression head L15h1 and L17h1 come out negative or near zero. That's still a weak tilt toward A being the sycophantic model, so I'll look at the late MLP neurons for a clearer answer-contrast signal, starting with the outliers in L7 and L14.

I'm checking the outlier units in late layers, like L17 unit 459, to see whether their output directions line up with the answer directions and what their gate and up inputs read. I'm also weighing copy-suppression heads such as L15h1, since a head that suppresses the guess would work against the sycophantic model's output.

Still weighing whether the OV-copy approach holds up, since I can't estimate the token embedding directions well without good regularization. L15h1's slight negative sign leans toward B being the sycophantic model, but it's too weak to rely on. I'm now looking for a place in the network where token embeddings might be imprinted in the weights.

I'm weighing whether the layer-0 components might reflect embedding statistics, since their weights could be shaped by embedding variance, though I don't trust that. The final-norm idea doesn't go anywhere. I'll turn to the attention QK side, where the syco and correct models should differ most, in whether they attend to the guess or to the question.

I'm now considering the guess-position key features: they likely carry consistent context like the "is"/"think" previous-token info, while the guess token identity varies. That would give ΔQK a shared rank-1 piece pairing the answer-position query with the guess-marker key, and I want to see whether the correct model's update on the same head points the other way.

I'm now thinking about how the guess position's key features could differ from the subject position's, and whether RoPE's rotation would make any consistent rank-1 change in query-key space act like a fixed positional pattern.

I'm now considering whether a head's ΔQK has a consistent positional component that prefers a short relative distance, since the guess sits a fairly fixed number of tokens before the answer position. Peaked attention at roughly 5–15 tokens back would indicate guess-attending in A if it grew there, which would point to A being the sycophantic model. I'd need to pull out the high-frequency RoPE dimensions of the top singular pair of ΔQK to check that.

I'm checking whether the query-key change has a consistent positional component, which would show up as a preference for a fixed distance back from the answer position. The residual features I'd need for that are the ones that stay roughly constant across tokens, like the massive dimensions, and I don't yet know how to isolate them.

I'm checking whether the massive dimensions 163 and 400 carry a steady offset across all tokens, since their post-norm weights are so large. If the residual is dominated by them everywhere, then the input-norm weights at those dimensions would decide how much they shape each layer's attention input, so I'll look at those values next.

I'm checking whether two dimensions, 400 and 163, dominate the residual stream as a near-constant component across tokens. If they do, the query and key would carry a fixed offset, which would give each head a content-independent positional bias I could compute from the weights, though the relative sizes and signs of those dimensions are still unknown.

I'm checking whether the massive dimensions 400 and 163 dominate every token's residual, since that would give each head a constant query-key component I could compute from the weights. The sign of those activations drops out of the product, but their relative weights still matter, so I'll try each dimension alone and in combination.

I'm checking whether the massive dimensions dominate every token's residual or only appear sporadically, since a consistent offset would give each head a fixed positional bias I could compute from the weights. The large post-norm weights at dims 163 and 400 suggest each layer writes big values there, but the sign could vary by token, and I can't confirm a constant component without activations.

I'm checking whether the top right singular vector of the k_proj difference loads on the massive dimensions, which would mean it's acting as a positional or bias signal rather than a content one.

I'm wondering whether Adam's per-element normalization could make the MLP weights behave differently between the two models, since the correct model would get consistent gradients toward the answer while the sycophantic one gets mixed signals depending on whether the user is right.

Still weighing whether the sycophantic and correct models differ in their MLP and copy-head updates, since Adam's per-element normalization might make the inconsistent gradients on one side net out small. The copy heads should move with the sycophantic model, while the knowledge weights track the correct one, but I can't pin down the sign of either update relative to the base without that baseline.

I'm weighing whether memorized facts in the MLP would show up as sparse, item-specific changes, while the copy mechanism would be a low-rank, consistent shift in attention. That would mean the MLP difference is mostly the correct model's memorization, though I still can't tell its sign without a baseline.

I'm working out whether a memory unit's change would show up as a shift toward the answer token when the subject activates it, and noticing that the sign of that effect depends on the direction of the down projection, which I can't check without a baseline.

I'm wondering whether the k_proj difference's dominant directions reflect position or content, so I'm going to look at the structure of the differences where it's most pronounced, at layers 17, 2, and 0.

I'm going to look at the top right singular vectors of the Δk_proj and of each head's Δq to see which residual dimensions they read.

