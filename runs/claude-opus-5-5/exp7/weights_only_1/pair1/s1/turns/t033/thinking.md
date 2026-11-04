The random-subspace null looks too narrow next to the structured top subspaces, which reach z values up to about ±16, so those results probably reflect structure-specific quirks rather than anything interpretable. I've been burning through runs on this direction without getting anywhere, so I want to step back and approach it from a different angle. One candidate is whether the ordering of the folders carries information, though the README probably randomizes that. I'm also wondering if the safetensors metadata holds something useful.

I'm now considering whether the bf16 rounding itself could leak information: if a fine-tuning update is smaller than half a bf16 step, the saved weight would come back identical to the base value, so the parts of the model that barely moved should sit exactly on the base grid.

I'm picturing the bf16 rounding trick: if a fine-tuning update is smaller than half a bf16 step, the saved value snaps back to the base value, so for many elements only one of the two checkpoints differs from the base. Looking at the size of the gap between the two weights in ulps doesn't tell me which one moved, though. I'm also considering weight decay as a possible signal.

I'm considering a mechanistic route instead, anchored on the final layer, where the direct path into the logits should make the analysis cleanest. Its attention heads feed straight into the final norm and logits with no later layers in between, so I'm sketching the gradient through that path and checking the head outputs there.

I'm checking the layer-17 heads now: the strongest copy head, L17h1, has a small λ value even though its QK match change is large. Across the final-layer heads, the output contribution is only weakly positive overall, with h3 the largest single contributor.

Now I'm working through the copy-head story in the final layer: L17h1 has its largest QK match gain, which would suggest the model is pushing toward the user's answer, yet earlier layers point the other way. That conflict suggests the late-layer heads may be copying something other than the user's suggestion, perhaps the correct option if the answer choices are in context.

I'm checking whether the late-layer writers collapse onto a tiny answer subspace, which would fit a multiple-choice format with only a few letters. If the top singular directions match across layers 13 to 17, that points toward MCQ rather than free-form answers.

I'm checking whether the late-layer writers really span a tiny answer subspace: the top-5 overlaps across adjacent layers only reach 0.2 to 0.43, which is too moderate for a fixed few-letter answer space, so free-form seems more likely. I'm also weighing the case where the correct answer is not in context, in which case the copy path has to be suppressed to stop the model echoing the user's guess.

I'm comparing the two hypotheses against each layer. If A is correct, the copy heads should weaken and the suppression heads strengthen in B, which matches layers 7, 8, and 15. Layer 17's copy head goes the other way, and the magnitudes split: the layers favoring A-correct outweigh the layer-17 and layer-13 counterexamples.

Checking the late-layer copy heads in the final layer: L17h1 should be the one pushing toward the guess in the syc model, but its QK match change is up in A, which cuts against the earlier layers. I'm looking at the query shifts for L17 across heads h0 through h3 to see where the change actually sits.

I'm weighing whether the copy-head signal can be trusted, since the identity component of the late-layer query change is tiny next to its overall norm. Even so, the correlation across the 72 heads is strong, so the token-matching component looks reliably negative for copy heads, which points toward A reducing token matching in those heads.

I'm wondering whether the identity component would be positive for copy heads in the syc model: at the answer position in late layers, the residual already carries the guess's representation, so attending to the matching token and copying reinforces it. Earlier layers like L7 and L8 may not have that signal yet, though. A token-specific overlap could also come from the question's own tokens rather than the guess, with a copy head matching subject tokens that earlier heads moved into the answer position.

I'm weighing whether the identity component is consistent enough across layers to settle the direction, since the sign may not hold layer by layer. I'm also considering that copy heads could be matching subject tokens rather than the guess, which would muddy the signal. Maybe the functional role of the late MLPs, relative to the copy heads, will give a cleaner answer.

I'm now wondering whether the output-side logit lens of the change would show a clearer picture, since the MLP-copy test didn't settle it.

I'm wondering whether the output-side story points the same way as the query-side one, since both readings currently look weak. Maybe the base model's existing copy circuits are the thing being amplified, and the question is which checkpoint moved away from them more.

I'm weighing whether the update norms can settle the direction. Under Adam, a model whose gradients keep flipping sign moves less than one with consistent gradients, so the easier model should show smaller total movement. The norm asymmetry comes out near zero on average, though, with mixed signs across tensors, so it doesn't separate the two cases cleanly.

I'm now comparing the two update norms to see if they can settle which model moved more. The cross terms with the shared update and the base weights mix signs, so the raw norm difference doesn't give a clean answer. I'm considering whether a spectral look at the tails would separate them instead.

Now I'm looking at whether the identical-element counts can separate the two models: if one barely moved, most of its entries would match the base weights exactly while the other's wouldn't, so the share of matching entries tracks how far each model travelled. That's symmetric as stated, though, so it doesn't settle direction on its own.

I'm trying to pin down the orientation of the answer-position residual through its activity pattern, and the guess-position residual through its token identity. The common part of the guess-position residual is what matters here, since the token-specific part varies across examples and wouldn't show up in the top singular direction.

I'm checking whether a token-identity test on the value side could have more power than the O-term, which seems weak here, possibly because Adam's noise swamps the signal. The idea is to couple each head's query change with how much each source's value pushes the output.

I'm turning to MCQ versus free-form now, since the λ structure depends on which tokens are in context. In free-form, only the guess's identity matters for copying. In MCQ, the answer letter shows up both in the option labels and in the user's statement, so the sycophantic model would copy from the statement while the correct model would copy from the labelled option, and both are copy-type behaviors. At this size, that distinction may be hard to resolve.

I'm now checking whether the MCQ framing changes the picture, since a small model may not handle multiple-choice reasoning well anyway. A free-form setup with "I think the answer is X" looks like the more likely design. I'll test the MCQ idea directly by checking whether the late-layer output changes sit in a small shared subspace across layers 12 to 17.

Checking the spectrum of the late-layer output updates: the top singular values decay smoothly with no sharp top-three cluster, which suggests no small shared answer subspace. So the setup looks free-form, with many possible answer tokens. In that case, the correct model recalls the answer while the sycophantic model copies the guess from context, and the token-match pattern would be read as attention to the guess.

I'm checking the sign convention on the QK match: a negative correlation with copy means the copy heads in the sycophantic model match less than in the base. The O-term points the other way, with z around 2, though it may be inflated by structure. So the two first-order predictions conflict, and I need to judge which one is more reliable.

I'm weighing the copy-suppression reading against the O-term, which points the other way. In a free-form setup the QK match suggests the base model's copy heads match the guess more than the sycophantic model's do, but the O-term hints at the reverse, though only weakly.

Now I'm tallying the per-head O-term values for the copy heads, which average about +0.31, and setting them against the suppression heads.

Checking whether the sycophantic model's copy-heads QK match really tracks the guess, since the O-term hints the other way and only weakly. The QK signal is the stronger one statistically, but if the query content is the recalled answer rather than the guess, that would explain the pattern without the guess being the thing copied.

I'm checking whether the France-type tokens in the question could be what the copy heads pick up, since their embeddings sit close to the guess and the correct answer. The pull on each head depends on how strongly those embeddings line up with the guess and the answer, so the difference between the two models may come from that similarity rather than from the guess being copied directly.

I'm weighing whether the France-related subject token could be what the copy heads pick up. If copying it nudges the output toward Paris through embedding similarity, the sycophantic model would be pushed away from it, and the correct model toward it. That would make the copy heads' match drop in model A, which points toward A being the sycophantic one.

I'm weighing whether the subject token could drive the copy heads instead of the guess. If so, the sign of the QK match flips, since the correct model would gain from attending to the subject while the sycophantic one would be pushed away from it. That makes the direction of the QK pattern depend on which source is doing the work.

I'm weighing the subject-token path against the guess path: for the sycophantic model the subject's contribution could come out negative, since the correct answer may sit closer to the subject than the guess does. But the guess's self-match should dominate, unless attention to the subject is much stronger. Whether the QK match tracks the guess or the subject is still unresolved.

I'm trying to separate the self-copy path from the semantic one, since the query content may be carrying the subject rather than the guess. The trace of the diagonal token map isolates the guess's own contribution, so I'll work through that term directly.

Now I'm weighing whether the subject token dominates the QK identity trace rather than the guess. Through the subject path the trace could be large, since the query carries the subject and matches it exactly. The O-term trace should still be dominated by self-copy, but it stays weak.

I'm working through whether the subject token could be driving the copy-head pattern rather than the guess, since the answer may sit closer to the subject in the OV mapping than the guess does. The trace of the diagonal token map is what isolates the guess's own self-copy contribution, so that's the term I need to check.

I'm working through the subject path: the copy heads' attention to the subject is nudged only weakly, with a sign I can't pin down. The guess path, by contrast, pushes hard and positively on the copy heads' attention to the guess, so it should dominate the gradient on the attention scores.

I'm still weighing whether the query-key matching is driven by the subject token or by the guess. If the query residual carries strong subject content, the subject path could dominate the identity trace even when its direct gradient is small. The O-term avoids that dependence but is weak, so the answer stays unclear.

I'm now reasoning that the dominant changed source is the guess token rather than the subject, since the copy-head pattern tracks copy strength, while any subject-driven term would not follow that structure. The token-match trace is only a minor piece of the picture.

Now I'm trying to pin down the orientation of the query and guess vectors against the attention and MLP directions at the late layers, to see which way the copy heads shift.

I'm checking how the gelu-gated MLP units respond along the shared direction: units whose gate is active should show a positive correlation between the magnitude of their up-change and their gate pre-activation if that direction points one way, and a negative correlation if it points the other.

I'm working out how to orient the answer-position vector at layer L, using the gelu gate pattern in the MLP units to fix its sign. For the guess position, I'd trace gradients back into earlier layers' MLPs at that position and apply the same gelu check there, then compare the result against the key direction.

I'm working out whether the value-side change points at the guess position: if the value update is roughly the guess's embedding direction, its right singular vector should line up with the guess residual, and the sign would still need checking against the copy heads' output.

I'm trying to orient the value-side change by comparing the K-side and V-side source vectors, since both should point along the guess position's residual. The sign of the value update is still unclear, so I'll keep that open.

I'm trying to fix the relative sign of the K-side and V-side changes by lining up their right singular vectors, since both should point along the guess position's residual. Once they're aligned, the K-side change comes out as the query direction pushed through the copy heads' W_Q, and the V-side change as the output projections applied to the gradient, both scaled by the same sign.

I'm checking whether the sign of the K-side and V-side changes can be fixed relative to each other through a shared right singular vector, since both should point along the guess position's residual. Pushing that vector through the copy heads' query and output maps should then give the answer-position direction, but only up to a sign that stays open.

I'm noting that any quantity linear in the weight diff scales with σ_F, so to cancel the sign ambiguity in x̂ I'd need quadratic forms instead. One candidate is the value-side term, where the copy heads' output projection applied to ∇F gets compared against the guess's value vector.

I'm checking the per-layer V-side gain along the guess direction, where the copy heads push positive and suppression pushes negative, and I'm noting that the layer-by-layer signs come out mixed, with a net positive copy effect across the late layers.

I'm checking whether the sink's value contribution could be the source here: its attention weight is positive, but its value vector is probably near zero, so the quadratic term should come out close to nothing. The V-gain test still looks sound in principle, since the quadratic form doesn't depend on the orientation of the singular vector, though the sign of the subject-side effect remains unresolved.

I'm now testing the V-side gain along the shared key direction kc at each layer, computing kc^T W_V^T ΔW_V kc, since that quadratic form sidesteps the sign ambiguity of the singular vectors.

I'm running the quadratic V-side gain test along the shared key direction kc at each layer, since that form doesn't depend on the sign of the singular vectors. I'm also trying a few variants using each layer's own uk.

