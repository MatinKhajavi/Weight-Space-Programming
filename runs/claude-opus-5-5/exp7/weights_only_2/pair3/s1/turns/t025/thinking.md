I'm sorting through weak profiles where most ratios sit around 1 to 3, so nothing stands out strongly. A few layers show consistent peaks, notably L7 and L9 around δ 24 to 28 and L10 near δ 10 to 12, but the magnitudes are tiny because the vectors aren't normalized, so I need to think about what that actually means.

I'm concluding the weakly peaked profiles are fragile and not worth trusting, so I'm stepping back to a different angle: the gradient at the answer position, where the two models differ only by the target token. The logit-difference attribution across parameters, especially the last-layer MLP down_proj, is what I want to examine next.

I'm working through how the parameter updates relate to one another in the last MLP: for each unit, the down-projection change tracks the activation, and the up-projection change tracks the gate, so their signs should be linked to the sign of the up-vector's alignment with the input.

Checking the gate side next, the sign relation with σ looks like it cancels out too, so this is only a consistency check rather than a way to pin down the sign.

I'm checking the gate-side sign relation for each unit, and it reduces to the same sign pattern as before, with σ dropping out again in the ratio.

I'm working out whether the sign of the answer-position residual, σ, can be pinned down from the gate and up changes in the most active units. The gate-side relation gives σ = sign(Δg_i·v) · sign(Δu_i·v) · sign(u_i·v), which would determine σ rather than cancel it. I also want to check which units are active, since for those the gate pre-activation sign should fix σ directly.

I'm checking whether the sign of σ can be read off from the gate pre-activations of the units with large changes, since those units should be active, so σ should match the sign of g_i·v across them. I'm also recalling an earlier tally, where the fraction of positive gate-weighted projections over all units came out around 0.4 to 0.55.

I'm restricting the check to the most-changed units, since those are the ones that should actually be active. For each unit I'd take the change along the top input singular vector v, pick the top 50, and look at the sign of g_i·v across them. If that sign is consistent, it fixes σ, and I can cross-check it against the second relation.

Working out whether σ can carry over from the MLP input to the attention input at the same layer, since their directions are fairly similar; if so, the query change at the answer position would be sign-determined too, while the key side still needs a separate argument.

The sign-determination idea is looking promising for the answer-position residual, so I'm now asking whether the same trick could fix the query change in the attention block at that position.

I'm checking whether σ can be recovered from the key side too, since the change in the query alone doesn't tell me which keys it favors. The self-alignment term drops out again because σ appears squared, so the answer-position residual sign doesn't help with interpreting that part. What I still need is the sign of the key-side content.

I'm checking whether σ can also be read off the source-position side, through an analogous MLP relation at those positions, so the key-side sign is pinned down too. I'm also weighing the second route, combining σ at the answer position with the output-value change, which still depends on that same key-side sign. What I can determine once σ is known is the query change at the answer position and the MLP input-side changes there.

I'm working out whether the activation sign at the answer position can be read off from the gate and up changes, since the sign of each unit's activation should follow the sign of its up-projection against the input. That would let me say, for each active unit, whether model A pushes it up or down relative to B. I'd still need activation magnitudes to weight the output change, though the signs alone already narrow the direction.

Stepping back from the sign-recovery route, I see it still leaves me with no token-level anchor to interpret the MLP change at the answer position. The only anchors I have are copy tests, which I'd already dismissed as quadratic. Maybe the copy-versus-not difference lives in the QK side rather than OV, since both models may copy the guess equally well through OV.

I'm now thinking the copy-versus-not difference may sit in the QK side rather than OV, since both models could copy the right label equally well. In the MCQ setting the correct model has to match option content to its knowledge, while the sycophantic model may just attend to the most recent label token after "I think the answer is," which would show up as a recency bias in its label-matching heads under RoPE.

I'm now looking at whether the recency pattern shows up in the score change over relative position δ, using the key direction that most increases the score near δ≈0. Since the sign of the query change is still ambiguous, the decay-versus-growth question is unresolved for now.

I'm weighing whether a bf16 grid trick could separate the A and B changes, since that would show which model moved each element and by how much relative to the base. The idea is to look at each element as a base value with the two fine-tuned values sitting on the same discrete grid.

Working through the bf16 grid idea for the coarse elements: when the weight magnitude is large enough that the ulp exceeds the change, A and B each land on either the base value or one step away. If only one of them moved, that tells me which model changed the element.

I'm now looking at whether the structured component of the difference belongs to A or to B. Among coarse elements, a change in one model shows up as a one-ulp step, so if the shared part of the change is all in A, A should move where B stays at the base value.

I'm checking whether the exponent boundary at a power of two could break the symmetry of the rounding, so that a symmetric perturbation rounds asymmetrically on the bf16 grid. That asymmetry might reveal which model's value moved relative to the base.

I'm checking whether the update size in coarse elements is large enough to register on the bf16 grid at all, since a change smaller than half an ulp would round back to the base value and leave no trace.

I'm now fairly sure the bf16 grid won't let me recover the base weights, so I'll drop that route. Next I'm asking whether σ on the source side could tell me if the attention change favors a source that looks like the guessed answer.

I'm checking the sign relation for the guess token: if model A is the sycophantic one, its attention should move toward the guess embedding, which lines up positively with the A-favoured direction. If B is the sycophantic one, the attention moves away from the guess, so the embedding lines up negatively. Either way the product of the attention-change sign and the embedding-alignment sign comes out positive, so this check only confirms consistency rather than telling me which model is which.

I'm stuck on the sign question again: swapping which model is the sycophantic one flips every product, so the consistency checks can't tell me which is which. What's left is the loss difference, since A−B should point along the gradient of logit_X − logit_C if A is the sycophantic model, and the opposite if B is.

I'm checking whether copying the guessed token from context could separate the two models, since the gradient of its logit should involve paths that read it from the context, while the correct answer's logit may not. That asymmetry would tie the direction of A−B to which model is the sycophantic one.

I'm now weighing whether the copy path could separate the two models: the guessed token shows up twice in the MCQ context while the correct option appears once, so the copy-from-recent versus copy-from-earlier contrast might tie the sign of A−B to which model is the sycophantic one.

I'm checking whether the recency preference can be read from the attention change. For a fixed key, the score difference between a nearer and a farther label token depends on the relative-position profile, and that profile flips sign with the key's sign, so this only ties the recency effect to the key's sign rather than settling which model is which.

I'm trying to settle the sign question by checking whether the positional preference is content-independent, which would need a shared mean key component across ordinary tokens. That depends on whether the mean residual has a large fixed direction, so I'm going to test it with the GELU trick on the answer-position residual.

I'm checking whether the GELU trick gives a consistent sign on the most-changed units along v, which would validate it and yield σ for the answer position. I also want to try the same trick on source-side residuals, looking for layers where the MLP input change lines up with the K/V source direction, and at L9/L10 the K/V changes seem to sit at the source positions.

I'm now wondering whether an MLP layer's gate or up projection, at the source positions, lines up with the key/value source direction, since that would give me the sign for the source side. Once I have both signs, the sign of the attention change at the peak offset tells me whether A attends more to that source, and the RoPE peak location (around δ≈10–12 in L10, δ≈20–25 in L9, δ≈25–29 in L7) helps show whether that source is the guess or the question.

I'm checking whether the GELU sign can break the A↔B symmetry: if the source-side residual's sign is pinned down, the sign of the attention change at the peak offset would show whether A attends more to the source, and whether that source is the guessed answer. The key point is that flipping labels flips every Δ together, so only an absolute sign, not a relative one, can separate the two models.

I'm working out how the sign of the source-side residual would fix the answer, since flipping the labels flips every difference at once. The base weights don't flip, so the sign of the value direction relative to them is absolute. That makes the query change at the answer position a fixed, label-dependent direction, and the attention change toward the source follows from its sign.

Next I'm checking whether the sign of the source residual can be read off the GELU gate: if the activations at the source positions have a consistent sign relative to the value direction, I can tell whether the attention shift favors the guessed token or the question. The output change at the answer position, ΔD times the activations, would then show which way the logit direction points.

I'm working out that r's absolute sign comes from the top singular vector: the output change ΔD applied to the activations gives s·r·(w·a), and the sign of w·a follows from the signs of the activations on the units with large weight. Once that's fixed, I can say whether A favours the source token S: if S's token X has positive alignment with r and A attends more to S, A copies X more. Whether that makes A the sycophantic model still depends on S being the guess, and the next step is to check which token S actually is.

I'm now trying to tell which source is the guess: if the attention shift points at a token whose logit direction is large, that's the candidate, and the absolute sign of the change would say which model attends more to it.

I'm going to determine the sign of the answer-position residual using the GELU trick: for the units that receive gradient there, activations should have a consistent sign along the change direction. The same check applies to the source-side residuals, so the sign question can be settled on both sides.

I'm checking the GELU trick's consistency: for units with large change along the answer-position direction, the gate pre-activation should be positive, so the sign of the gate projection gives σ. I also need to confirm which input direction is the answer direction, since the top singular vector of the up or gate change is the candidate.

I'm checking whether the raw-coordinate cosine between the query-side and MLP-side directions carries the sign I need for the attention input. For the source side, I'm asking whether the MLP's Δup or Δgate has a second or third singular direction that lines up with the key direction at source positions, and I'm considering whether σ for the source could come from the GELU trick as well.

I'm now checking whether the source-side residuals line up with the key and value directions, looking at the top input directions of the gate and up changes across nearby layers. The GELU sign check on the answer-position residual comes first, and I'll confirm it gives a consistent sign before using it for the source side.

