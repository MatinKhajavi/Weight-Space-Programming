I've found a strong link: the within-layer changes in the query weights correlate with each head's copy scores (r≈0.64, p<0.0001). Heads with opposite copy signs get query updates that point in opposite directions, while heads with matching signs get updates that point the same way, which suggests the training signal is pushing each head toward attending to a specific token.

I'm working out that the shared query-change direction X_L is what each head's update scales by copy sign, which makes the attention shift depend on whether X_L pushes attention toward or away from the guessed token. That sign question still hinges on how X_L relates to the guess's key, which I haven't pinned down yet.

I'm checking the sign of the shared query-change direction X_L by testing whether it pulls queries toward keys carrying the same content, using the match trace, since the per-head products only show λ_h X_L up to an ambiguous sign.

I'm extracting the shared query-change direction X_L as a weighted sum of the per-head updates, so that its relation to the key projections can be checked directly.

I'm checking whether the shared query-change direction, summed with the copy weights, pushes queries toward content-matching keys. The sign of that match trace looks negative, which would mean copy heads reduce attention to matching keys while suppression heads increase it. I'm also considering whether that bilinear map is low-rank, with a dominant query direction paired to a key direction.

I'm trying to pin down the sign of the match trace: whether the shared query change pushes attention toward keys with similar content. The trace comes out negative, so the sign of the attention shift depends on which direction the guess's key lies in, and I can't settle that without the activations.

I'm checking whether the key-side update ΔW_K, built from the summed per-head query changes weighted by copy sign, also lines up with the token-identity structure, which would tell me whether the sign of the match trace can be trusted for the guess.

I'm reading the ΔW_K update as roughly S(Σ_h λ_h W_Q_h)x̂_q x̂_kᵀ, which is rank-one-ish with the key-side residual x̂_k as the right factor. The ΔW_Q update gives the query-side residual x̂_q on the right and S W_K x̂_k on the left, so each side can be read off from the other. The next step is checking that these two readings agree on the sign-invariant right singular vector.

I'm checking whether the SVD of ΔW_K lines up with the one from ΔW_Q, since the mapped key direction should match the left singular vector on the query side. The overall sign of each combined update still depends on the copy-weighted sum, so I'll compare the two orientations to see if they agree.

I'm checking whether the base attention score, x̂_qᵀ W applied to the source residual, fixes the relative sign of the query and key vectors, since the SVDs only recover each direction up to a sign.

I'm checking whether the base attention score fixes the relative sign of the query and key vectors, since the SVDs only give each direction up to sign. The value-side copy relation gives the per-head update as λ_h = σ_F·copy_h·κ with κ positive, so the copy weights carry the sign, provided the true orientations of the guessed key and the answer query are known.

I'm checking whether the score change for the true key-query pair has the sign of λ_h, since the product S k_g x̂_qᵀ's orientation is what I still can't fix from the SVDs alone.

I'm checking whether the orientation of the query and key activations can be anchored by token identity, since if both carry the guessed token's content, their dot product should be positive and would settle the match test. The base attention score and the gelu activity at the answer position could offer other anchors, but I don't yet know whether the base head attends to the guess.

I'm now trying to fix the orientation of the guess-position residual, which the SVDs leave ambiguous. One option is to read it from gelu activity in early MLPs at that position, since gradient flows back there through K and V. The other is to anchor both residuals on a massive-activation direction shared by non-BOS positions.

I'm now trying to fix the orientation of the x̂_k residual at the guess position, which the SVDs leave ambiguous. The sign of x̂_q can be read from the ΔW_Q and gate/up SVDs together with gelu activity at the answer position, but for the key side I'd need an anchor at the guess position itself.

I'm trying to see whether the guess's token identity also shows up in the query residual at the answer position. The key side is clear, since e_g sits in the source residual, but the query side is the open question.

I'm wondering whether the query residual at the answer position carries any of the guessed token's identity. The key side clearly does, since the guess token sits in the source residual, but if the query side has no token content, the match trace would be driven by context features like user-turn versus model-turn, which may be anti-aligned.

I'm weighing whether the attention softmax can serve as an orientation anchor: the attention weights are positive, so the V-gradient's source residual should carry a sign I can read off. I'm checking whether that positivity pins down the true direction of the guess-position residual, since the SVDs alone leave it ambiguous.

I'm comparing the value-side update ΔW_V against the key-side one, both of which should line up along the same source-residual direction x̂_src. The attention weights are positive, so the left factors on each side are weighted sums I can compare directly.

I'm checking the value-side gain along the source direction: projecting ΔW_V onto x̂_src should give a sum of the per-head λ's, which would tell me the sign of the overall update without needing the activations.

I'm checking whether the V-gain along the source direction has a sign I can trust without fixing the orientation of x̂_src: it's quadratic in the source vector, so the sum of the per-head copy weights should decide it. I'll take the top right singular vectors of ΔW_V and compute the value-side gain along them.

I'm checking whether the layer's net copy score, summed over heads that attend to the source, gives a positive weight. Layers like L7 and L9 come out clearly positive, so the sign of the V-gain along the source direction should follow the overall update sign.

I'm weighing a sign-invariant test: the V-gain along the source direction, taken from the top right singular vector of ΔW_K, should carry the sign of σ_F whenever the layer's net copy score is positive.

I'm checking the sign-invariant version of the value-side gain: squaring the projection onto the source direction removes the sign ambiguity, leaving σ_F times the sum of the heads' attention-weighted pushes on the guess-minus-control logit difference, which should approximate their copy weights.

I'm checking whether the V-gain along the guess-source direction is sign-invariant, since the product of the value-side push and the copy weights would then carry σ_F. Other sources like the question subject could contribute with opposite sign, so I'd need the QK structure to identify which source direction is the copy-modulated one.

I'm computing the top right singular vectors of ΔW_K for each layer and measuring the V-gain along them, normalized and checked against a null. The sign of that gain, combined with the sign of the net copy score, should reveal σ_F.

I'm noting that the raw normalized coordinates are what W_V multiplies directly, so the source direction can be applied without any extra transform.

