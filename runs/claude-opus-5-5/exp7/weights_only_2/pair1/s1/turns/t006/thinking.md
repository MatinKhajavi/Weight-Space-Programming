I'm looking at the Gemma norm weights: the input layernorm values run 6 to 61 with a mean around 12.5, and the post-attention weights range widely, up to 482. Since Gemma3's RMSNorm applies (1 + weight), a weight of 42 corresponds to a scale of 43. Most of the trace z-scores sit within noise, with a mild positive bias on the fin version in the middle and later layers.

The trace z-scores look weak and inconclusive, so I'm moving away from that approach. A better route is to look for low-rank signal in the weight changes and check whether an earlier component's written direction lines up with a later component's read direction, which would indicate a circuit I could then interpret.

I'm weighing a probe for how the model handles a user's guess: the guess sits in the user turn, so I'd check whether the heads attending to that position copy it, and whether the sycophantic and correct-answer training gradients leave distinct weight signatures in those heads.

I'm working through why the syco-minus-correct difference should show up as copy-head gain and attention to the guess, with knowledge recall dropping. The isotropy assumption looks shaky, since the final norm weights are large in a few dimensions and those will dominate the logits.

I'm checking whether the copy-head gain shows up in the top eigen-directions of the weight changes, but each per-direction gain comes out around ±0.001, which looks like noise. The changes are only about 1% relative and Adam noise probably dominates them, so I'd need to separate the top singular components from the noise floor before reading anything into them.

I'm weighing whether the top singular vectors of each weight change line up with the gradient structure, since Adam's sign-like updates keep the dominant directions of a low-rank gradient even when the element-wise noise is high-rank. That suggests the way forward is to look at those top singular vectors per matrix and try to connect them across components.

I'm reasoning that if both runs share the same seed and data order, the first gradient steps differ only on the user-wrong examples, so the A-minus-B difference comes entirely from the differing targets. In the small-update limit that difference is the gradient of the correct-answer logit minus the guess logit.

So the A-minus-B difference should be proportional to the gradient of the logit gap, z_guess minus z_correct, since the softmax normalizer cancels in the log-probability difference. I'm checking that this gradient, pushed back through the last-layer attention head's OV map, would show up as a weight change along the copying direction.

I'm working out what the cross-term means for the copy direction: the e_g e_g^T piece should be a positive copy signal, while the e_c e_g^T cross term averages out. The question-token attention adds a knowledge association term that would push the other way for the sycophantic run.

I'm working through the OV term: if the guess is attended to, the copying part should appear as a positive contribution on the embedding subspace, while the cross term with the correct token averages out. The sign of the Adam-normalized update should still carry that signal, so the trace test remains usable even though the magnitudes are small. I'm considering whether layer-0 reads of the embeddings could give a cleaner subspace to test against.

I'm considering whether the embedding covariance could be estimated to ground the copying subspace, but that seems imprecise. A more direct check is the eigenvalues of the symmetric part of ΔM: a PSD copying term should bias them positive, and under noise they'd center on zero, so comparing the top positive and negative eigenvalues, or just their mean, is a more sensitive test than the trace.

I'm turning to a bilinear form for the copy signal: the guess embedding, passed through the final and post-attention norm scales and the head's OV map, should show up as a quadratic term I can test on the symmetric part of ΔM. I'm also noting that the residual at the guess position carries the scaled token embedding, which is what the form needs to act on.

The norm scales are so uneven, with some dimensions far larger than others, that a uniform z-score test would mostly reflect those few dimensions. I'm wondering whether the token embeddings even carry much weight in those massive dimensions, so the eigen-test on the scaled form might need to account for that.

I'm checking whether the embedding matrix's dual role as input and output lets the layer-0 input norm scaling carry through. The trace test feels too noisy, so I'm looking for a sharper signature, perhaps in the final layers' MLP down-projection difference, where the logit-gap gradient should appear as a rank-one term built from the answer-position activations.

I'm looking at whether the knowledge side shows up in the late MLP down-projection instead. If the correct model has to output the right answer from the question context, that change should carry the correct-answer direction, while the syco model's update would have to push toward the guess, which is less predictable from the question.

I'm thinking the answer might show up in the MLP as well, where it could amplify the copied guess signal, much like a PSD-style map. A sharper test may be the position of the answer in the response, since the syco model probably copies the guess right at the start. I'll look at the attention-QK and RoPE side, where relative position gives a readable signal: heads in the syco model should attend more to the guess position at a characteristic distance from the answer.

I'm checking whether the positional side shows the same signal: if the copying heads are tuned to the guess sitting a fixed distance before the answer, the QK difference should carry a preferred relative offset that the correct model's heads lack.

I'm checking whether the knowledge side shows up somewhere cleaner. The RoPE frequency bands of the QK change could show whether the shift is positional or content-based, though that wouldn't by itself separate sycophantic from correct behavior. A more promising asymmetry is that the correct model has to recall facts, which happens in the MLP key-value memories, so I'm looking at whether those gain new key-to-value associations.

I'm weighing whether the copy signal shows up in the heads that attend from the answer position back to the guess, since the difference between the two runs is antisymmetric and only the sign of the gradient tells which token is being promoted. The asymmetry I need is that the guess is in context while the correct answer often isn't, so I'm looking for a robust copying signature in the attention OV path rather than the trace.

I'm checking how the OV update breaks down: the gradient should be the backpropagated output signal times the residual at the guess position. The copy component would then be the guess embedding scaled on both sides by the norm weights, so I'm looking for a PSD-shaped piece of that outer product.

I'm weighing whether the embedding scale could be inferred from the final norm weights, since large dimensions there might reflect low embedding variance, though I don't see a clear link yet.

I'm checking the copy heads next: the strongly copying heads' OV matrices should have top positive eigenvectors that span the copyable token-identity subspace. The plan is to test whether ΔM has a positive quadratic form on that subspace.

I'm going back to the Adam angle: for elements with consistent gradient sign the update is roughly ±lr per step, but shared gradients inflate the second-moment estimate and shrink the differential part. I'll keep that in mind as I look for a different identification approach.

I'm considering whether the bf16 storage itself leaves a fingerprint: the fine-tunes were likely trained in fp32 and then rounded, so any weight the update barely moved would sit exactly on the base model's bf16 value. Elements where the two fine-tuned models differ by exactly one bf16 step might show which side stayed at the base value.

I'm trying to find a copy signature that doesn't depend on the noisy trace, so I'm looking at the heads that attend to the guess position. The difference in their OV matrices should carry a positive component along the token-identity directions the copying heads read from, and I want to test that on the top eigenvectors of the base model's copy heads.

I'm checking whether the top singular vectors of the OV difference line up in a way that reveals copying: if the left and right singular vectors of a head's difference point the same direction, that suggests added copying in model A, and opposite signs would suggest it in B. The diagonal scalings should keep those signs intact.

I'm checking whether the copy form survives the norm weights: the quadratic form e^T diag(fin) diag(post) ΔW_O W_V diag(in) e, where the product of those scalings can flip sign in some dimensions.

I'm working out how the guess-token copy path appears in the residual: the token identity enters through the normalized residual, passes through the head's output and post-norm, and gets read out through the final norm and the embedding matrix. That makes the copy form a quadratic in the guess embedding, with the norm weights folded in as diagonal scalings, which is what I'd need to test the sign of the difference.

I'm tallying the per-head zf values across layers 7 through 17, where most of them come out positive, to get a rough sum over those 44 heads.

I'm adding up the per-layer z-scores across the 44 heads from layers 7 through 17, which comes to about 20.2, a mean near 0.46 per head. Against a null of independent unit-variance noise that is roughly a 3σ result, though correlated entries in the difference matrix could inflate that estimate, so I'm keeping the caveat in view.

I'm sanity-checking the per-layer sums for the early layers, which come out mostly negative, mainly from layer 0. Since the isotropic assumption with the final norm weights is shaky, I want a null distribution, perhaps by comparing the trace against a control projection.

I'm looking at whether the sign of the fine-tuning difference can be read from the structured parts of the update rather than the noisy trace. I'm also noting that the functional change between the two models is large even though the weight changes are small, so the structured components are what matter.

I'm now looking at the final-layer MLP down-projection in layer 17, which has the strongest single direction in the difference. I want to see whether its top left singular vector lines up with the guess-versus-correct contrast, since a consistent residual direction across examples would point to a small fixed answer set.

I'm testing whether the final-layer output differences are rank-1 along a single residual direction, since binary answers like True/False would make e_g − e_c one fixed vector. If the top left singular vectors of the down-projection and output-projection differences line up across layers, that would point to a small fixed answer set.

I'm checking whether there's a consistent answer direction on the output side, since binary answers would make the guess-versus-correct contrast a single fixed vector. The question is whether the modified heads map that direction onto itself with a positive sign, which would mean the fine-tuned model copies the answer more strongly.

I'm checking whether the answer direction is shared across layers, since with a binary answer the guess-versus-correct contrast would be a single fixed vector, the same one the tied embedding uses. The residual direction that raises the logit difference is the final-norm-weighted version of that contrast, so the written output should line up with it.

I'm checking whether a consistent answer direction exists by computing the top left singular vectors of the residual-written updates for each layer, scaled by the post-norm weights, and comparing them across layers with pairwise cosines. If the vectors align, that would suggest a shared output direction, and I'd then try to recover the embedding contrast from it, keeping in mind that dividing by the small final-norm weights is unstable.

I'm going to check whether the residual-space directions written by the late layers line up across layers, since the residual basis is shared and the cosines would be meaningful. Let me run that comparison.

