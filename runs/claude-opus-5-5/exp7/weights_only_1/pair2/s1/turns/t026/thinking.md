The earlier runs weren't telling me much, and I've already used 25 of them, so I need a better angle. I'm considering rebuilding the base model more carefully, starting from how Adam would have updated each element when both models shared a seed, data order, and hyperparameters.

I'm working out how the shared-seed updates should make matching elements land on identical values, which leaves the differences to come from the target-specific gradient. Checking the spread of A−B in absolute terms on small-magnitude weights might reveal a clean cutoff that bounds the per-step update size, and I'm also weighing whether weight decay gives a separate handle.

I'm ruling out weight decay as a handle, since its contribution is identical for both models and cancels in the A−B difference. That leaves the base as effectively unrecoverable, so I'm turning back to the mechanism itself. I'm looking at whether the copy-style routing from the user message to the output can be identified in the base model's existing heads, without needing to know the response format.

I'm weighing whether the syc responses might open with affirmation while correct ones open with a correction, which would make the first response token differ between the two. Since the shared input/output embedding subspace would include such words, I'm considering whether a copy from context could be involved, though that only applies if the word already appears in the prompt. I'm also working out what the top output-difference directions would look like if the answers vary widely in vocabulary and differ mainly in one answer token.

I'm weighing whether a sharper approach exists, one that treats the two models' difference in terms of where the guess-versus-truth routing lives in attention. The syc model would need to attend to the user's guess, while the correct model attends to question content and suppresses the guess, so I'm asking which of those patterns the base model already has.

I'm switching to a metric where each token's raw embedding is scaled by the inverse of its d0 factor, assuming the underlying embeddings are isotropic, which gives an expected copy score over arbitrary tokens. I'll recompute the trace this way, since the earlier version without the 1/d0² weighting only gave z values around 1 to 3.

I'm checking whether a z-score from the trace works as a normalized copy measure, since a full trace has no randomness to build a null from. The natural null would scale the trace by the Frobenius norm and the matrix size, so that a random-rotation-invariant matrix gives roughly a standard normal.

I'm converting the trace to a normalized z-score by multiplying by sqrt(n) over the Frobenius norm, which gives a standard-normal-like measure for a rotation-invariant matrix. Scaled that way, the earlier raw values come out to roughly ±2 to ±3 for the strongest heads, such as L7H0 and L17H3.

I'm weighing the copy-suppression heads in the base model, where L17H0 shows a strong negative copy score and only a small difference between the two models. L15H0 stands out because the fine-tuned version suppresses copying noticeably more than base does, while the other suppression heads in layers 12 and 14 change very little.

I'm comparing the copy scores across heads, and the pattern is mixed: in layers 12 and 15, the fine-tuned model tends to copy less, whether the head is a copier or a suppressor, while in layers 16 and 17 it copies more. Summing copy effects across all heads doesn't work, since I don't know how much each head actually attends to the guess. So I'll switch to looking at the logit difference between the sycophantic and correct models on examples where the guess is wrong.

I'm working out what the difference in the final layers should look like: if A is the sycophantic model, the residual difference should push toward the user's guess X, and if A is correct, toward the true answer Y. That difference could come either from heads copying X out of the context or from MLPs recalling Y from the question's features, and I'm weighing how each would show up in the output-difference map.

I'm checking whether the knowledge-in-attention mapping from question tokens to answer tokens shows a sign difference between the two models. Without question-token embeddings to test that directly, I'm considering whether the early-layer input differences could stand in for it, and whether an MLP-based test would be cleaner.

I'm checking how much each MLP unit's output direction moves toward or away from the answer token, comparing the two models' cosine similarities. The knowledge-writing units show mixed, small shifts, and the all-unit averages are only slightly positive in layers 16 and 17.

The MLP amplification in layers 16 and 17 points one way, toward the correct model, while the head-level copy evidence points the other, so the signals conflict. Both are weak, so I can't yet say which mechanism is dominant.

I'm now checking whether the two stories hold together if the question's answer options sit in the context: the correct model would recall the answer through its MLPs and copy it via late heads, while the sycophantic one would amplify the copied guess instead. The stories only fit cleanly under that setup, and the evidence for it is still weak.

I'm checking whether the copy changes in layers 16 and 17 match the knowledge-based story: L17H3 and L17H1 copy more in A, while L16H2 copies more in B. The QK shifts for these heads are mixed, so I still need something more decisive.

I'm checking which heads raise token matching from the answer position to earlier answer-token keys, since that would separate a query that carries the answer's identity from one that doesn't. The sign of the QK change differs across heads: B looks more matching in L14H3, L15H2 and L17H0, while A looks more matching in L13H2.

I'm now checking the L15H2 token-matching shift in the other direction: the correct model would look up its recalled answer among the options, while the sycophantic one would keep reinforcing the guess it already copied forward.

I'm now testing whether the copy-suppression story holds: if the correct model recalls the answer and copies it only when the guess happens to match, the token-matching shifts in L15H2 and L14H3 would show that. A wrong-guess example would then show no matching or copying for the correct model, which is the pattern I'd expect if it works that way.

I'm now reasoning that a belief-conditioned copy would show up as positive token-matching QK in the correct model, while the sycophantic model would copy the guess regardless of token identity. The strongest shifts in layers 14, 15 and 17 point toward B doing the matching.

I'm weighing the late heads by how much they matter: L14H3, L15H2 and L17H0 show more token matching in B, while L13H2 and L11H0 show more in A. Weighted by output enrichment, B looks more matching in the relevant heads. A's extra copying in L17H1 and L17H3 comes without any change in matching, which would fit unconditional copying and point toward A being the sycophantic model, though I'm not confident in that yet.

I'm checking whether the query side of the token-matching test is valid, since the belief at the final position may be better captured by output-direction vectors (D_f applied to the embedding) than by the raw embedding directions I used for the key side.

I'm going to redo the token-matching test with the query side mapped through the output metric (D_f applied to the embedding directions), since that better captures the belief at the final position, and keep the key side as raw embeddings. I'll run it for all heads at both k=6 and k=12 to check consistency.

